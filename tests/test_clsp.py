"""
Regression and acceptance tests for pyclsp 5.0.0.

The references are computed with numpy, scipy and cvxpy only. Runs under
pytest or as a plain script: `python tests/test_clsp.py`. With the
environment variable CLSP_TEST_INSTALLED=1, the installed package is tested
instead of the repo copy.
"""
import io
import os
import sys
import contextlib
import itertools
import traceback
import warnings
from   pathlib        import Path
import numpy          as np
import cvxpy          as cp
from   scipy.optimize import lsq_linear

ROOT      = Path(__file__).resolve().parents[1]
INSTALLED = os.environ.get("CLSP_TEST_INSTALLED") == "1"
if not INSTALLED:                                    # test the repo copy
    sys.path.insert(0, str(ROOT))
import clsp                                          # noqa: E402
import clsp.solver                                   # noqa: E402
from   clsp           import CLSP                    # noqa: E402
from   clsp.errors    import CLSPError               # noqa: E402

EPS      = np.finfo(float).eps
TOL      = 1e-8                                      # numerical tolerance
X_AP     = np.array([[.50, .25, .25],                # known AP case
                     [.25, 4.75, 5.0],
                     [.25, 5.0, 4.75]])
B_AP     = np.r_[X_AP.sum(1), X_AP.sum(0)]           # [1,10,10,1,10,10]
X_AP_NN  = np.array([[0.0, .50, .50],                # min-norm, x ≥ 0
                     [.50, 4.75, 4.75],
                     [.50, 4.75, 4.75]])
CLARABEL = dict(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12,
                tol_feas=1e-12, tol_ktratio=1e-10, max_iter=500)

# Ancillary functions (independent references)
def _ap_matrix(m, p):
    """Row sums, then column sums of an m×p matrix (row-major vec)."""
    return np.vstack([np.kron(np.eye(m), np.ones((1, p))),
                      np.kron(np.ones((1, m)), np.eye(p))])

def _min_norm(A, b):
    """Moore–Penrose minimum-norm least-squares solution."""
    return np.linalg.lstsq(A, b, rcond=None)[0]

def _tsvd(A, b, cutoff):
    """Truncated-SVD solution, s ≤ cutoff·s_max discarded."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    k        = s > cutoff * s[0]
    return Vt[k].T @ ((U[:, k].T @ b) / s[k])

def _svd_matrix(rng, n_row, n_col, s):
    """Random matrix with prescribed singular values s."""
    U = np.linalg.qr(rng.standard_normal((n_row, n_row)))[0]
    V = np.linalg.qr(rng.standard_normal((n_col, n_col)))[0]
    return U[:, :len(s)] @ np.diag(s) @ V[:, :len(s)].T

def _bounded_ls(A, b, lb, ub):
    """Bounded least squares (BVLS): z and the residual norm ‖b - Az‖."""
    eq     = lb == ub                                # BVLS needs lb < ub
    z      = np.where(eq, lb, 0.0)
    res    = lsq_linear(A[:, ~eq], b - A[:, eq] @ lb[eq],
                        bounds=(lb[~eq], ub[~eq]), method="bvls",
                        tol=1e-14, max_iter=10_000)
    z[~eq] = res.x
    return z, np.linalg.norm(b - A @ z)

def _cvx_closest(z0, A, w, lb, ub, l2=False):
    """argmin ‖z - z0‖² (‖z - z0‖₂ if l2) s.t. Az = w,
    lb ≤ z ≤ ub (CLARABEL, tight)."""
    z   = cp.Variable(A.shape[1])
    con = [A @ z == w]
    con += [z[i] >= lb[i] for i in np.flatnonzero(np.isfinite(lb))]
    con += [z[i] <= ub[i] for i in np.flatnonzero(np.isfinite(ub))]
    f   = cp.norm(z - z0, 2) if l2 else cp.sum_squares(z - z0)
    with warnings.catch_warnings():                  # status not needed
        warnings.simplefilter("ignore", UserWarning)
        cp.Problem(cp.Minimize(f), con).solve(**CLARABEL)
    return z.value

def _ap(**kw):
    """Known AP case, m = p = 3."""
    kw = dict(problem="ap", m=3, p=3, b=B_AP, final=False) | kw
    return CLSP().solve(**kw)

def _close(a, b, tol=TOL):
    return np.allclose(np.ravel(a), np.ravel(b), rtol=0.0, atol=tol)

# 0. Package under test
def test_imports_package_under_test():
    path = Path(clsp.__file__).resolve().parent
    assert (path != ROOT / "clsp") if INSTALLED else (path == ROOT / "clsp")
    assert clsp.__version__ == "5.0.0"

# 1. Canonical form
def test_canonical_form_general():
    rng   = np.random.default_rng(1)
    C, M  = rng.standard_normal((2, 4)), rng.standard_normal((3, 4))
    S     = np.array([[1.0], [-1.0]])
    b     = rng.standard_normal(5)
    c     = CLSP().solve(C=C, S=S, M=M, b=b, final=False)
    # Q empty for r = 1
    A_ref = np.block([[C, S], [M, np.zeros((3, 1))]])
    assert c.A.shape == (5, 5) and _close(c.A, A_ref, 0.0)
    assert tuple(c.C_idx) == (2, 4) and tuple(c.S_idx) == (4, 5)
    assert _close(c.b, b, 0.0)
    assert _close(c.x, c.z[:4], 0.0) and _close(c.y, c.z[4:], 0.0)

def test_canonical_form_ap():
    m, p = 2, 3
    X    = np.arange(1.0, 7.0).reshape(m, p)
    b    = np.concatenate([X.sum(1), X.sum(0)])
    c    = CLSP().solve(problem="ap", m=m, p=p, b=b, final=False)
    assert c.A.shape == (m + p, m * p)
    assert _close(c.A, _ap_matrix(m, p), 0.0)
    assert _close(c.A @ X.reshape(-1), b, 1e-12)     # row-major vec
    assert c.x.shape == (m, p)
    assert _close(c.x, c.z[:m * p].reshape(m, p), 0.0)

# 2. Step 1 without bounds: zhat = (A·Z)⁺ b
def test_step1_full_row_rank():
    rng = np.random.default_rng(2)
    C   = rng.standard_normal((2, 6))
    M   = rng.standard_normal((2, 6))
    b   = rng.standard_normal(4)
    c   = CLSP().solve(C=C, M=M, b=b, final=False)
    assert _close(c.zhat, _min_norm(np.vstack([C, M]), b))
    assert _close(c.A @ c.zhat, b)                   # consistent

def test_step1_rank_deficient_consistent():
    c = _ap()                                        # rank 5 of 6 rows
    A = _ap_matrix(3, 3)
    assert np.linalg.matrix_rank(A) == 5
    assert _close(c.zhat, _min_norm(A, B_AP))
    assert _close(c.A @ c.zhat, B_AP)
    assert abs(c.x[0, 0] + 5 / 3) < TOL

def test_step1_diagonal_Z_matches_old_formula():
    rng = np.random.default_rng(3)
    A   = rng.standard_normal((3, 6))
    b   = rng.standard_normal(3)
    Z   = np.diag([1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    old = np.linalg.pinv(Z @ A.T @ A @ Z) @ Z @ A.T @ b
    c   = CLSP().solve(C=A, b=b, Z=Z, final=False)
    assert _close(c.zhat, old)                       # 4.0.1 formula
    assert _close(c.zhat[[2, 5]], 0.0)               # outside range(Z)

def test_step1_projector_Z():
    rng = np.random.default_rng(4)
    A   = rng.standard_normal((3, 6))
    b   = rng.standard_normal(3)
    U   = np.linalg.qr(rng.standard_normal((6, 4)))[0]
    Z   = U @ U.T                                    # non-diagonal projector
    c   = CLSP().solve(C=A, b=b, Z=Z, final=False)
    assert _close(c.zhat, _min_norm(A @ Z, b))
    assert _close(Z @ c.zhat, c.zhat)                # zhat ∈ range(Z)

# 3. rcond
def test_rcond_default_cutoff():
    rng    = np.random.default_rng(5)
    A      = _svd_matrix(rng, 4, 6, [1.0, 1e-3, 1e-9])
    b      = rng.standard_normal(4)
    cutoff = np.sqrt(6 * EPS)
    assert 1e-9 < cutoff < 1e-3
    c      = CLSP().solve(C=A, b=b, final=False)
    assert _close(c.zhat, _tsvd(A, b, cutoff))
    assert np.linalg.norm(c.zhat) < 1e4              # 1/1e-9 not inverted

def test_rcond_float_truncates():
    rng = np.random.default_rng(6)
    A   = _svd_matrix(rng, 4, 6, [1.0, 0.6, 0.3, 0.1])
    b   = rng.standard_normal(4)
    c   = CLSP().solve(C=A, b=b, rcond=0.5, final=False)
    ref = _tsvd(A, b, 0.5)                           # keeps 1.0 and 0.6
    assert _close(c.zhat, ref)
    assert not _close(ref, _min_norm(A, b), 1e-3)    # truncation matters

# 4. Residual-sign iteration (r > 1)
def _consistent_general(seed, k=2, l=3, n=7):
    rng = np.random.default_rng(seed)
    C   = rng.standard_normal((k, n))
    M   = rng.standard_normal((l, n))
    x0  = rng.standard_normal(n)
    return C, M, np.concatenate([C @ x0, M @ x0])

def _sign_free_reference(C, M, b):
    """x = argmin ‖x‖² + ‖b_M - Mx‖² s.t. Cx = b_C (KKT system)."""
    k, n = C.shape
    K    = np.block([[np.eye(n) + M.T @ M, C.T], [C, np.zeros((k, k))]])
    rhs  = np.concatenate([M.T @ b[k:], b[:k]])
    return np.linalg.solve(K, rhs)[:n]

def test_r_consistent_equals_r1():
    C, M, b = _consistent_general(7)
    x1      = CLSP().solve(C=C, M=M, b=b, r=1, final=False).x
    for r in (2, 3, 5):
        c = CLSP().solve(C=C, M=M, b=b, r=r, final=False)
        assert _close(c.x, x1), f"r = {r}"
        assert _close(c.A[:, :7] @ c.x.ravel(), b)
    xb = CLSP().solve(problem="ap", m=3, p=3, M=np.eye(9)[[0]],
                      b=np.append(B_AP, 0.5), r=4, final=False).x
    assert _close(xb, CLSP().solve(problem="ap", m=3, p=3, M=np.eye(9)[[0]],
                                   b=np.append(B_AP, 0.5), final=False).x)

def test_r_signs_do_not_change_x():
    rng  = np.random.default_rng(8)
    C, M = rng.standard_normal((2, 5)), rng.standard_normal((4, 5))
    b    = rng.standard_normal(6)                    # inconsistent M rows
    ref  = _sign_free_reference(C, M, b)
    for s in ([1, 1, 1, 1], [-1, -1, -1, -1], [1, -1, 1, -1], [-1, 1, 1, 1]):
        c = CLSP()
        c.canonize("", C, None, M, np.diag(np.array(s, float)), b)
        c.solve(final=False)
        assert _close(c.x, ref), f"signs {s}"

def test_r_inconsistent_iterates_to_sign_free_x():
    rng  = np.random.default_rng(9)
    C, M = rng.standard_normal((2, 5)), rng.standard_normal((4, 5))
    b    = rng.standard_normal(6)
    c    = CLSP().solve(C=C, M=M, b=b, r=3, final=False)
    assert _close(c.x, _sign_free_reference(C, M, b))

# 5. Bounds
def test_bounds_invalid_raise():
    bad = [(1.0, 0.0), [(0, 1)] * 2, (np.nan, 1.0), (0.0, np.nan), "ab",
           "abc", [(0, 1, 2)], ("a", 1)]
    for bounds in bad:
        try:
            _ap(bounds=bounds)
        except CLSPError:
            continue
        raise AssertionError(f"no CLSPError for bounds={bounds!r}")

def test_bounds_non_diagonal_Z_raise():
    rng = np.random.default_rng(10)
    U   = np.linalg.qr(rng.standard_normal((9, 6)))[0]
    try:
        _ap(Z=U @ U.T, bounds=(0, None))
    except CLSPError:
        return
    raise AssertionError("no CLSPError for bounds with non-diagonal Z")

def test_bounds_ap_unbounded_reference():
    c = _ap()
    assert abs(c.x[0, 0] + 5 / 3) < TOL
    assert not c.fixed.any() and not c.fixed_at.any()

def test_bounds_ap_nonnegative():
    c   = _ap(bounds=(0, None))
    A   = _ap_matrix(3, 3)
    ref = _cvx_closest(np.zeros(9), A, B_AP, np.zeros(9), np.full(9, np.inf))
    assert _close(c.x, X_AP_NN)
    assert _close(ref, X_AP_NN, 1e-6)                # min-norm, z ≥ 0
    assert _close(c.A @ c.z, B_AP)
    assert c.fixed.dtype == bool and c.fixed.shape == (c.A.shape[1],)
    assert np.issubdtype(c.fixed_at.dtype, np.integer)
    assert c.fixed[0] and c.fixed_at[0] >= 1
    assert np.all(c.fixed_at[~c.fixed] == 0)
    assert np.all(c.fixed_at[c.fixed] > 0)
    assert _close(c.x_lb, 0.0, 0.0) and np.all(np.isinf(c.x_ub))

def test_bounds_ap_list_forms():
    x_pair = _ap(bounds=(0, None)).x
    for bounds in ([(0, None)], [(0, None)] * 9, [(0.0, np.inf)] * 9):
        assert _close(_ap(bounds=bounds).x, x_pair), f"bounds={bounds!r}"

def test_bounds_ap_single_variable():
    bounds = [(0, None)] + [(None, None)] * 8        # only x[0,0] ≥ 0
    c      = _ap(bounds=bounds)
    lb     = np.r_[0.0, np.full(8, -np.inf)]
    ref    = _cvx_closest(np.zeros(9), _ap_matrix(3, 3), B_AP, lb,
                          np.full(9, np.inf))
    assert c.x[0, 0] >= -TOL and c.fixed[0] and c.fixed.sum() == 1
    assert _close(c.x, ref, 1e-6)

def test_bounds_ap_upper():
    x_free = _ap().x
    c      = _ap(bounds=(None, 4.5))                 # inactive (max 13/3)
    assert np.max(c.x) <= 4.5 + TOL and _close(c.x, x_free)
    assert _close(c.A @ c.z, B_AP)
    c      = _ap(bounds=(None, 4.0))                 # active on x[1:,1:]
    assert np.max(c.x) <= 4.0 + TOL
    assert _close(c.A @ c.z, B_AP)                   # feasibility kept
    assert _close(c.x, [[-3, 2, 2], [2, 4, 4], [2, 4, 4]])
    assert c.fixed[:9].sum() == 4

# 6. Signed S-block slacks
def _slack_problem(seed):
    rng = np.random.default_rng(seed)
    C   = rng.standard_normal((3, 5))
    S   = np.diag([1.0, -1.0, 1.0])
    M   = rng.standard_normal((4, 5))
    b   = rng.standard_normal(7)
    return C, S, M, b

def test_slack_sign_step1_and_step2():
    cases = [(np.array([[1.0, 1.0]]), np.array([[1.0]]), np.eye(2),
              np.array([3.0, 2.0, 2.0]))]            # x1 + x2 ≤ 3, x = (2, 2)
    cases += [_slack_problem(s) for s in (11, 12)]
    for C, S, M, b in cases:
        n, k   = C.shape[1], S.shape[1]
        A      = np.block([[C, S], [M, np.zeros((M.shape[0], k))]])
        lb     = np.r_[np.full(n, -np.inf), np.zeros(k)]
        _, opt = _bounded_ls(A, b, lb, np.full(n + k, np.inf))
        free   = CLSP().solve(C=C, S=S, M=M, b=b, final=False)
        c      = CLSP().solve(C=C, S=S, M=M, b=b, final=False, slack_sign=True)
        assert np.all(c.z[n:n + k] >= -TOL)
        if np.any(free.z[n:n + k] < -1e-6):          # sign was binding
            assert c.fixed[n:n + k].any()
        for alpha in (0.0, 0.5, 1.0):
            c = CLSP().solve(C=C, S=S, M=M, b=b, alpha=alpha,
                             slack_sign=True)
            y = c.z[c.S_idx[0]:c.S_idx[1]]
            assert np.all(y >= -TOL), f"alpha = {alpha}"
            assert np.linalg.norm(b - c.A @ c.z.ravel()) <= opt + 1e-7

# 7. Step 2
def test_step2_consistent_unbounded():
    C, M, b = _consistent_general(13)
    for alpha in (0.0, 0.5, 1.0):
        c = CLSP().solve(C=C, M=M, b=b, alpha=alpha)
        assert _close(c.z, c.zhat, 1e-6), f"alpha = {alpha}"
        assert _close(c.A @ c.z, b, 1e-6)
        assert c.rho < 1e-8
    c = _ap(final=True, alpha=1.0)
    assert _close(c.z, c.zhat, 1e-6) and _close(c.A @ c.z, B_AP, 1e-6)

def test_step2_inconsistent_unbounded():
    rng  = np.random.default_rng(14)
    C, M = rng.standard_normal((2, 4)), rng.standard_normal((4, 4))
    b    = rng.standard_normal(6)
    A    = np.vstack([C, M])
    rmin = np.linalg.norm(b - A @ np.linalg.pinv(A) @ b)
    assert rmin > 1e-3                               # truly inconsistent
    for alpha in (0.0, 0.5, 1.0):
        c = CLSP().solve(C=C, M=M, b=b, alpha=alpha)
        r = np.linalg.norm(b - c.A @ c.z.ravel())
        assert abs(r - rmin) < TOL, f"alpha = {alpha}: {r} vs {rmin}"
        assert _close(c.z, c.zhat, 1e-6)
        assert abs(c.rho - rmin) < TOL
    assert CLSP().solve(C=C, M=M, b=b, final=False).rho is None

def _bounded_cases():
    """(kwargs, A, b, lb, ub) for inconsistent bounded problems."""
    # row and column totals differ
    b_ap      = np.array([1.0, 10.0, 10.0, 1.0, 10.0, 11.0])
    yield (dict(problem="ap", m=3, p=3, b=b_ap, bounds=(0, None)),
           _ap_matrix(3, 3), b_ap, np.zeros(9), np.full(9, np.inf))
    rng       = np.random.default_rng(15)
    C, M      = rng.standard_normal((3, 6)), rng.standard_normal((5, 6))
    b         = 3.0 * rng.standard_normal(8)
    bounds    = [(-0.2, 0.3), (None, 0.1), (0.0, None), (-1, 1),
                 (None, None), (0.05, 0.05)]
    lb        = np.array([-np.inf if lo is None else lo for lo, _ in bounds])
    ub        = np.array([ np.inf if hi is None else hi for _, hi in bounds])
    yield (dict(C=C, M=M, b=b, bounds=bounds), np.vstack([C, M]), b, lb, ub)

def test_step2_bounded_residual_and_bounds():
    for kw, A, b, lb, ub in _bounded_cases():
        _, opt = _bounded_ls(A, b, lb, ub)
        for alpha in (0.0, 0.5, 1.0):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = CLSP().solve(alpha=alpha, **kw)
            x = c.z[:A.shape[1], 0]
            assert np.all(x >= lb - TOL) and np.all(x <= ub + TOL)
            r = np.linalg.norm(b - c.A @ c.z.ravel())
            assert r <= opt + 1e-7, f"alpha = {alpha}: {r} > {opt}"
            assert abs(c.rho - opt) < 1e-7, f"rho {c.rho} vs {opt}"

def test_step2_ridge_is_projection():
    for kw, A, b, lb, ub in _bounded_cases():
        z_ls, _ = _bounded_ls(A, b, lb, ub)
        w       = A @ z_ls                           # unique fitted vector
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c   = CLSP().solve(alpha=1.0, **kw)
        ref     = _cvx_closest(c.zhat0.ravel(), A, w, lb, ub)
        err     = np.max(np.abs(c.z.ravel() - ref))
        assert err <= 1e-6, f"max |z - proj| = {err:.3e}"

# 8. Regression vs 4.0.1 without bounds: x = minimum-norm solution
def test_regression_min_norm_random():
    for seed in (101, 202, 303):
        C, M, b = _consistent_general(seed, k=3, l=2, n=8)
        A       = np.vstack([C, M])
        assert np.linalg.matrix_rank(M) == 2 and np.linalg.matrix_rank(A) == 5
        ref     = _min_norm(A, b)
        x0      = CLSP().solve(C=C, M=M, b=b, final=False).x
        x1      = CLSP().solve(C=C, M=M, b=b, alpha=1.0).x
        assert _close(x0, ref),       f"seed {seed}, final=False"
        assert _close(x1, ref, 1e-6), f"seed {seed}, final=True"

# 9. Diagnostics on a bounded AP problem
def test_diagnostics_bounded_ap():
    c   = CLSP().solve(problem="ap", m=3, p=3, b=B_AP, bounds=(0, None),
                       alpha=1.0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        c.summary(display=True)
    assert "Bounds (x)" in out.getvalue()
    assert isinstance(c.summary(), dict)
    corr = c.corr()
    assert len(corr["rmsa_i"]) == c.C_idx[0]
    t    = c.ttest(sample_size=10)
    assert {"p_one_left", "p_one_right", "p_two_sided"} <= set(t)

# 10. Audit regressions (one test per counterexample)
A_KKT = np.array([[1.0, -2.0], [0.0, 1.0]])          # x ≥ 0: x = [1, 0]
B_KKT = np.array([1.0, -1.0])

def _warns(func):
    """Result of func() and the messages of the RuntimeWarnings issued."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = func()
    return out, [str(v.message) for v in w
                 if issubclass(v.category, RuntimeWarning)]

def test_audit_fallback_warning_and_residual():
    c, msg = _warns(lambda: CLSP().solve(C=A_KKT, b=B_KKT, bounds=(0, None),
                                         alpha=1.0, solver="NOT_INSTALLED"))
    assert any("falling back" in m for m in msg)
    assert c.status["final"].startswith("fallback:")
    assert c.status["solver"] == "NOT_INSTALLED"
    assert _close(c.z, c.zhat, 0.0)
    assert abs(c.residual - np.linalg.norm(B_KKT - A_KKT @ c.z.ravel())) < TOL
    assert abs(c.residual - 1.0) < TOL and abs(c.rho - 1.0) < 1e-7
    c, msg = _warns(lambda: CLSP().solve(C=A_KKT, b=B_KKT, bounds=(0, None),
                                         alpha=1.0))
    assert not msg and c.status["final"] in {"optimal", "optimal_inaccurate"}
    assert c.status["rho"] == "bvls" and c.status["step1"] == "converged"
    assert c.status["eq_error"] <= c.status["tol_v"]
    assert abs(c.residual - c.rho) < 1e-7            # no fallback
    c = CLSP().solve(C=A_KKT, b=B_KKT, final=False)
    assert c.rho is None and c.status["final"] == "skipped"

def test_audit_b_replacement():
    c = CLSP().solve(C=np.eye(2), b=np.array([1.0, 2.0]), final=False,
                     rcond=0.5)
    c.solve(b=np.array([3.0, 4.0]))                  # final, rcond kept
    assert _close(c.b, [3.0, 4.0], 0.0) and _close(c.x, [3.0, 4.0])
    assert c.final is False and c.rcond == 0.5 and c.rho is None
    try:
        c.solve(b=np.array([1.0, 2.0, 3.0]))
    except CLSPError:
        assert _close(c.b, [3.0, 4.0], 0.0)          # b left unchanged
        return
    raise AssertionError("no CLSPError for a non-conformable b")

def test_audit_ttest_cache_key():
    c  = CLSP().solve(C=np.array([[1.0], [2.0], [3.0]]),
                      b=np.array([1.0, 4.0, 2.0]), final=False)
    c.ttest(sample_size=20, seed=123)
    s1 = list(c.nrmse_ttest)
    c.ttest(sample_size=20, seed=123)
    assert c.nrmse_ttest == s1                       # same key: cached
    c.ttest(sample_size=20, seed=999, simulate=True, distribution="uniform")
    assert c.nrmse_ttest != s1
    assert c.nrmse_ttest_key == (20, 999, "uniform", False, True)
    c.solve()                                        # invalidated by solve
    assert c.nrmse_ttest == [] and c.nrmse_ttest_key is None

def test_audit_corr_reset_after_refit():
    c = CLSP().solve(C=np.eye(2), b=np.array([1.0, 2.0]), alpha=1.0)
    c.corr()
    assert abs(c.rmsa) < TOL
    c.solve(C=np.array([[1.0, 0.0], [1.0, 1.0]]), b=np.array([1.0, 2.0]),
            alpha=1.0)
    assert c.rmsa is None and c.rmsa_i == []         # invalidated by solve
    c.corr()
    assert abs(c.rmsa - np.sqrt(0.5)) < TOL

def test_audit_partial_bootstrap_model_residual():
    c, _ = _warns(lambda: CLSP().solve(C=np.array([[1.0]]),
                                       M=np.array([[1.0], [1.0]]),
                                       b=np.array([0.0, 1.0, 2.0]), r=3,
                                       final=False))
    e_M  = np.array([1.0, 2.0]) - c.x.ravel()        # b_M - M·x
    assert abs(c.nrmse_partial - np.linalg.norm(e_M) / np.sqrt(2) /
               np.std([1.0, 2.0])) < TOL
    t, msg = _warns(lambda: c.ttest(sample_size=30, seed=123, partial=True))
    assert not msg and t["std_null"] > 0 and t["mean_null"] > 1.0
    assert np.isfinite(t["p_two_sided"])
    # zero residual
    c      = CLSP().solve(C=np.eye(2), b=np.array([1.0, 2.0]), final=False)
    t, msg = _warns(lambda: c.ttest(sample_size=10, seed=1))
    assert any("zero variance" in m for m in msg)
    assert np.isnan(t["p_two_sided"]) and np.isnan(t["p_one_left"])
    for n in (1, 0, 2.5):
        try:
            c.ttest(sample_size=n)
        except CLSPError:
            continue
        raise AssertionError(f"no CLSPError for sample_size={n!r}")

def test_audit_band_ordering():
    c = CLSP().solve(C=np.array([[1.0], [1.0]]), b=np.array([-1.0, -3.0]),
                     final=False)
    assert abs(c.x[0, 0] + 2.0) < TOL
    assert np.all(c.z_lower <= c.z) and np.all(c.z <= c.z_upper)
    assert _close(c.z_upper - c.z, c.z - c.z_lower)

def test_audit_Z_bounds_conflict():
    kw = dict(C=np.eye(2), b=np.array([1.0, 2.0]), Z=np.diag([1.0, 0.0]),
              final=False)
    for bounds in ([(None, None), (1.0, None)], [(None, None), (None, -1.0)]):
        try:
            CLSP().solve(bounds=bounds, **kw)
        except CLSPError:
            continue
        raise AssertionError(f"no CLSPError for bounds={bounds!r}")
    c = CLSP().solve(bounds=[(None, None), (-1.0, 1.0)], **kw)
    assert _close(c.x, [1.0, 0.0])                   # Z-excluded = 0

def test_audit_kkt_release_2x2():
    c, msg = _warns(lambda: CLSP().solve(C=A_KKT, b=B_KKT, bounds=(0, None),
                                         final=False))
    assert not msg
    assert _close(c.x, [1.0, 0.0]) and abs(c.residual - 1.0) < TOL
    assert list(c.fixed) == [False, True]            # x[0] released
    assert c.fixed_at[0] == 0 and c.fixed_at[1] > 0

def test_audit_bvls_random_100():
    for seed in range(100):
        rng    = np.random.default_rng(seed)
        D, y   = rng.normal(size=(4, 3)), rng.normal(size=4)
        _, opt = _bounded_ls(D, y, np.zeros(3), np.full(3, np.inf))
        c, msg = _warns(lambda: CLSP().solve(C=D, b=y, bounds=(0, None),
                                             final=False))
        gap    = np.linalg.norm(y - D @ c.x.ravel()) - opt
        assert not msg and np.all(c.x >= 0.0), f"seed {seed}"
        assert gap <= 1e-8, f"seed {seed}: residual gap {gap:.3e}"

def test_audit_min_norm_anchor():
    rng = np.random.default_rng(1)                   # audit follow-up case
    A   = rng.normal(size=(2, 4))
    b   = rng.normal(size=2)
    c   = CLSP().solve(C=A, b=b, bounds=(0, None), alpha=1.0)
    _, opt = _bounded_ls(A, b, np.zeros(4), np.full(4, np.inf))
    assert abs(c.residual - opt) < 1e-7
    ref = _cvx_closest(np.zeros(4), A, A @ c.z.ravel(), np.zeros(4),
                       np.full(4, np.inf))           # min ‖z‖², Az = w*
    assert _close(c.z, ref, 1e-6)
    assert _close(c.zhat0, _min_norm(A, b))          # unbounded anchor

def test_audit_alpha_inputs():
    kw = dict(C=A_KKT, b=B_KKT, bounds=(0, None))    # zhat0 infeasible
    c  = CLSP().solve(**kw)                          # None: α = 1 (Ridge)
    assert c.alpha == 1.0 and c.status["solver"] == str(cp.OSQP)
    for alpha in ((0.0, 1.0), np.array([0.0, 1.0]), [0.0, 0.5, 1.0], [0.5]):
        try:
            CLSP().solve(alpha=alpha, **kw)
        except CLSPError as e:
            assert "use cv_alpha()" in str(e)        # no NRMSE selection
            continue
        raise AssertionError(f"no CLSPError for alpha={alpha!r}")
    for alpha in (np.nan, np.inf, "0.5", -0.1, 1.5, [0.0, np.nan], ("a",)):
        try:
            CLSP().solve(alpha=alpha, **kw)
        except CLSPError:
            continue
        raise AssertionError(f"no CLSPError for alpha={alpha!r}")
    c = CLSP().solve(alpha=1e-9, **kw)               # exact endpoints
    assert c.status["solver"] == str(cp.CLARABEL)    # Elastic Net, not L1

# 11. One numerical-rank policy: A_k in Step 1, w*, Step 2 and the KKT check
def test_rank_policy_unbounded_pairs():
    rng   = np.random.default_rng(16)
    cases = [(np.diag([1.0, 0.1]), np.array([1.0, 1.0]), dict(rcond=0.5)),
             (np.diag([1.0, 1e-9]), np.array([1.0, 2.0]), dict()),
             (_svd_matrix(rng, 4, 6, [1.0, 0.6, 1e-10]),
              rng.standard_normal(4), dict())]
    for A, b, kw in cases:                           # identical results
        for final in (False, True):
            c = CLSP().solve(C=A, b=b, alpha=1.0, final=final, **kw)
            for bounds in ((None, None), [(None, None)] * A.shape[1]):
                d = CLSP().solve(C=A, b=b, alpha=1.0, final=final,
                                 bounds=bounds, **kw)
                assert np.array_equal(c.z, d.z) and c.rho == d.rho
                assert c.residual == d.residual and c.status == d.status
    for bounds in (None, (None, None)):              # audit cases
        c = CLSP().solve(C=np.diag([1.0, 0.1]), b=np.array([1.0, 1.0]),
                         rcond=0.5, alpha=1.0, bounds=bounds)
        assert _close(c.x, [1.0, 0.0]) and abs(c.rho - 1.0) < TOL
        assert c.rank == 1 and _close(c.sv_discarded, [0.1], 0.0)
        c = CLSP().solve(C=np.diag([1.0, 1e-9]), b=np.array([1.0, 2.0]),
                         alpha=1.0, bounds=bounds)
        assert _close(c.x, [1.0, 0.0]) and abs(c.rho - 2.0) < TOL
        assert abs(c.residual - 2.0) < TOL           # with the original A
        assert c.rank == 1 and _close(c.sv_discarded, [1e-9], 0.0)
    c = _ap()
    assert c.rank == 5 and c.sv_discarded.shape == (1,)
    assert CLSP().solve(C=np.eye(2), b=np.ones(2)).sv_discarded.size == 0

def test_rank_policy_bounded_kkt():
    c, msg = _warns(lambda: CLSP().solve(C=np.diag([1.0, 0.1]),
                                         b=np.array([1.0, 1.0]), rcond=0.5,
                                         bounds=(0, None), alpha=1.0))
    A_k    = np.diag([1.0, 0.0])                     # σ = 0.1 discarded
    g      = A_k.T @ (A_k @ c.z.ravel() - c.b.ravel())
    assert not msg and _close(c.x, [1.0, 0.0]) and _close(g, 0.0)
    assert c.status["step1"] == "converged" and abs(c.rho - 1.0) < TOL

# 12. Step 1: bounded-variable least squares
def _bvls_case(kind, seed):
    """Audit regimes: A, b and the box (lo, hi)."""
    rng    = np.random.default_rng(seed)
    A      = (rng.normal(size=(6, 4))         if kind == "over_box"        else
              rng.normal(size=(3, 6))         if kind.startswith("under")  else
              rng.normal(size=(5, 3)) @ rng.normal(size=(3, 5)))
    b      = rng.normal(size=A.shape[0])
    lo, hi = (0.0, np.inf) if kind == "under_nonnegative" else (-0.3, 0.7)
    return A, b, lo, hi

def _bvls_gap(kind, seed):
    """Residual gap to BVLS, bound violation, fit and warnings."""
    A, b, lo, hi = _bvls_case(kind, seed)
    n            = A.shape[1]
    _, opt       = _bounded_ls(A, b, np.full(n, lo), np.full(n, hi))
    c, msg       = _warns(lambda: CLSP().solve(C=A, b=b, bounds=(lo, hi),
                                               final=False))
    z            = c.z.ravel()
    viol         = max(0.0, np.max(lo - z), np.max(z - hi))
    return np.linalg.norm(b - A @ z) - opt, viol, c, msg

def test_bvls_audit_cycles():
    for kind, seed in (("under_box", 1166), ("rank_deficient_box", 1122),
                       ("rank_deficient_box", 1171)):
        gap, viol, c, msg = _bvls_gap(kind, seed)
        assert not msg and c.status["step1"] == "converged", f"seed {seed}"
        assert gap <= 1e-8 and viol <= 1e-9, f"seed {seed}: gap {gap:.3e}"
        assert c.status["bound_error"] == 0.0

def test_bvls_benchmark_50():
    for kind in ("over_box", "under_nonnegative", "under_box",
                 "rank_deficient_box"):
        for seed in range(1000, 1050):
            gap, viol, c, msg = _bvls_gap(kind, seed)
            assert not msg, f"{kind}, seed {seed}: {msg}"
            assert gap <= 1e-8 and viol <= 1e-9, f"{kind}, seed {seed}"

def test_bvls_iteration_limit_feasible():
    kw     = dict(C=A_KKT, b=B_KKT, bounds=(0, None), iteration_limit=1,
                  final=False)
    c, msg = _warns(lambda: CLSP().solve(**kw))      # release needs pass 2
    assert c.status["step1"] == "bvls fallback"
    assert any("scipy BVLS" in m for m in msg) and _close(c.x, [1.0, 0.0])
    lsq    = clsp.solver.lsq_linear                  # fallback unavailable
    clsp.solver.lsq_linear = lambda *a, **k: (_ for _ in ()).throw(
                             ValueError("disabled"))
    try:
        c, msg = _warns(lambda: CLSP().solve(**kw))
    finally:
        clsp.solver.lsq_linear = lsq
    assert c.status["step1"].startswith("not converged")
    assert any("last feasible" in m for m in msg)
    assert np.all(c.x >= 0.0) and c.status["bound_error"] == 0.0
    c = CLSP().solve(C=np.eye(2), b=np.array([-1.0, 2.0]), bounds=(0, None),
                     iteration_limit=1, final=False)
    assert c.status["step1"] == "converged" and _close(c.x, [0.0, 2.0])

# 13. Repeated fits
def test_refit_r3_then_r1():
    C, M, b = np.array([[1.0]]), np.array([[1.0], [1.0]]), np.array([0., 1, 2])
    c       = CLSP().solve(C=C, M=M, b=b, r=3, final=False)
    assert c.A.shape == (3, 3)                       # with Q
    c.solve(b=b, r=1)
    d       = CLSP().solve(C=C, M=M, b=b, r=1, final=False)
    assert c.A.shape == d.A.shape == (3, 1)
    assert _close(c.x, d.x, 0.0) and _close(c.x, [1.0])
    c.solve(r=3)
    assert _close(c.x, CLSP().solve(C=C, M=M, b=b, r=3, final=False).x, 0.0)

def test_refit_ap_shape():
    c = CLSP().solve(problem="ap", m=2, p=3, b=np.array([3.0, 6, 3, 3, 3]),
                     final=False)
    b = np.array([6.0, 12, 6, 6, 6])
    c.solve(b=b)
    d = CLSP().solve(problem="ap", m=2, p=3, b=b, final=False)
    assert c.x.shape == c.x_lower.shape == (2, 3) and _close(c.x, d.x, 0.0)

def test_refit_without_M_rows():
    c = CLSP().solve(C=np.ones((1, 1)), M=np.ones((2, 1)),
                     b=np.array([0.0, 1.0, 2.0]), final=False)
    assert np.isfinite(c.nrmse_partial) and np.isfinite(c.r2_partial)
    c.solve(C=np.eye(2), b=np.array([1.0, 2.0]), Z=np.eye(2))
    assert np.isnan(c.nrmse_partial) and np.isnan(c.r2_partial)
    assert c.A.shape == (2, 2) and c.spec["M"] is None

def test_refit_diagnostics_use_base():
    C, M, b = _consistent_general(17)
    b       = b + np.r_[0.0, 0.0, 0.3, -0.2, 0.1]    # inconsistent M rows
    c       = CLSP().solve(C=C, M=M, b=b, r=3, final=False)
    d       = CLSP().solve(C=C[1:], M=M, b=b[1:], r=3, final=False)
    assert abs(c.corr()["rmsa_dnrmse"][0] - (d.nrmse - c.nrmse)) < TOL
    c.ttest(sample_size=5, seed=1, simulate=True)    # refits with r = 3
    assert len(c.nrmse_ttest) == 5 and c.A.shape == (5, 10)

# 14. cv_alpha()
def _cv_ap(seed):
    """AP 4×4, x ≥ 0, with 8 known cells observed with noise (M rows)."""
    rng   = np.random.default_rng(seed)
    X     = rng.uniform(0.0, 3.0, size=(4, 4))
    X[rng.random((4, 4)) < 0.3] = 0.0
    cells = rng.choice(16, size=8, replace=False)
    b     = np.r_[X.sum(1), X.sum(0),
                  X.ravel()[cells] + 0.3 * rng.normal(size=8)]
    return CLSP().solve(problem="ap", m=4, p=4, M=np.eye(16)[cells], b=b,
                        bounds=(0, None))

def test_cv_alpha_uninformative():
    C, M, b = _consistent_general(18, k=2, l=6, n=7)
    cases   = [(CLSP().solve(C=C, M=M, b=b), "no bounds"),
               (CLSP().solve(C=C, b=b[:2], bounds=(0, None)), "no M rows"),
               (CLSP().solve(C=C, M=M[:3], b=b[:5], bounds=(0, None)),
                "fewer M rows"),
               (CLSP().solve(C=C, M=M, b=b, bounds=(-9, 9), final=False),
                "final is False")]
    for c, reason in cases:
        z      = c.z.copy()
        r, msg = _warns(lambda: c.cv_alpha(seed=1, refit=True))
        assert not r["informative"] and r["best"] == 1.0
        assert r["reason"].startswith(reason) and msg
        assert np.all(np.isnan(r["cv_rmse_mean"]))   # no refits
        assert np.array_equal(c.z, z) and c.cv_alpha_result is r
    c = _cv_ap(0)
    c.solve(bounds=(-100, 100))                      # inactive bounds
    r, _ = _warns(lambda: c.cv_alpha(alphas=(0, 1), folds=4, seed=1))
    assert not r["informative"] and r["best"] == 1.0
    assert r["reason"] == "all α give the same predictions"
    for kw in (dict(alphas=[0.5, 2.0]), dict(alphas=[]), dict(folds=1)):
        try:
            c.cv_alpha(**kw)
        except CLSPError:
            continue
        raise AssertionError(f"no CLSPError for {kw!r}")

def test_cv_alpha_informative():
    c, msg = _warns(lambda: _cv_ap(0))
    r      = c.cv_alpha(folds=4, seed=1)
    assert not msg and r["informative"] and r["alpha"] == [0, .25, .5, .75, 1]
    assert np.ptp(r["cv_rmse_mean"]) > 1e-3          # α changes predictions
    assert r["best"] == r["alpha"][int(np.argmin(r["cv_rmse_mean"]))]
    assert c.cv_alpha(folds=4, seed=1) == r          # deterministic (seed)
    c.cv_alpha(folds=4, seed=1, refit=True)
    assert c.alpha == r["best"] and c.cv_alpha_result == r
    c.solve()                                        # invalidated by solve
    assert c.cv_alpha_result is None and c.alpha == 1.0

# 14. Reduced free-column problems, KKT verification, Step 2 accuracy and
#     failure reporting of the diagnostic refits
A_RED = np.array([[1.0, 1e-8], [0.0, 0.0]])          # σ = 1e-8 after fixing
B_RED = np.array([1.0, 0.0])

def _inject(fail):
    """Patch cvxpy's Problem.solve to raise SolverError where fail(...)."""
    solve = cp.Problem.solve
    def patched(self, *args, **kwargs):
        if fail(self, kwargs.get("solver")):
            raise cp.SolverError("injected failure")
        return solve(self, *args, **kwargs)
    return patched

def _patched(obj, name, value, func):
    """func() with obj.name temporarily replaced by value."""
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        return func()
    finally:
        setattr(obj, name, old)

def test_bvls_reduced_columns_keep_A_k():
    kw     = dict(C=A_RED, b=B_RED, bounds=[(0, 0), (0, None)])
    c, msg = _warns(lambda: CLSP().solve(tolerance=1e-12, final=False, **kw))
    assert not msg and c.status["step1"] == "converged"
    assert _close(c.x.ravel() / [1.0, 1e8], [0.0, 1.0])
    assert c.residual < TOL and c.status["kkt_error"] <= 1e-12
    for eps in (1e-6, 1e-8, 1e-9):                   # default tolerance
        c = CLSP().solve(C=np.array([[1.0, eps]]), b=np.array([1.0]),
                         bounds=[(0, 0), (0, None)], final=False)
        assert c.status["step1"] == "converged" and c.residual < TOL
        assert abs(c.x[1, 0] * eps - 1.0) < TOL
    c, msg = _warns(lambda: CLSP().solve(tolerance=1e-12, final=True, **kw))
    assert c.rho < TOL and c.residual < TOL          # target: exact fit
    if c.status["final"].startswith("fallback"):
        assert msg and _close(c.z, c.zhat, 0.0)      # explicit fallback
    else:
        assert c.status["final"] in {"optimal", "optimal_inaccurate"}

def test_bvls_kkt_verification_fallback():
    kw     = dict(C=A_KKT, b=B_KKT, bounds=(0, None), final=False)
    zeros  = lambda a, **k: np.zeros(a.shape[::-1])  # wrong reduced solves
    c, msg = _patched(clsp.solver.la, "pinv", zeros,
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert c.status["step1"] == "bvls fallback" and _close(c.x, [1.0, 0.0])
    assert any("KKT error" in m and "scipy BVLS" in m for m in msg)
    assert c.status["kkt_error"] <= c.tolerance
    off    = lambda *a, **k: (_ for _ in ()).throw(ValueError("disabled"))
    c, msg = _patched(clsp.solver, "lsq_linear", off,
                      lambda: _patched(clsp.solver.la, "pinv", zeros,
                      lambda: _warns(lambda: CLSP().solve(**kw))))
    assert c.status["step1"].startswith("not converged: KKT error")
    assert any("last feasible" in m for m in msg)
    assert c.status["kkt_error"] > c.tolerance and np.all(c.x >= 0.0)
    c = CLSP().solve(C=A_KKT, b=B_KKT, final=False)  # unbounded: recorded
    assert c.status["kkt_error"] < TOL

def test_step2_ridge_accuracy_and_retry():
    for kind, seed in (("under_nonnegative", 1021), ("under_nonnegative",
                       1027), ("under_box", 1100), ("under_box", 1180),
                       ("rank_deficient_box", 1129)):
        A, b, lo, hi = _bvls_case(kind, seed)        # former gaps, fallbacks
        _, opt       = _bounded_ls(A, b, np.full(A.shape[1], lo),
                                   np.full(A.shape[1], hi))
        c, msg       = _warns(lambda: CLSP().solve(C=A, b=b, bounds=(lo, hi),
                                                   alpha=1.0))
        gap          = np.linalg.norm(b - A @ c.z.ravel()) - opt
        assert not msg and c.status["final"] == "optimal", f"seed {seed}"
        assert gap <= 1e-8, f"{kind}, seed {seed}: gap {gap:.3e}"
        assert c.status["solver"] in {"OSQP", "CLARABEL"}
    kw     = dict(C=A_KKT, b=B_KKT, bounds=(0, None), alpha=1.0)
    c      = CLSP().solve(**kw)
    assert c.status["solver"] == "OSQP" and c.status["retry"] is None
    c, msg = _patched(cp.Problem, "solve",
                      _inject(lambda p, s: s == cp.OSQP),
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert not msg and c.status["final"] == "optimal"
    assert c.status["solver"] == "CLARABEL"
    assert c.status["retry"].startswith("OSQP: SolverError")
    assert abs(c.residual - 1.0) < TOL and _close(c.x, [1.0, 0.0], 1e-7)
    c, msg = _patched(cp.Problem, "solve", _inject(lambda p, s: True),
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert msg and c.status["final"].startswith("fallback")
    assert c.status["solver"] == "CLARABEL" and c.status["retry"]
    c, _   = _warns(lambda: CLSP().solve(solver="NOT_INSTALLED", **kw))
    assert c.status["retry"] is None                 # user solver: no retry

def test_cv_alpha_failed_candidate():
    ref    = _cv_ap(0).cv_alpha(folds=4, seed=1)
    assert all(ref["valid"]) and ref["failures"] == []
    l1     = _inject(lambda p, s: isinstance(p.objective.expr, cp.norm1))
    r, msg = _patched(cp.Problem, "solve", l1,
                      lambda: _warns(lambda: _cv_ap(0).cv_alpha(folds=4,
                                                                seed=1)))
    assert r["valid"] == [False, True, True, True, True]
    assert [(v["alpha"], v["fold"]) for v in r["failures"]] == [
           (0.0, f) for f in range(4)]
    assert all(v["reason"].startswith("final: fallback") for v in
               r["failures"])
    assert any("α = [0.0] invalid" in m for m in msg)
    assert np.isnan(r["cv_rmse_mean"][0]) and np.isnan(r["cv_rmse_sd"][0])
    assert (r["cv_rmse_mean"][1:] ==                 # no contamination
            ref["cv_rmse_mean"][1:])
    assert r["informative"] and r["best"] == ref["best"]
    ecos   = _inject(lambda p, s: s == cp.ECOS)      # retried with CLARABEL
    r, msg = _patched(cp.Problem, "solve", ecos,
                      lambda: _warns(lambda: _cv_ap(0).cv_alpha(folds=4,
                                                                seed=1)))
    assert all(r["valid"]) and not r["failures"] and not msg
    c      = _cv_ap(0)
    r, msg = _patched(cp.Problem, "solve", l1,
                      lambda: _warns(lambda: c.cv_alpha(alphas=(0, 1),
                                                        folds=4, seed=1)))
    assert not r["informative"] and r["reason"] == "one valid α"
    assert r["best"] == 1.0 and len(msg) == 2
    r, msg = _patched(cp.Problem, "solve", _inject(lambda p, s: True),
                      lambda: _warns(lambda: c.cv_alpha(folds=4, seed=1,
                                                        refit=True)))
    assert not any(r["valid"]) and len(r["failures"]) == 20
    assert not r["informative"] and r["reason"] == "no valid α"
    assert np.isnan(r["best"]) and np.all(np.isnan(r["cv_rmse_mean"]))
    assert any("not informative" in m for m in msg) and c.alpha == 1.0

def test_corr_ttest_failure_summary():
    C, M, b = _consistent_general(19, k=3, l=4, n=5)
    c       = CLSP().solve(C=C, M=M, b=b,            # zhat0 infeasible:
                           bounds=(-0.25, 0.25))     # Step 2 is solved
    _, msg  = _warns(lambda: (c.corr(), c.ttest(sample_size=5, seed=1,
                                                simulate=True)))
    assert not msg and c.rmsa_failures == 0 and c.nrmse_ttest_failures == 0
    fail    = _inject(lambda p, s: True)
    corr, m = _patched(cp.Problem, "solve", fail,
                       lambda: _warns(lambda: c.corr(reset=True)))
    assert len(m) == 1 and m[0].startswith("corr(): 3 of 3 refits failed")
    assert c.rmsa_failures == 3 and len(corr["rmsa_i"]) == 3
    t, m    = _patched(cp.Problem, "solve", fail,
                       lambda: _warns(lambda: c.ttest(sample_size=5, seed=1,
                                                      simulate=True,
                                                      reset=True)))
    assert len(m) == 1 and m[0].startswith("ttest(): 5 of 5 refits failed")
    assert t["failures"] == 5 and c.nrmse_ttest_failures == 5
    assert c.ttest(sample_size=5, seed=1, simulate=True)["failures"] == 5
    t       = c.ttest(sample_size=5, seed=1)         # bootstrap
    assert t["failures"] == 0
    c.solve()
    assert c.rmsa_failures == 0 and c.nrmse_ttest_failures == 0

# 15. Column scaling, retry notes and the cv_alpha() solver log
def test_scale_reduced_columns():
    kw     = dict(C=A_RED, b=B_RED, bounds=[(0, 0), (0, None)])
    c, msg = _warns(lambda: CLSP().solve(**kw))      # σ = 1e-8: fallback
    assert msg and c.status["final"].startswith("fallback")
    assert c.status["scale"] is False and not c.scale
    c, msg = _warns(lambda: CLSP().solve(scale=True, **kw))
    assert not msg and c.status["final"] in {"optimal", "optimal_inaccurate"}
    assert c.status["scale"] is True and c.rank == 1
    assert _close(c.x.ravel() / [1.0, 1e8], [0.0, 1.0])
    assert c.residual < TOL and c.rho < TOL
    c.solve(b=B_RED)                                 # stored setting
    assert c.scale and c.status["scale"]
    c, msg = _warns(lambda: c.solve(scale=False))
    assert msg and c.status["scale"] is False
    rng = np.random.default_rng(10)
    U   = np.linalg.qr(rng.standard_normal((9, 6)))[0]
    try:
        _ap(Z=U @ U.T, scale=True)
    except CLSPError:
        return
    raise AssertionError("no CLSPError for scale with non-diagonal Z")

def test_scale_bounded_ap():
    for kw, A, b, lb, ub in _bounded_cases():
        c0, _  = _warns(lambda: CLSP().solve(**kw))
        c1, m1 = _warns(lambda: CLSP().solve(scale=True, **kw))
        x      = c1.z[:A.shape[1], 0]
        assert not m1 and abs(c1.residual - c0.residual) <= 1e-8
        assert np.all(x >= lb) and np.all(x <= ub)
        assert c1.kappaA == c0.kappaA                # original A
    c0 = CLSP().solve(problem="ap", m=3, p=3, b=B_AP, bounds=(0, None))
    c1 = CLSP().solve(problem="ap", m=3, p=3, b=B_AP, bounds=(0, None),
                      scale=True)                    # equal column norms
    assert _close(c1.x, c0.x, 1e-7) and np.all(c1.x >= 0.0)
    assert _close(c1.zhat0, c0.zhat0) and c1.rank == c0.rank

def test_step2_retry_note():
    kw     = dict(C=A_KKT, b=B_KKT, bounds=(0, None), alpha=1.0)
    c, msg = _patched(cp.Problem, "solve",
                      _inject(lambda p, s: s == cp.OSQP),
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert not msg and c.status["final"] == "optimal"
    assert c.status["retry"].startswith("OSQP: SolverError")
    out    = io.StringIO()
    with contextlib.redirect_stdout(out):
        c.summary(display=True)
    assert "Step 2 retry" in out.getvalue()
    assert "OSQP: SolverError" in out.getvalue()
    out    = io.StringIO()
    with contextlib.redirect_stdout(out):
        CLSP().solve(**kw).summary(display=True)
    assert "Step 2 retry" not in out.getvalue()
    solve  = cp.Problem.solve
    def inaccurate(self, *args, **kwargs):           # OSQP inaccurate,
        if kwargs.get("solver") == cp.CLARABEL:      # CLARABEL fails
            raise cp.SolverError("injected failure")
        out          = solve(self, *args, **kwargs)
        self._status = cp.OPTIMAL_INACCURATE
        return out
    c, msg = _patched(cp.Problem, "solve", inaccurate,
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert not msg and c.status["final"] == "optimal_inaccurate"
    assert c.status["solver"] == "OSQP"
    assert c.status["retry"] == "OSQP: status optimal_inaccurate"
    assert _close(c.x, [1.0, 0.0], 1e-7)

def test_cv_alpha_fits_log():
    ref    = _cv_ap(0).cv_alpha(folds=4, seed=1)
    fits   = ref["fits"]
    assert len(fits) == len(ref["alpha"]) * 4
    assert [(v["alpha"], v["fold"]) for v in fits] == [
           (a, f) for f in range(4) for a in ref["alpha"]]
    assert all(v["solver"] in {"ECOS", "OSQP", "CLARABEL"} and v["valid"]
               for v in fits)
    assert {v["solver"] for v in fits if v["alpha"] == 0} == {"ECOS"}
    ecos   = _inject(lambda p, s: s == cp.ECOS)      # rescued by CLARABEL
    r, msg = _patched(cp.Problem, "solve", ecos,
                      lambda: _warns(lambda: _cv_ap(0).cv_alpha(folds=4,
                                                                seed=1)))
    assert not msg and all(r["valid"]) and len(r["fits"]) == len(fits)
    for v, w in zip(r["fits"], fits):
        if v["alpha"] == 0:
            assert v["solver"] == "CLARABEL" and v["valid"]
            assert v["retry"].startswith("ECOS: final: fallback")
        else:
            assert v == w
    c      = _cv_ap(0)
    assert c.cv_alpha(folds=4, seed=1)["fits"] == fits
    c.solve(final=False)
    r, msg = _warns(lambda: c.cv_alpha(folds=4, seed=1))
    assert msg and r["fits"] == []                   # no refits

# 16. Constant b in the refinement, non-finite inputs and CVXPY's
#     inaccuracy warning
INACCURATE = "Solution may be inaccurate"            # CVXPY UserWarning

def _warns_all(func):
    """Result of func() and (category, message) of all warnings issued."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = func()
    return out, [(v.category, str(v.message)) for v in w]

def _osqp_inaccurate(func):
    """func() with every OSQP solution reported as optimal_inaccurate."""
    from cvxpy.reductions.solvers.qp_solvers.osqp_qpif import OSQP
    invert = OSQP.invert
    def patched(self, solution, inverse_data):
        out        = invert(self, solution, inverse_data)
        out.status = cp.OPTIMAL_INACCURATE           # CVXPY warns
        return out
    return _patched(OSQP, "invert", patched, func)

def test_refinement_constant_b():
    kw     = dict(C=np.ones((1, 2)), M=np.eye(2),
                  b=np.ones(3))                      # sd(b) = 0
    c, msg = _warns(lambda: CLSP().solve(r=5, **kw))
    assert not msg, msg                              # no numpy warnings
    assert c.r == 2 and c.r_max == 5 and np.isinf(c.nrmse)
    assert np.isnan(c.r2_partial) and np.all(np.isfinite(c.x))
    assert c.status["final"] in {"optimal", "optimal_inaccurate"}
    c, msg = _warns(lambda: CLSP().solve(r=1, **kw))
    assert not msg and c.r == 1 and np.isinf(c.nrmse)
    c, msg = _warns(lambda: CLSP().solve(r=5, C=np.ones((1, 2)),
                                         M=np.eye(2),
                                         b=np.array([1.0, 1.0, 2.0])))
    assert not msg and c.r == 3                      # sd > 0
    assert np.isfinite(c.nrmse)

def test_nonfinite_inputs_raise():
    C, S, M = np.eye(2), np.array([[1.0], [-1.0]]), np.eye(2)
    base    = dict(C=C, S=S, M=M, b=np.ones(4), final=False)
    bad     = lambda a, i, v: np.where(np.arange(a.size).reshape(a.shape) ==
                                       i, v, a)
    cases   = [dict(C=bad(C, 0, np.nan)), dict(S=bad(S, 1, np.inf)),
               dict(M=bad(M, 3, -np.inf)), dict(b=bad(np.ones(4), 2, np.nan)),
               dict(Z=bad(np.eye(3), 8, np.nan))]
    for kw in cases:
        try:
            CLSP().solve(**(base | kw))
        except CLSPError as e:
            assert "must be finite" in str(e), str(e)
            continue
        raise AssertionError(f"no CLSPError for {sorted(kw)}")
    c       = CLSP().solve(**base)
    try:
        c.solve(b=np.array([1.0, np.inf, 1.0, 1.0])) # stored A, new b
    except CLSPError as e:
        assert "must be finite" in str(e), str(e)
        return
    raise AssertionError("no CLSPError for a non-finite stored-A b")

def test_step2_inaccurate_no_user_warning():
    kw     = dict(C=A_KKT, b=B_KKT, bounds=(0, None), alpha=1.0)
    x      = cp.Variable(2)                          # the patch warns
    _, msg = _osqp_inaccurate(lambda: _warns_all(lambda: cp.Problem(
                              cp.Minimize(cp.sum_squares(x - 1)),
                              [x >= 0]).solve(solver=cp.OSQP)))
    assert any(k is UserWarning and INACCURATE in m for k, m in msg)
    c, msg = _osqp_inaccurate(lambda: _warns_all(lambda: CLSP().solve(**kw)))
    assert not msg, msg                              # retried: CLARABEL
    assert c.status["final"] == "optimal" and c.status["solver"] == "CLARABEL"
    assert c.status["retry"] == "OSQP: status optimal_inaccurate"
    assert _close(c.x, [1.0, 0.0], 1e-7)
    c, msg = _patched(cp.Problem, "solve",
                      _inject(lambda p, s: s == cp.CLARABEL),
                      lambda: _osqp_inaccurate(
                      lambda: _warns_all(lambda: CLSP().solve(**kw))))
    assert not msg, msg                              # kept: verified OSQP
    assert c.status["final"] == "optimal_inaccurate"
    assert c.status["solver"] == "OSQP" and _close(c.x, [1.0, 0.0], 1e-7)
    c, msg = _patched(cp.Problem, "solve", _inject(lambda p, s: True),
                      lambda: _warns_all(lambda: CLSP().solve(**kw)))
    assert [k for k, _ in msg] == [RuntimeWarning]   # genuine fallback
    assert c.status["final"].startswith("fallback")

# 17. Step 2 anchor shortcut: a feasible zhat0 is returned without a solver
ANCHOR = "none (anchor feasible)"                    # status["solver"]

def test_step2_anchor_feasible_exact():
    kw     = dict(C=np.array([[1.0, 2.0]]), b=np.array([5.0]),
                  bounds=[(0, 1), (0, None)])        # zhat0 = (1, 2)
    fail   = _inject(lambda p, s: True)              # no solver is called
    for alpha in (0.0, 0.5, 1.0):
        c, msg = _patched(cp.Problem, "solve", fail,
                          lambda: _warns(lambda: CLSP().solve(alpha=alpha,
                                                              **kw)))
        assert not msg and _close(c.x, [1.0, 2.0], 1e-9), f"alpha = {alpha}"
        assert c.status["final"] == "optimal" and c.status["retry"] is None
        assert c.status["solver"] == ANCHOR and c.status["eq_error"] < TOL
    C, M, b = _consistent_general(13)                # unbounded: exact no-op
    rng     = np.random.default_rng(14)
    for kw in (dict(C=C, M=M, b=b), dict(C=rng.standard_normal((2, 4)),
               M=rng.standard_normal((4, 4)), b=rng.standard_normal(6))):
        for alpha in (0.0, 0.5, 1.0):
            c = CLSP().solve(alpha=alpha, **kw)
            assert np.array_equal(c.z, c.zhat) and c.status["solver"] == ANCHOR
    c       = CLSP().solve(C=np.array([[1.0, 2.0]]), b=np.array([3.0]),
                           bounds=(0, None),
                           scale=True)               # zhat0 in u
    assert np.array_equal(c.z, c.zhat0) and c.status["solver"] == ANCHOR

def test_step2_anchor_infeasible_reference():
    C, b   = np.array([[1.0, 2.0]]), np.array([5.0])
    kw     = dict(C=C, b=b, bounds=[(0, 0.5), (0, None)], alpha=1.0)
    lb, ub = np.zeros(2), np.array([0.5, np.inf])    # z* = (0.5, 2.25)
    ref    = _cvx_closest(np.array([1.0, 2.0]), C, b, lb, ub, l2=True)
    assert _close(ref, [0.5, 2.25], 1e-7)
    c, msg = _warns(lambda: CLSP().solve(**kw))      # OSQP
    assert not msg and c.status["solver"] == "OSQP"
    assert _close(c.x, ref, 1e-7)
    c, msg = _osqp_inaccurate(lambda: _warns(lambda: CLSP().solve(**kw)))
    assert not msg                                   # retry
    assert c.status["solver"] == "CLARABEL"
    assert c.status["retry"] == "OSQP: status optimal_inaccurate"
    assert _close(c.x, ref, 1e-7)
    for kw, A, b, lb, ub in _bounded_cases():
        z_ls, _ = _bounded_ls(A, b, lb, ub)
        for s in ({}, {"solver": "CLARABEL"}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                c = CLSP().solve(alpha=1.0, **kw, **s)
            ref = _cvx_closest(c.zhat0.ravel(), A, A @ z_ls, lb, ub, l2=True)
            err = np.max(np.abs(c.z.ravel() - ref))
            assert c.status["solver"] != ANCHOR
            assert err <= 1e-7, f"{s}: max |z - ref| = {err:.3e}"

# 18. Step 2 polish (α = 1) on the active set and ttest() with sd(b) = 0
def _exact_projection(A, w, z0, lb, ub):
    """argmin ‖z - z0‖₂ s.t. Az = w, lb ≤ z ≤ ub (active sets)."""
    U, s, _ = np.linalg.svd(A, full_matrices=False)
    U       = U[:, s > 1e-12 * s[0]]
    best    = (np.inf, None)
    states  = [[0] + [1] * int(lo > -np.inf) + [2] * int(hi < np.inf)
               for lo, hi in zip(lb, ub)]            # free, at lb, at ub
    for st in map(np.array, itertools.product(*states)):
        F    = st == 0
        z    = np.where(st == 1, lb, np.where(st == 2, ub, 0.0))
        if F.any():
            P    = U.T @ A[:, F]
            z[F] = z0[F] + np.linalg.pinv(P) @ (U.T @ (w - A[:, ~F] @ z[~F]) -
                                                P @ z0[F])
        if (np.linalg.norm(A @ z - w) <= 1e-10 * max(1.0, np.linalg.norm(w))
            and np.all(z >= lb - 1e-12) and np.all(z <= ub + 1e-12)
            and np.sum((z - z0) ** 2) < best[0]):
            best = (np.sum((z - z0) ** 2), z)
    return best[1]

def test_step2_polish_exact():
    kw     = dict(C=np.array([[1.0, 2.0]]), b=np.array([5.0]),
                  bounds=[(0, 0.5), (0, None)], alpha=1.0)
    for s in ({}, {"solver": "CLARABEL"}):           # z* = (0.5, 2.25)
        c, msg = _warns(lambda: CLSP().solve(**kw, **s))
        assert not msg and c.status["polish"] is True
        assert _close(c.x, [0.5, 2.25], 1e-13) and c.status["eq_error"] < 1e-13
    c      = CLSP().solve(C=np.ones((1, 3)), b=np.array([3.0]),
                          bounds=[(None, 0.5), (None, 1.25 + 1e-6),
                                  (None, None)])     # x2 free, near its bound
    assert c.status["polish"] is True
    assert _close(c.x, [0.5, 1.25, 1.25], 1e-13)
    for kind in ("over_box", "under_nonnegative", "under_box",
                 "rank_deficient_box"):
        for seed in range(1000, 1010):
            A, b, lo, hi = _bvls_case(kind, seed)
            n            = A.shape[1]
            c            = CLSP().solve(C=A, b=b, bounds=(lo, hi), alpha=1.0)
            ref          = _exact_projection(A, A @ c.zhat.ravel(),
                                             c.zhat0.ravel(),
                                             np.full(n, lo), np.full(n, hi))
            err          = np.max(np.abs(c.z.ravel() - ref))
            assert c.status["polish"] is (c.status["solver"] != ANCHOR)
            assert err <= 1e-12, f"{kind}, seed {seed}: {err:.3e}"
    c      = CLSP().solve(**kw | {"alpha": 0.5})     # not applied
    assert c.status["polish"] is False
    c      = CLSP().solve(**kw | {"bounds": [(0, 1), (0, None)]})
    assert c.status["solver"] == ANCHOR and c.status["polish"] is False
    c      = CLSP().solve(**kw | {"final": False})
    assert c.status["polish"] is None

def test_step2_polish_rejected():
    solve  = cp.Problem.solve
    def wrong(self, *args, **kwargs):                # verified, not optimal
        out                       = solve(self, *args, **kwargs)
        self.variables()[0].value = np.array([0.0, 2.5])
        return out
    kw     = dict(C=np.array([[1.0, 2.0]]), b=np.array([5.0]),
                  bounds=[(0, 0.5), (0, None)], alpha=1.0)
    c, msg = _patched(cp.Problem, "solve", wrong,
                      lambda: _warns(lambda: CLSP().solve(**kw)))
    assert not msg and c.status["final"] == "optimal"
    assert c.status["polish"] is False               # KKT signs fail
    assert _close(c.x, [0.0, 2.5], 0.0)              # solver result kept

def test_ttest_constant_b():
    c      = CLSP().solve(C=np.ones((1, 2)), M=np.eye(2), b=np.ones(3))
    for kw in ({}, {"partial": True}, {"simulate": True},
               {"simulate": True, "partial": True}):
        t, w = _warns_all(lambda: c.ttest(sample_size=5, seed=1, **kw))
        assert len(w) == 1 and w[0][0] is RuntimeWarning, f"{kw}: {w}"
        assert "not finite" in w[0][1]               # no numpy warnings
        assert all(np.isnan(t[k]) for k in ("p_one_left", "p_one_right",
                                            "p_two_sided")), f"{kw}"
        assert np.isinf(t["nrmse"])

# 19. Verification of the clipped anchor and soft zero-diagonal rows
def test_step2_clipped_anchor_verified():
    # zhat0 ≈ (1/c0, ·) is within tol of the bound x1 = 0, but clipping
    # it to 0 costs c0 · 1/c0 = 1 in the fit: the shortcut must not apply
    for c0 in (1e8, 1e9, 1e12):
        kw = dict(C=np.array([[c0, 1.0]]), b=np.array([1.0]),
                  bounds=[(0.0, 0.0), (None, None)])
        for alpha in (0.0, 0.5, 1.0):
            c, w = _warns_all(lambda: CLSP().solve(alpha=alpha, **kw))
            s    = c.status
            assert s["solver"] != ANCHOR, f"{c0}, {alpha}"
            assert s["eq_error"] <= s["tol_v"] and c.residual <= s["tol_v"]
            assert (s["final"] == "optimal") != bool(w), f"{s}, {w}"
            assert _close(c.x, [0.0, 1.0], 2e-4)     # zhat = (0, 1)

def test_zero_diagonal_rows_kept():
    # x11 = 3 (M) and x11 = 0 (zero diagonal) are both least-squares rows
    b = np.array([2.0, 2.0, 2.0, 2.0, 3.0])
    c = CLSP().solve(problem="ap", b=b, M=np.eye(4)[[0]], m=2, p=2,
                     zero_diagonal=True)
    assert c.A.shape[0] == 4 + 1 + 2
    assert _close(c.x, [[1.2, 1.1], [1.1, 0.6]], 1e-9)  # ‖r‖² = 5.4
    c = CLSP().solve(problem="ap", b=np.r_[b, 1.0], M=np.eye(4)[[0, 0]],
                     m=2, p=2)                       # repeated rows kept
    d = CLSP().solve(problem="ap", b=np.r_[b[:4], 2.0 * np.sqrt(2)],
                     M=np.sqrt(2) * np.eye(4)[[0]], m=2, p=2)
    assert c.A.shape[0] == 6 and _close(c.x, d.x, 1e-9)

# Plain-script runner
if __name__ == "__main__":
    tests  = [(k, f) for k, f in list(globals().items())
              if k.startswith("test_") and callable(f)]
    failed = []
    for name, func in tests:
        try:
            func()
            print(f"PASS  {name}")
        except Exception as e:
            failed.append(name)
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=-2)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
