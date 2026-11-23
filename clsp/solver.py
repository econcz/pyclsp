import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*Converting (A|P) to a CSC.*",
    category=UserWarning,
    module=r"scs(\.|$)"
)
import numbers
import numpy          as     np
import scipy.linalg   as     la
import cvxpy          as     cp
from   scipy.sparse   import csc_matrix
from   scipy.optimize import lsq_linear, linprog

def CLSPSolveInstance(
    self, problem:         str   = "",   C: np.ndarray | None = None,
          S: np.ndarray |  None  = None, M: np.ndarray | None = None,
                                         b: np.ndarray | None = None,
          m:     int    |  None  = None, p: int        | None = None,
          i:     int             = 1,    j: int               = 1,
          zero_diagonal:   bool  = False,
          r:     int             = 1,    Z: np.ndarray | None = None,
          rcond:           float | bool                | None = None,
          tolerance:       float                       | None = None,
          iteration_limit: int                         | None = None,
          final: bool   |  None  = None, alpha: float  | None = None,
          cond_tolerance:  float                       | None = None,
          bounds:          tuple | list                | None = None,
          slack_sign:      bool                        | None = None,
          scale:           bool                        | None = None,
          *args, **kwargs
) -> "CLSP":
    """
    Solve the Convex Least Squares Programming (CLSP) problem.

    This method performs a two-step estimation:
    (1) a pseudoinverse-based solution using either the Moore–Penrose or
        Bott–Duffin inverse, optionally iterated for convergence;
    (2) a convex-programming correction using Lasso, Ridge, or Elastic Net
        regularization (if enabled).

    Parameters
    ----------
    problem : str, optional
        Structural template for matrix construction. One of:
        - 'ap'   or 'tm' : allocation or tabular matrix problem.
        - 'cmls' or 'rp' : constrained modular least squares or RP-type.
        - ''     or other: General CLSP problems (user-defined C and/or M).

    C, S, M : np.ndarray or None
        Blocks of the constraint matrix A = [C | S; M | Q].
        If `C` and/or `M` are provided, the matrix A is constructed
        accordingly. If both are None and A is not yet defined, an error
        is raised. The rows of C and M are both fitted in the least-squares
        sense.

    b : np.ndarray or None
        Right-hand side vector. Must have as many rows as A. Required. If
        C, M and (m, p) are not provided, the stored specification (A
        without Q) is reused with the new b.

    m, p : int or None
        Dimensions of X ∈ ℝ^{m×p}, relevant for allocation problems ('ap').

    i, j : int, default = 1
        Grouping sizes for row and column sum constraints in AP problems.

    zero_diagonal : bool, default = False
        If True, adds soft model rows x_kk = 0 for the diagonal of X in AP
        problems (exact zeros via equal bounds, lo = hi = 0). The rows are
        appended to M without merging: a known diagonal cell in M and its
        row x_kk = 0 are both fitted in least squares.

    r : int, default = 1
        Number of refinement iterations for the pseudoinverse-based estimator.
        When `r > 1`, the slack block Q is updated iteratively to improve
        feasibility in underdetermined or ill-posed systems:
        Q = diag(-sign(e_M)), e_M = b_M - M·x, with entries
        |e_i| ≤ tolerance · max(1, ‖b‖₂) set to zero.

    Z : np.ndarray or None
        A symmetric idempotent matrix (projector) defining the subspace for
        Bott–Duffin pseudoinversion. If None, the identity matrix is used,
        reducing to the Moore–Penrose case.

    rcond : float, bool or None, default = None
        Relative cutoff on the singular values of A·Z. A_k is A·Z with the
        singular values σ ≤ rcond·σ_max set to zero, used in Step 1
        (zhat0 = A_k⁺·b and the bounded fit), for w* and in Step 2:
        - False: sqrt(max(A.shape) · eps) (initial value)
        - True : `tolerance`
        - float: the given cutoff
        - None : the stored value is kept

    tolerance : float or None, default = None
        Convergence tolerance for NRMSE change between iterations (the
        iteration also stops if NRMSE = inf in both, i.e. sd(b) = 0).

    iteration_limit : int or None, default = None
        Maximum number of iterations allowed in the refinement loop and of
        active-set passes in the bounded Step 1.

    final : bool or None, default = None
        If True, a convex programming problem is solved to refine `zhat`:
        `z` minimizes a weighted L1/L2 norm around `zhat0` subject to
        A_k·z = w*, `bounds` and `slack_sign`, where w* = A_k·zhat. For
        Z = I and α = 1, z is the minimum-norm (bounded) least-squares
        solution. A feasible zhat0 is returned without a solver
        (status["solver"] = "none (anchor feasible)"). If the solver fails
        or the equality or bound error of z exceeds
        tol_v = sqrt(tolerance) · max(1, ‖b‖₂), z = zhat is returned with
        a RuntimeWarning; OSQP is first retried with CLARABEL
        (status["retry"]). For α = 1, a verified solution is polished on
        its active set (status["polish"]). If None, the stored value is
        kept (initially True).

    alpha : float or None, default = None
        Regularization parameter in [0, 1]:
        - α = 0: Lasso (L1 norm)
        - α = 1: Ridge (L2 norm)
        - 0 < α < 1: Elastic Net
        If None, α = 1. Solved with ECOS (α = 0), OSQP (α = 1, with
        eps_abs = eps_rel = 1e-10 and max_iter = 100000 unless given) or
        CLARABEL (0 < α < 1), unless `solver` is passed in kwargs.

    cond_tolerance : float or None, default = None
        Singular-value cutoff for the custom condition number function.
        If None, the implementation uses an internal relative cutoff of 1e-14.

    bounds : tuple, list or None, default = None
        Bounds on x, enforced in Step 1 by bounded-variable least squares
        on A_k (verified by the KKT conditions, else scipy's BVLS with a
        RuntimeWarning) and imposed as constraints in Step 2. One of:
        - None           : no bounds (a previously set value is kept).
        - (lo, hi)       : the same bounds for all x.
        - [(lo, hi), ...]: one pair per x column (or a single pair).
        None inside a pair denotes an unbounded side. Bounds require a
        diagonal (0/1) Z and must admit 0 where Z_jj = 0.

    slack_sign : bool or None, default = None
        If True, the S-block slacks must be ≥ 0 (S_ii = +1 encodes ≤,
        S_ii = -1 encodes ≥); Q is never constrained. If None, the stored
        value is kept (initially False).

    scale : bool or None, default = None
        If True, the columns of A·Z are equilibrated, A·Z·D with
        D = diag(1/‖(A·Z)_j‖₂) (1 for zero columns), and Steps 1
        and 2 are solved in u = D⁻¹·z (minimum norm in u). Requires a
        diagonal (0/1) Z. If None, the stored value is kept (initially
        False).

    *args, **kwargs : optional
        Additional arguments passed to the CVXPY solver backend.

    Attributes Set
    ---------------
    self.A : np.ndarray
        Canonical design matrix constructed from (C, S, M, Q).

    self.b : np.ndarray
        Conformable right-hand side vector.

    self.Z : np.ndarray
        Projector matrix used for Bott–Duffin inversion.

    self.r : int
        Number of refinement iterations performed.

    self.r_max : int
        Requested number of refinement iterations (reused by `corr()` and
        `ttest()`).

    self.zhat : np.ndarray
        First-step solution (unregularized pseudoinverse estimate).

    self.zhat0 : np.ndarray
        Unbounded first-step estimate A_k⁺·b, the center of Step 2.

    self.rank, self.sv_discarded : int, np.ndarray
        Effective rank of A·Z (the rank of A_k) and the discarded singular
        values, of A·Z·D if `scale` is True.

    self.x_lb, self.x_ub : np.ndarray
        Lower and upper bounds on x (-inf/inf if unbounded).

    self.fixed, self.fixed_at : np.ndarray
        Components of z fixed at a bound in Step 1 and the pass of fixing
        (0 = not fixed).

    self.alpha : float
        Regularization parameter.

    self.z : np.ndarray
        Final estimate after optional convex refinement.

    self.rho : float or None
        Smallest residual norm ‖b - w*‖₂ within the numerical range of A
        (singular values above the cutoff) under the bounds (None if
        `final` is False).

    self.residual : float
        Residual norm ‖b - A·z‖₂.

    self.status : dict
        Statuses of Step 1 ("step1"), the w* fit ("rho") and Step 2
        ("final"), the Step 2 solver ("solver"), a CLARABEL retry
        ("retry"), the α = 1 polish ("polish"), the verification
        tolerance ("tol_v"), the achieved errors ("kkt_error",
        "bound_error", "eq_error") and the column scaling ("scale").

    self.x, self.y : np.ndarray
        Variable and slack components reshaped from `z`.

    self.nrmse : float
        Normalized root mean squared error over the full system.

    self.r2_partial : float or np.nan
        R² for the M block (nan without M rows), computed from partial
        residuals.

    self.nrmse_partial : float
        NRMSE over the M block (nan without M rows).

    self.kappaA, self.kappaB, self.kappaC : float
        Condition numbers for the full, projected, and constrained system.

    self.z_lower, self.z_upper : np.ndarray
        Condition-weighted sensitivity band for z (heuristic),
        z ∓ |z|·κ(A)·‖b - A·z‖₂ / ‖b‖₂.

    Returns
    -------
    self

    Raises
    ------
    CLSPError
        If the design matrix A or right-hand side b is malformed, inconsistent,
        not finite (NaN or inf in C, S, M, b or Z), or incompatible with the
        structural assumptions of the problem.
    """
    # (rmsa), (nrmse_ttest), (cv_alpha_result) Reset previous diagnostics
    self.rmsa,         self.rmsa_i          = None, []
    self.rmsa_dkappaC, self.rmsa_dkappaB    = [],   []
    self.rmsa_dkappaA, self.rmsa_dnrmse     = [],   []
    self.rmsa_dzhat,   self.rmsa_dz         = [],   []
    self.rmsa_dx,      self.rmsa_failures   = [],   0
    self.nrmse_ttest,  self.nrmse_ttest_key = [],   None
    self.nrmse_ttest_failures               = 0
    self.cv_alpha_result                    = None

    # (A), (b) Construct a conformable canonical form for the CLSP estimator
    if (C is not None or M is not None) or (m is not None and p is not None):
        self.canonize(problem, C, S, M, None, b,
                               m, p, i, j, zero_diagonal)
    elif self.A_base is None:
        raise self.error("At least one of C, M, m, or p must be provided.")
    elif b is not None:                                # stored A, new b
        b      = np.asarray(b, dtype=np.float64).reshape(-1, 1)
        if self.A_base.shape[0] != b.shape[0]:
            raise self.error(f"The matrix A and vector b must have the same "
                             f"number of rows: A has {self.A_base.shape[0]}, "
                             f"but b has {b.shape[0]}")
        self.b = b
    self.A     = self.A_base                           # A without Q
    m, p       = self.spec["m"], self.spec["p"]        # dimensions of X
    if self.A.shape[0] != self.b.shape[0]:
        raise self.error(f"The matrix A and vector b must have the same "
                         f"number of rows: A has {self.A.shape[0]}, but "
                         f"b has {self.b.shape[0]}")

    # (zhat) (Iterated if r > 1) first-step estimate
    if r < 1:
        raise self.error("Number of refinement iterations r must be ≥ 1.")
    if Z is not None:
        self.Z               = Z
    elif self.Z is None:
        self.Z               = np.eye(self.A.shape[1], dtype=np.float64)
    else:
        self.Z               = self.Z[0:self.A.shape[1], 0:self.A.shape[1]]
    if not (np.all(np.isfinite(self.A)) and np.all(np.isfinite(self.b)) and
            np.all(np.isfinite(self.Z))):
        raise self.error("C, S, M, b and Z must be finite (no NaN or inf)")
    if rcond           is not None:
        self.rcond           = rcond
    if tolerance       is not None:
        self.tolerance       = tolerance
    if iteration_limit is not None:
        self.iteration_limit = iteration_limit
    if bounds          is not None:
        self.bounds          = bounds
    if slack_sign      is not None:
        self.slack_sign      = bool(slack_sign)
    if scale           is not None:
        self.scale           = bool(scale)
    try:
        if (not np.allclose(self.Z,          self.Z.T, atol=self.tolerance) or
            not np.allclose(self.Z @ self.Z, self.Z,   atol=self.tolerance) or
            self.Z.shape[0] != self.A.shape[1]):
            raise ValueError
    except ValueError:
        raise self.error(f"Matrix Z must be symmetric, idempotent and "
                         f"match the number of columns in A: expected "
                         f"({self.A.shape[1]}, {self.A.shape[1]}), "
                         f"got {self.Z.shape}")
    # (x_lb), (x_ub) Bounds on x and, if signed, on the S-block slacks
    n_x        = self.C_idx[1]
    try:
        pairs  = ([]            if self.bounds is None                 else
                  [self.bounds] if len(self.bounds) == 2 and
                                   all(v is None or
                                       isinstance(v, numbers.Real)
                                       for v in self.bounds)          else
                  list(self.bounds))
        if pairs and (len(pairs) not in {1, n_x} or
                      any(len(v) != 2 for v in pairs)):
            raise ValueError
        lb, ub = np.array([[-np.inf if v[0] is None else v[0],
                             np.inf if v[1] is None else v[1]]
                           for v in pairs] or [[-np.inf, np.inf]],
                          dtype=np.float64).T
        if np.any(np.isnan(lb)) or np.any(np.isnan(ub)) or np.any(lb > ub):
            raise ValueError
    except (TypeError, ValueError):
        raise self.error(f"bounds must be None, a (lo, hi) pair, or a "
                         f"sequence of {n_x} (or 1) (lo, hi) pairs with "
                         f"lo ≤ hi")
    self.x_lb  = np.broadcast_to(lb, (n_x,)).copy()
    self.x_ub  = np.broadcast_to(ub, (n_x,)).copy()
    S_idx      = self.S_idx if self.S_idx[0] is not None else (n_x, n_x)
    lb_z       = np.full(self.A.shape[1], -np.inf)     # bounds on z
    ub_z       = np.full(self.A.shape[1],  np.inf)
    lb_z[:n_x] = self.x_lb
    ub_z[:n_x] = self.x_ub
    if self.slack_sign:
        lb_z[S_idx[0]:S_idx[1]] = 0.0
    bounded    = bool(np.any(np.isfinite(lb_z)) or np.any(np.isfinite(ub_z)))
    if bounded and not np.allclose(self.Z, np.diag(np.diag(self.Z)),
                                   atol=self.tolerance):
        raise self.error("bounds require a diagonal (0/1) Z")
    if bounded and np.any((np.diag(self.Z) < 0.5) & ((lb_z > 0) |
                                                    (ub_z < 0))):
        raise self.error("bounds conflict with Z: a coordinate excluded by "
                         "Z (Z_jj = 0) is 0, but its bounds exclude 0")
    if self.scale and not np.allclose(self.Z, np.diag(np.diag(self.Z)),
                                      atol=self.tolerance):
        raise self.error("scale requires a diagonal (0/1) Z")

    # (zhat) Bounded least squares on A_k, iterated if r > 1
    eps_cut    = lambda A: np.sqrt(max(A.shape) * np.finfo(float).eps)
    rtol       = lambda A: (eps_cut(A)     if self.rcond is False else
                            self.tolerance if self.rcond is True  else
                            self.rcond)
    to_z       = lambda u: np.clip(d.reshape(-1, 1) * u,
                                   lb_z.reshape(-1, 1), ub_z.reshape(-1, 1))
    self.r_max = r
    r_max      = r if self.A.shape[0] > self.C_idx[0] else 1
    for n_iter in range(1, 1 + r_max):
        # save NRMSE from the previous step, construct Q and Z
        if n_iter > 1:
            nrmse_prev = (lambda residuals, sd:
                          np.linalg.norm(residuals) / np.sqrt(sd.shape[0]) /
                          np.std(sd) if np.std(sd) > 0                    else
                          np.inf)(self.b - self.A @ self.zhat, self.b)
            e          = (self.b[self.C_idx[0]:] -     # e_M = b_M - M·x
                          self.A[self.C_idx[0]:, :self.C_idx[1]] @
                          self.zhat[:self.C_idx[1]])
            e[np.abs(e) <= self.tolerance *            # rounding noise
                           max(1.0, np.linalg.norm(self.b))] = 0.0
            Q          = np.diagflat(-np.sign(e))
            self.A     = np.hstack([self.A_base[:, :S_idx[1]],
                                    np.vstack([np.zeros((self.C_idx[0],
                                                         Q.shape[1])), Q])])
            Z_delta    = self.A.shape[1] - self.Z.shape[0]
            if Z_delta > 0:                            # augment Z by I
               self.Z  = np.block([[self.Z,
                                    np.zeros((self.Z.shape[0], Z_delta))],
                                   [np.zeros((Z_delta, self.Z.shape[1])),
                                    np.eye(Z_delta, dtype=np.float64)]])
               lb_z    = np.r_[lb_z, np.full(Z_delta, -np.inf)]
               ub_z    = np.r_[ub_z, np.full(Z_delta,  np.inf)]
        # (optional) column scaling D = diag(d), u = D⁻¹·z
        d                 = ((lambda n: 1.0 / np.where(n > 0, n, 1.0))(
                             np.linalg.norm(self.A @ self.Z, axis=0))
                             if self.scale else np.ones(self.A.shape[1]))
        lb_u, ub_u        = lb_z / d, ub_z / d         # bounds on u
        # A_k: A·Z·D with the singular values below the cutoff set to zero
        try:
            U, s, Vt          = la.svd(self.A @ self.Z * d,
                                       full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            raise self.error("Pseudoinverse estimate zhat failed: SVD did "
                             "not converge")
        keep              = s > rtol(self.A) * np.max(s, initial=0.0)
        A_k               = (self.A @ self.Z * d if np.all(keep) else
                             (U[:, keep] * s[keep]) @ Vt[keep])
        self.rank         = int(np.sum(keep))
        self.sv_discarded = s[~keep]
        u0                = Vt[keep].T @ ((U[:, keep].T @ self.b) /
                                          s[keep].reshape(-1, 1))
        self.zhat0        = d.reshape(-1, 1) * u0
        # solve the bounded least-squares problem on A_k (BVLS)
        if bounded:
            (u_hat,     fixed,
             fixed_at,  s_step1,
             e_step1)            = _bvls(A_k, self.b, lb_u, ub_u,
                                         np.diag(self.Z) > 0.5,
                                         self.tolerance, self.iteration_limit,
                                         U[:, keep])
        else:
            u_hat                = u0.copy()
            fixed                = np.zeros(self.A.shape[1], dtype=bool)
            fixed_at             = np.zeros(self.A.shape[1], dtype=int)
            s_step1              = "converged"
            e_step1              = float(np.max(np.abs(A_k.T @ (A_k @
                                         u_hat - self.b)), initial=0.0))
        self.zhat      = to_z(u_hat)
        self.nrmse     = (lambda residuals, sd:
                          np.linalg.norm(residuals) / np.sqrt(sd.shape[0]) /
                          np.std(sd) if np.std(sd) > 0                    else
                          np.inf)(self.b - self.A @ self.zhat, self.b)
        # break on convergence (NRMSE = inf in both steps if sd(b) = 0)
        self.r         = n_iter
        if n_iter > 1:
            if (self.nrmse                   == nrmse_prev     or
                abs(self.nrmse - nrmse_prev) <  self.tolerance or
                n_iter                       >  self.iteration_limit):
                del nrmse_prev, Q
                break
    self.fixed         = fixed
    self.fixed_at      = fixed_at
    if not np.all(np.isfinite(self.zhat)):
        self.zhat = np.full((self.A.shape[1], 1), np.nan, dtype=np.float64)
        raise self.error("Pseudoinverse estimate zhat failed")

    # (z) Final solution (if available), or set self.z = self.zhat
    if final is not None:
        self.final = bool(final)
    self.alpha = np.float64(1.0 if alpha is None else alpha)
    if self.final:
        # build a convex problem (p_cvx) and its solver (c_cvx) around zhat0
        A_csc = csc_matrix(A_k)
        z_cvx = cp.Variable(A_csc.shape[1])
        d_cvx = z_cvx - u0.flatten()                   # u = D⁻¹·z
        if   self.alpha == 0:                          # Lasso
            f_obj = cp.norm1(d_cvx)
            s_cvx = cp.ECOS
        elif self.alpha == 1:                          # Ridge
            f_obj = cp.sum_squares(d_cvx)
            s_cvx = cp.OSQP
        else:                                          # Elastic Net
            f_obj = ((1 - self.alpha) * cp.norm1(d_cvx)      +
                     self.alpha       * cp.sum_squares(d_cvx))
            s_cvx = cp.CLARABEL
        kw    = {k: v for k, v in kwargs.items()
                      if k not in {"rcond"}}
        s_cvx = kw.pop("solver", s_cvx)                # user override
        kw    = ({"eps_abs": 1e-10, "eps_rel": 1e-10,  # tight OSQP
                  "max_iter": 100_000} | kw if s_cvx == cp.OSQP else kw)
        s_all = [s_cvx] + ([cp.CLARABEL] if s_cvx == cp.OSQP and
                                            "solver" not in kwargs else [])
        i_lb  = np.flatnonzero(np.isfinite(lb_u))
        i_ub  = np.flatnonzero(np.isfinite(ub_u))
        b_cvx = lambda z: (([z[i_lb] >= lb_u[i_lb]] if i_lb.size else []) +
                           ([z[i_ub] <= ub_u[i_ub]] if i_ub.size else []))
        # (bounded) least-squares fitted vector (w) and its residual (rho)
        w        = A_k @ u_hat                         # A_k·u0 if unbounded
        s_rho    = "bvls" if bounded else "pinv"
        s_ok     = {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} # accepted statuses
        self.rho = np.linalg.norm(self.b - w)
        c_cvx    = [A_csc @ z_cvx == w.flatten()]      # residual = rho
        p_cvx    = cp.Problem(cp.Minimize(f_obj), c_cvx + b_cvx(z_cvx))
        # anchor: a feasible u0 has f_obj = 0, i.e. is optimal
        tol_v = np.sqrt(self.tolerance) * max(1.0, np.linalg.norm(self.b))
        tol_a = self.tolerance          * max(1.0, np.linalg.norm(self.b))
        u_c   = np.clip(u0[:, 0], lb_u, ub_u).reshape(-1, 1)
        e_b   = np.max(np.r_[lb_u - u0[:, 0], u0[:, 0] - ub_u, 0.0])
        e_w   = max(np.linalg.norm(A_k @ u0  - w),     # verify the clipped
                    np.linalg.norm(A_k @ u_c - w))     # anchor, not only u0
        retry = None
        kept  = None                                   # verified solution
        if e_w <= tol_a and e_b <= tol_a:              # no solver is called
            kept  = (u_c, cp.OPTIMAL, "none (anchor feasible)")
            s_all = []
        # solve and verify, else retry with CLARABEL or fall back to zhat
        for n_try, s_cvx in enumerate(s_all):
            try:
                with warnings.catch_warnings():        # status is verified
                    warnings.filterwarnings(
                        "ignore",
                        message=r"Solution may be inaccurate\.",
                        category=UserWarning
                    )
                    p_cvx.solve(*args, solver=s_cvx, verbose=False,
                                **(kw if n_try == 0 else {}))
                if p_cvx.status not in s_ok or z_cvx.value is None:
                    reason = f"status {p_cvx.status}"
                else:
                    z_fin  = np.asarray(z_cvx.value, dtype=np.float64)
                    e_b    = np.max(np.r_[lb_u - z_fin, z_fin - ub_u, 0.0])
                    z_fin  = np.clip(z_fin, lb_u, ub_u).reshape(-1, 1)
                    e_w    = np.linalg.norm(A_k @ z_fin - w)
                    reason = (None if e_w <= tol_v and e_b <= tol_v else
                              f"verification failed: ‖A_k·z - w*‖₂ = "
                              f"{e_w:.3e}, bound violation = {e_b:.3e}, "
                              f"tol_v = {tol_v:.3e}")
            except (cp.SolverError, ValueError) as e:
                reason = f"{type(e).__name__}: {e}"
            if reason is None:
                kept   = (z_fin, p_cvx.status, s_cvx)
                if p_cvx.status == cp.OPTIMAL:
                    break
            if n_try < len(s_all) - 1:                 # inaccurate or failed
                retry  = f"{s_cvx}: {reason or 'status ' + p_cvx.status}"
        if kept is not None:                           # last verified
            (z_fin, s_fin, s_cvx), reason = kept, None
        polish = False
        if reason is None and s_all and self.alpha == 1:
            z_fin, polish = _polish(A_k, U[:, keep], w, u0, z_fin,
                                    lb_u, ub_u, tol_a, tol_v)
        if reason is None:                             # common final check
            e_b    = np.max(np.r_[lb_u - z_fin[:, 0], z_fin[:, 0] - ub_u,
                                  0.0])
            e_w    = np.linalg.norm(A_k @ z_fin - w)
            reason = (None if e_w <= tol_v and e_b <= tol_v else
                      f"final verification failed: ‖A_k·z - w*‖₂ = "
                      f"{e_w:.3e}, bound violation = {e_b:.3e}, "
                      f"tol_v = {tol_v:.3e}")
        if reason is None:
            self.z     = to_z(z_fin)
            self.nrmse = (lambda residuals, sd:
                          np.linalg.norm(residuals) / np.sqrt(sd.shape[0]) /
                          np.std(sd) if np.std(sd) > 0                else
                          np.inf)(self.b - self.A @ self.z,   self.b)
        else:
            warnings.warn(f"Step 2 failed ({reason}); falling back to zhat",
                          RuntimeWarning)
            self.z     = self.zhat.copy()
        self.status = {"step1" : s_step1,
                       "rho"   : s_rho,
                       "final" : s_fin if reason is None else
                                 f"fallback: {reason}",
                       "solver": str(s_cvx),
                       "retry" : retry,
                       "polish": polish,
                       "tol_v" : float(tol_v)}
    else:
        self.z      = self.zhat.copy()
        self.rho    = None
        self.status = {"step1" : s_step1, "rho": "skipped",
                       "final" : "skipped", "solver": None, "retry": None,
                       "polish": None,      "tol_v": None}
    self.status["kkt_error"]   = e_step1
    self.status["scale"]       = self.scale
    self.residual              = float(np.linalg.norm(self.b -
                                                      self.A @ self.z))
    self.status["bound_error"] = float(np.max(np.r_[lb_z - self.z[:, 0],
                                                    self.z[:, 0] - ub_z,
                                                    0.0]))
    self.status["eq_error"]    = (float(np.linalg.norm(A_k @ (self.z /
                                  d.reshape(-1, 1)) - w))
                                  if self.final else None)

    # (x), (y) Variable and slack components of z
    self.x = self.z[:self.C_idx[1]].reshape(m if m is not None else -1,
                                            p if p is not None else  1)
    self.y = self.z[self.C_idx[1]:]

    # (kappaC), (kappaB), (kappaA) Condition numbers
    self.kappaC            = _cond(                 self.A[:self.C_idx[0], :],
                                                    "e",       cond_tolerance)
    self.kappaB            = _cond(self.A @ la.pinv(self.A[:self.C_idx[0], :]),
                                                    "e",       cond_tolerance)
    self.kappaA            = _cond(self.A,          "e",       cond_tolerance)

    # (r2_partial), (nrmse_partial) M-block-based statistics
    self.r2_partial        = np.nan                    # no M rows
    self.nrmse_partial     = np.nan
    if self.A.shape[0]     > self.C_idx[0]:
        M                  = self.A[self.C_idx[0]:, :self.C_idx[1]]
        b_M                = self.b[-M.shape[0]:]
        residuals_M        = b_M - M @ self.x.reshape(-1, 1)
        self.r2_partial    = (lambda residuals, sd:
                          1  -  (np.linalg.norm(residuals) ** 2            /
                          np.linalg.norm(sd - np.mean(sd)) ** 2)
                          if np.std(sd) > 0                else
                          np.nan)(residuals_M,  b_M)
        self.nrmse_partial = (lambda residuals, sd:
                          np.linalg.norm(residuals) / np.sqrt(sd.shape[0]) /
                          np.std(sd) if np.std(sd) > 0                else
                          np.inf)(residuals_M,  b_M)
        del M, b_M, residuals_M

    # (z_lower), (z_upper) Condition-weighted sensitivity band
    dz           = (self.kappaA                              *
                    np.linalg.norm(self.b - self.A @ self.z) /
                    np.linalg.norm(self.b)
                    if np.linalg.norm(self.b) > 0                else
                    np.inf)
    self.z_lower = self.z - np.abs(self.z) * dz
    self.z_upper = self.z + np.abs(self.z) * dz

    # (x_lower), (x_upper), (y_lower), (y_upper)
    self.x_lower = self.z_lower[:self.C_idx[1]].reshape(m if m is not None
                                                          else -1,
                                                        p if p is not None
                                                          else  1)
    self.x_upper = self.z_upper[:self.C_idx[1]].reshape(m if m is not None
                                                          else -1,
                                                        p if p is not None
                                                          else  1)
    self.y_lower = self.z_lower[self.C_idx[1]:]
    self.y_upper = self.z_upper[self.C_idx[1]:]

    return self

def CLSPSolve(
    self, tolerance:      float       | None = None,
          final: bool                 | None = None,
          alpha: float                | None = None,
          bounds: tuple | list        | None = None,
          slack_sign:     bool        | None = None,
          *args, **kwargs
) -> "CLSP":
    """
    Solve the Convex Least Squares Programming (CLSP) problem for a given
    alpha in Step 2.

    This method performs a two-step estimation:
    (1) a pseudoinverse-based solution using either the Moore–Penrose or
        Bott–Duffin inverse, optionally iterated for convergence;
    (2) a convex-programming correction using Lasso, Ridge, or Elastic Net
        regularization (if enabled).

    Parameters
    ----------
    tolerance : float or None, default = None
        Convergence tolerance for NRMSE change between iterations.

    final : bool or None, default = None
        If True, Step 2 is solved. If None, the stored value is kept
        (initially True).

    alpha : float or None, default = None
        Regularization parameter, a single value in [0, 1]:
        - α = 0: Lasso (L1 norm)
        - α = 1: Ridge (L2 norm)
        - 0 < α < 1: Elastic Net
        If None, α = 1: the Step 2 solution is unique and, for Z = I, the
        minimum-norm (bounded) least-squares solution. A list, tuple or
        np.ndarray, a non-finite or an out-of-range value raises a
        CLSPError; see `cv_alpha()` for the selection of α.

    bounds : tuple, list or None, default = None
        Bounds on x, a (lo, hi) pair or a sequence of pairs. If None, the
        stored value is kept.

    slack_sign : bool or None, default = None
        If True, the S-block slacks must be ≥ 0. If None, the stored value
        is kept.

    The value is stored in `self.alpha` and passed to
    `CLSPSolveInstance(..., alpha=self.alpha)`.

    Returns
    -------
    self
    """
    # process keywords
    kw = {k: v for k, v in kwargs.items()
               if k not in {"tolerance", "final", "alpha"}}

    # process alpha
    if isinstance(alpha, (list, tuple, np.ndarray)):
        raise self.error("alpha must be a single value in [0, 1]; selection "
                         "by NRMSE was removed in 5.0.0 — use cv_alpha()")
    if alpha is not None and not (isinstance(alpha, numbers.Real) and
                                  0.0 <= alpha <= 1.0):
        raise self.error(f"alpha must be a single value in [0, 1], got "
                         f"{alpha!r}")

    return CLSPSolveInstance(self, tolerance=tolerance, final=final,
                                   alpha=alpha,         bounds=bounds,
                                   slack_sign=slack_sign,
                                   *args, **kw)

def _bvls(
    A:         np.ndarray,      b:               np.ndarray,
    lb:        np.ndarray,      ub:              np.ndarray,
    free:      np.ndarray,      tolerance:       float,
    iteration_limit: int = 50,  U:               np.ndarray | None = None
) -> tuple:
    """
    Solve min ‖b - A·z‖₂ s.t. lb ≤ z ≤ ub (bounded-variable
    least squares, Lawson–Hanson / Stark–Parker).

    Each pass solves the least-squares problem on the free columns, steps
    towards it up to the first blocking bound and releases the fixed
    component with the largest KKT violation. The result is verified by
    the KKT conditions. A cycle, `iteration_limit` passes or a failed
    verification switch to scipy's BVLS with a RuntimeWarning.

    Parameters
    ----------
    A, b : np.ndarray
        Matrix (A_k) and right-hand side.
    lb, ub : np.ndarray
        Bounds on z (-inf/inf if unbounded).
    free : np.ndarray
        Mask of the components of z that are estimated (others are 0).
    tolerance : float
        Scale of the bound and KKT tolerances.
    iteration_limit : int, default = 50
        Maximum number of passes.
    U : np.ndarray or None, default = None
        Orthonormal basis of the range of A (U_k), in which the free-column
        problems are solved.

    Returns
    -------
    tuple
        (z, fixed, fixed_at, status, kkt_error): the estimate, the
        components at a bound, the pass of fixing (0 = not fixed),
        "converged", "bvls fallback" or "not converged: <reason>", and the
        largest KKT violation of z.
    """
    b        = b.reshape(-1)
    U        = np.eye(A.shape[0]) if U is None else U
    var      = free & (lb < ub)                        # movable components
    F        = var & (np.isinf(lb) | np.isinf(ub) |    # free set
                      ((lb < 0) & (ub > 0)))
    z        = np.where(free, np.clip(0.0, lb, ub), 0.0)
    z        = np.where(var & ~F, np.where(np.abs(lb) <= np.abs(ub), lb, ub),
                        z)                             # feasible start
    fixed_at = np.where(free & ~F, 1, 0)
    tol_b    = tolerance * max(1.0, np.max(np.abs(b),       initial=0.0))
    tol_g    = tolerance * max(1.0, np.max(np.abs(A.T @ b), initial=0.0))
    e_kkt    = lambda z, g: float(np.max(np.where(~var, 0.0,
                                  np.where(z <= lb, -g,
                                  np.where(z >= ub,  g, np.abs(g)))),
                                  initial=0.0))        # largest violation
    visited  = set()
    reason   = "iteration_limit"
    for n_pass in range(1, 1 + max(1, iteration_limit)):
        # least-squares step on F, up to the first blocking bound
        while F.any():
            zs     = z.copy()
            zs[F]  = la.pinv(U.T @ A[:, F]) @ (U.T @ (b -
                     A[:, ~F] @ z[~F]))
            lo     = F & (zs < lb - tol_b)
            hi     = F & (zs > ub + tol_b)
            if not np.any(lo | hi):
                z[F]  = np.clip(zs[F], lb[F], ub[F])
                break
            t      = np.full(z.shape, np.inf)
            t[lo]  = (lb[lo] - z[lo]) / (zs[lo] - z[lo])
            t[hi]  = (ub[hi] - z[hi]) / (zs[hi] - z[hi])
            t_min  = max(0.0, np.min(t))
            hit    = t <= t_min + tolerance            # blocking bounds
            z[F]   = np.clip(z[F] + t_min * (zs[F] - z[F]), lb[F], ub[F])
            z[hit & lo], z[hit & hi] = lb[hit & lo], ub[hit & hi]
            F[hit], fixed_at[hit]    = False, n_pass
        # KKT conditions, else release the largest violation
        g      = A.T @ (A @ z - b)
        kkt    = np.where(var & ~F & (z == lb), -g,
                 np.where(var & ~F & (z == ub),  g, -np.inf)) - tol_g
        if not np.any(kkt > 0):
            e_z    = e_kkt(z, g)                       # full KKT check
            if e_z <= tol_g:
                return z.reshape(-1, 1), free & ~F, fixed_at, "converged", e_z
            reason = f"KKT error {e_z:.3e} > tol_g = {tol_g:.3e}"
            break
        k_rel                = int(np.argmax(kkt))
        F[k_rel]             = True
        fixed_at[k_rel]      = 0
        if (F.tobytes(), np.where(F, 0.0, z).tobytes()) in visited:
            reason = "cycle"
            F[k_rel], fixed_at[k_rel] = False, n_pass  # last feasible point
            break
        visited.add((F.tobytes(), np.where(F, 0.0, z).tobytes()))
    else:
        F[k_rel], fixed_at[k_rel] = False, n_pass      # last feasible point
    e_z      = e_kkt(z, A.T @ (A @ z - b))

    # scipy's BVLS on the movable components, else the last feasible point
    try:
        res  = lsq_linear(A[:, var], b - A[:, ~var] @ z[~var],
                          bounds=(lb[var], ub[var]), method="bvls")
        if res.status < 1:
            raise ValueError(res.message)
        z_s                       = z.copy()
        z_s[var]                  = np.clip(res.x, lb[var], ub[var])
        e_s                       = e_kkt(z_s, A.T @ (A @ z_s - b))
        if e_s > tol_g:
            raise ValueError(f"KKT error {e_s:.3e} > tol_g = {tol_g:.3e}")
        fixed                     = free & ~var
        fixed[np.flatnonzero(var)[res.active_mask != 0]] = True
        fixed_at[fixed & (fixed_at == 0)] = n_pass + 1
        fixed_at[~fixed]          = 0
        warnings.warn(f"Bounded least-squares active set did not converge "
                      f"({reason}); using scipy BVLS", RuntimeWarning)
        return z_s.reshape(-1, 1), fixed, fixed_at, "bvls fallback", e_s
    except (ValueError, np.linalg.LinAlgError) as e:
        warnings.warn(f"Bounded least-squares active set did not converge "
                      f"({reason}; scipy BVLS: {e}); the last feasible "
                      f"estimate is kept", RuntimeWarning)
        return (z.reshape(-1, 1), free & ~F, fixed_at,
                f"not converged: {reason}", e_z)

def _polish(
    A:         np.ndarray,      U:               np.ndarray,
    w:         np.ndarray,      u0:              np.ndarray,
    u:         np.ndarray,      lb:              np.ndarray,
    ub:        np.ndarray,      tolerance:       float,
    tol_set:   float
) -> tuple:
    """
    Polish a Step 2 solution u (α = 1): fix the components within
    `tol_set` (then `tolerance`) of a finite bound and project u0 onto
    A·u = w on the free set. The result is kept if its bound violation,
    ‖A·u - w‖₂ and the KKT sign violation are ≤ `tolerance`.

    Returns
    -------
    tuple
        (u, polished): the polished estimate (else u) and True/False.
    """
    u_s, u0, w = u.ravel(), u0.ravel(), w.ravel()
    B          = U.T @ A                               # A in the U_k basis
    for tol_s in (tol_set, tolerance):
        # active set and the projection on F
        at_lb  = np.isfinite(lb) & (u_s - lb <= tol_s)
        at_ub  = np.isfinite(ub) & (ub - u_s <= tol_s) & ~at_lb
        F      = ~(at_lb | at_ub)                      # free set
        I      = ~F & (lb < ub)                        # sign conditions
        u_p    = np.where(at_lb, lb, np.where(at_ub, ub, u_s))
        U_F, s_F, Vt_F = la.svd(B[:, F], full_matrices=False)
        k_F    = s_F > (max(B[:, F].shape) * np.finfo(float).eps *
                        np.max(s_F, initial=0.0))      # cutoff of la.pinv
        P_inv  = (Vt_F[k_F].T / s_F[k_F]) @ U_F[:, k_F].T
        u_p[F] = u0[F] + P_inv @ (U.T @ (w - A[:, ~F] @ u_p[~F]) -
                                  B[:, F] @ u0[F])
        u_c    = np.clip(u_p, lb, ub)                  # verify the clipped
        if (np.max(np.r_[lb - u_p, u_p - ub, 0.0]) > tolerance or
            np.linalg.norm(A @ u_c - w)            > tolerance):
            continue
        # bound multipliers μ = h - G·t ≥ 0 (t: free equality multipliers)
        y      = P_inv.T @ (u_p[F] - u0[F])            # u_F - u0_F = Pᵀ·y
        N      = la.null_space(U_F[:, k_F].T)          # B_Fᵀ·N = 0
        s      = np.where(at_lb, 1.0, -1.0)[I]
        h      = s * (u_p - u0 - B.T @ y)[I]
        G      = s.reshape(-1, 1) * (B[:, I].T @ N)
        t      = np.zeros(N.shape[1])
        if I.any() and N.shape[1] and np.min(h) < -tolerance:
            res = linprog(np.r_[t, 1.0],               # min max(-μ)
                          A_ub=np.c_[G, -np.ones(I.sum())], b_ub=h,
                          bounds=[(None, None)] * t.size + [(0.0, None)],
                          method="highs")
            t   = res.x[:-1] if res.status == 0 else t
        if np.max(np.r_[G @ t - h, 0.0]) <= tolerance:
            return u_c.reshape(-1, 1), True
    return u, False

def _cond(
    x:         np.ndarray = np.empty((0, 0), dtype=np.float64),
    norm:      str        = "e",
    tolerance: float                                           | None = None
) -> float:
    """
    Compute a custom condition number using the singular values of `x`.

    Parameters
    ----------
    x : np.ndarray, default = np.empty((0, 0), dtype=np.float64)
        Input matrix whose condition number is to be computed.
    norm : str, default = "e"
        Norm used to define the condition number. Only the first non-space
        character is used after lowercasing. Supported values are "e" and "f".
    tolerance : float or None, default = None
        Singular-value cutoff. Only singular values `s` satisfying
       `s > tolerance * np.max(s)` are retained.

    Returns
    -------
    float
        The requested condition number as a Python float. Returns `np.nan` if
        `x` is not two-dimensional, if it is empty, or if no singular values
        remain after applying the cutoff.

    Notes
    -----
    This function is intentionally not identical to `numpy.linalg.cond()`.
    """
    x         = np.asarray(x, dtype=np.float64)
    norm      = str(norm).strip().lower()[:1] or "e"
    tolerance = 1e-14 if tolerance is None else float(tolerance)

    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return np.nan
    s    = np.linalg.svd(x, compute_uv=False)
    smax = np.max(s)
    s    = s[s > tolerance * smax]
    if s.size == 0:
        return np.nan
    if norm == "e":
        return float(smax                  / np.min(s)                    )
    else:
        return float(np.sqrt(np.sum(s**2)) * np.sqrt(np.sum((1.0 / s)**2)))
