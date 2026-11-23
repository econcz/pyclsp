# CLSP — Convex Least Squares Programming

**Convex Least Squares Programming (CLSP)** is a two-step estimator for solving underdetermined, ill-posed, or structurally constrained least-squares problems. It combines pseudoinverse-based estimation with convex-programming correction (Lasso, Ridge, Elastic Net) to ensure numerical stability and interpretability. All rows of the system, the constraint rows included, are fitted in least squares; only bounds on the unknowns and the signs of the slack variables are enforced exactly (see [How CLSP Works](#how-clsp-works) and [Limitations](#limitations)). The package also provides numerical stability analysis and CLSP-specific diagnostics, including partial R², normalized RMSE (NRMSE), Monte Carlo t-tests for mean NRMSE, and heuristic condition-number-based sensitivity bands. All calculations use numpy.float64 precision.

The method is described in Bolotov (2025), see [Bibliography](#bibliography). Changes between versions are listed in [CHANGELOG.md](CHANGELOG.md).

## Contents

- [Installation](#installation)
- [How CLSP Works](#how-clsp-works): [Problem Types](#problem-types), [The Canonical Form](#the-canonical-form), [Notation](#notation), [Step 1](#step-1-pseudoinverse-estimate), [Step 2](#step-2-convex-correction), [Choosing α by Cross-Validation](#choosing-α-by-cross-validation), [Diagnostics](#diagnostics)
- [Examples](#examples):
  [1. CMLS (RP)](#1-quick-example-cmls-rp),
  [2. Non-Negativity Bounds](#2-ap-tm-with-non-negativity-bounds),
  [3. Inequality Constraints](#3-inequality-constraints-via-s-and-slack_sign),
  [4. Step 2](#4-step-2-α-ρ-residual-and-status),
  [5. Diagnostics](#5-diagnostics-summary-corr-ttest),
  [6. Cross-Validation](#6-choosing-α-by-cross-validation-cv_alpha),
  [7. Subspace Restriction](#7-subspace-restriction-with-z),
  [8. Residual-Sign Refinement](#8-residual-sign-refinement-r--1),
  [9. Truncation](#9-truncation-of-small-singular-values-rcond),
  [10. Grouped Totals and a Zero Diagonal](#10-ap-tm-with-grouped-totals-and-a-zero-diagonal),
  [11. Reading the Results](#11-reading-the-results)
- [User Reference](#user-reference): [`CLSP`](#the-clsp-class), [`solve()`](#solver-method-solve), [`corr()`](#correlogram-method-corr), [`ttest()`](#t-test-method-ttest), [`cv_alpha()`](#cross-validation-method-cv_alpha), [`summary()`](#summary-method-summarize-or-summary)
- [Warnings and Troubleshooting](#warnings-and-troubleshooting)
- [Limitations](#limitations)
- [What's New in 5.0.0](#whats-new-in-500)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements) · [Bibliography](#bibliography) · [License](#license)

## Installation

```bash
pip install pyclsp
```

The package requires Python ≥ 3.11 (the minimum of its dependency cvxpy ≥ 1.8.1; tested on Python 3.11, 3.12 and 3.13) and depends on numpy, scipy, cvxpy and the solvers ECOS, OSQP, SCS and CLARABEL (installed automatically).

## How CLSP Works

### In Plain Terms

CLSP writes a problem as one linear system `A`z = `b`. Its rows are of two kinds: *constraint rows* (`C`), such as known totals or linear restrictions on the unknowns, and *model rows* (`M`), such as observations of a regression or known cells of a table. Inequality restrictions are written as equalities with an additional slack variable (`S`).

The system is usually underdetermined (more unknowns than independent equations), inconsistent (no exact solution), or both. CLSP solves it in two steps:

1. **Step 1** computes a least-squares solution with a pseudoinverse, the minimum-norm one among all least-squares solutions (restricted to a subspace `Z` if one is given). If `bounds` on the unknowns are set, the bounded least-squares problem is solved by bounded-variable least squares (BVLS), an active-set method that moves variables to and from their bounds.
2. **Step 2** (optional) keeps the best achievable fit of Step 1 fixed and, among all solutions that attain it, chooses the one closest to the Step 1 estimate in a Lasso (L1), Ridge (L2) or Elastic Net sense, subject to the bounds.

Both steps work with the numerical range of `A`: singular values below a relative cutoff (`rcond`) are treated as zero. Afterwards, the package reports condition numbers, goodness-of-fit statistics, a row-deletion correlogram of the constraint rows, t-tests on the NRMSE, heuristic sensitivity bands, and a cross-validation of α over the model rows.

Please note that all rows of `A`, the constraint rows `C` included, are fitted *in least squares*: in an inconsistent system, the fit of the `C` rows is traded off against the fit of the `M` rows. Only `bounds` on x and the sign of the `S` slacks (`slack_sign`) are enforced exactly; a signed slack does not make its row a hard equality or inequality (see Example 3). A hard-constraint mode (`C`x + `S`y = `b`_C imposed exactly) is not implemented.

### Problem Types

The argument `problem` of `solve()` selects how `A` is built. The type is recognized by a substring of the lower-cased string (*'ap'*, *'tm'*, *'cmls'* or *'rp'* anywhere in it, so that, e.g., *'tmp'* also selects AP); the default `problem=""` is a general problem.

- **General problems** (`problem=""`): `A` = [`C` `S` 0; `M` 0 `Q`] is assembled from the supplied blocks, at least one of `C` and `M`; missing blocks are zero blocks of conformable size (no `S` means no slack columns, no `C` means no constraint rows). `b` is `b`_C followed by `b`_M. Examples 7–9 are general problems.
- **AP (TM): allocation and tabular-matrix problems** (`problem="ap"` or `"tm"`): the unknowns are the cells of a matrix X ∈ ℝ^{m×p}, x is X stored row by row, and `self.x` is returned as an `m` × `p` matrix. The constraint rows are built from `m`, `p`, `i` and `j`: `m`/`i` row-sum rows (each covering `i` consecutive rows of X) and `p`/`j` column-sum rows (each covering `j` consecutive columns), with `b` starting with these totals; a supplied `C` (with `m`·`p` columns) is appended below them, and `M` holds the model rows, e.g. the selection of known cells. `zero_diagonal=True` adds the min(`m`, `p`) model rows x_kk = 0 and appends the zeros to `b` itself; these rows are least-squares rows like all others, so an exact zero requires `bounds` with lo = hi = 0 (Example 10). Examples 2, 4, 6 and 10 are AP (TM) problems.
- **CMLS (RP): regression-type problems** (`problem="cmls"` or `"rp"`): `A` is assembled from the supplied `C`, `S` and `M` exactly as for a general problem, but both `C` and `M` are required, e.g. `M` = a regression design matrix with `b`_M = y and `C` = linear restrictions on the coefficients. Examples 1, 3, 5 and 11 are CMLS (RP) problems.

For related problem classes, see the author's companion packages `pytmpinv` (tabular matrix problems) and `pylppinv` (linear programming via regularized least squares), which are documented separately.

### The Canonical Form

```text
A z = b,      A = [ C  S  0 ]      z = [ x     ]      b = [ b_C ]
                  [ M  0  Q ]          [ y     ]          [ b_M ]
                                       [ y_res ]
```

- x ∈ ℝⁿ are the unknowns of interest (for AP (TM) problems, the m × p matrix X stored row by row, n = m·p), returned as `self.x`;
- `C` (k × n) are the constraint rows with right-hand side `b`_C; for AP (TM) problems, the row- and column-sum rows are built automatically from `m`, `p`, `i`, `j`, and a user-supplied `C` is appended below them;
- `S` (k × s) holds the coefficients of the slack variables y of the constraint rows, e.g. S_ii = +1 for a "≤" row and S_ii = -1 for a "≥" row, which is a restriction only if the slacks are signed (`slack_sign=True`);
- `M` (l × n) are the model rows with right-hand side `b`_M, e.g. a regression design matrix or the selection of known cells in an AP (TM) problem;
- `Q` (l × l) holds the coefficients of the residual slacks y_res of the model rows; it is empty (no columns) unless `r` > 1 and is rebuilt by every solve, see [Residual-Sign Refinement](#residual-sign-refinement-r--1);
- `Z` is a symmetric idempotent matrix (a projector) on the space of z; with `Z` = I (the default), Step 1 uses the Moore–Penrose inverse, otherwise the Bott–Duffin inverse restricts the Step 1 estimate to the range of `Z`.

The slack components (y, y_res) are returned together as `self.y`; `self.S_idx` gives the column range of y in `A`.

### Notation

| Symbol | Meaning | In the package |
|---|---|---|
| `A` | canonical design matrix [`C` `S` 0; `M` 0 `Q`] | `self.A`; `self.A_base` without `Q` |
| `b` = (`b`_C, `b`_M) | right-hand side: constraint rows, then model rows | `self.b` |
| `C` | constraint rows, k × n | `self.A[:k, :n]`, `self.C_idx` = (k, n) |
| `S` | coefficients of the slacks y of the constraint rows, k × s | columns `self.S_idx` of `A` |
| `M` | model rows, l × n | `self.A[k:, :n]` |
| `Q` | coefficients of the residual slacks y_res of the model rows, l × l (only for `r` > 1) | last l columns of `self.A` |
| `Z` | projector on the space of z (`Z` = I: Moore–Penrose; otherwise Bott–Duffin) | `self.Z` |
| z = (x, y, y_res) | all unknowns: x (n), the slacks y (s) and the residual slacks y_res | `self.z`; `self.x`; `self.y` = (y, y_res) |
| `A`_k | `A`·`Z` with the singular values σ ≤ `rcond`·σ_max set to zero (from the SVD `A`·`Z` = UΣVᵀ; U_k, Σ_k, V_k are the retained parts); `A`·`Z`·D with `scale=True` | `self.rank`, `self.sv_discarded` |
| zhat0 | unbounded Step 1 estimate `A`_k⁺`b` (D·`A`_k⁺`b` with `scale=True`), the anchor of Step 2 | `self.zhat0` |
| zhat | Step 1 estimate: zhat0 without bounds, otherwise the bounded least-squares (BVLS) solution on `A`_k | `self.zhat` |
| w* | fitted vector `A`_k·zhat, the target of Step 2 (`A`_k z = w*) | not stored; see `self.rho` and `self.status["eq_error"]` |
| ρ | ‖`b` − w*‖₂, the smallest residual within the numerical range of `A` under the bounds | `self.rho` |
| α | weight of the Step 2 objective (0: Lasso, 1: Ridge, in between: Elastic Net) | `self.alpha` |
| m, p | dimensions of X in AP (TM) problems (n = m·p) | `self.x` is `m` × `p` |
| k | number of constraint rows (for AP (TM), the built totals plus the rows of a supplied `C`); the subscript k of `A`_k, U_k, Σ_k, V_k refers to the retained singular values, not to this number | `self.C_idx[0]` |
| n, s, l | numbers of unknowns x, of slacks y, and of model rows | `self.C_idx[1]`, `self.S_idx`, `self.A.shape[0]` − k |

### Step 1: Pseudoinverse Estimate

```text
A_k   = U_k Σ_k V_kᵀ    (SVD of A·Z, singular values σ ≤ rcond·σ_max set to zero)
zhat0 = A_k⁺ b
```

One numerical-rank policy is used throughout: `A`_k, i.e. `A`·`Z` with the singular values σ ≤ `rcond`·σ_max set to zero (computed from one SVD), replaces `A`·`Z` in the Step 1 estimate, in the bounded least-squares problem (including its reduced free-column problems) and its KKT check, in the target w* and ρ, and in the Step 2 constraint. No second cutoff is applied anywhere: a direction kept in `A`_k is kept in every reduced problem of the bounded first step. The default cutoff (`rcond=False`) is sqrt(max(`A`.shape)·eps), which reproduces the effective cutoff of the normal-equations form and is robust to the structural rank deficiency of AP (TM) problems; `rcond=True` uses `tolerance`, and a float is used as given. The effective rank is reported in `self.rank` and the discarded singular values in `self.sv_discarded` (empty if none). zhat0 equals the Bott–Duffin form (`Z`·`A`ᵀ`A`·`Z`)⁺`Z`·`A`ᵀ`b` with the same cutoff but does not square the condition number of `A`·`Z`. For `Z` = I, zhat0 is the minimum-norm least-squares solution of `A`_k z = `b`, which is that of `A`z = `b` when no singular value is discarded.

A discarded singular value is treated as an exact zero: for example, with `A` = diag(1, 10⁻⁹) and `b` = (1, 2), the default cutoff gives x = (1, 0) and ρ = 2, although (1, 2·10⁹) fits `b` exactly. Unbounded variables (`bounds=(None, None)`) are the same problem as no bounds and give identical results.

#### Bounds and Signed Slacks (Bounded-Variable Least Squares)

If `bounds` on x with a finite side are set and/or `slack_sign=True` (S-slacks ≥ 0), Step 1 solves the bounded least-squares problem

```text
min ‖b − A_k z‖₂   subject to   lb ≤ z ≤ ub   (z restricted to the coordinates admitted by Z)
```

by bounded-variable least squares (the Lawson–Hanson active-set method in the bounded form of Stark and Parker). The iterates are always feasible:

1. Start from a feasible point: a variable with an infinite side, or with 0 inside its bounds, is free at the point of [lb, ub] nearest to 0; any other variable is fixed at the bound nearest to 0 (a variable with lo = hi is fixed throughout).
2. Solve the least-squares problem on the free columns F with the fixed values v moved to the right-hand side, z_F = (U_kᵀ`A`_k,F)⁺ U_kᵀ(`b` − `A`_k,B v), i.e. on the columns of `A`_k expressed in the orthonormal basis U_k of its range (which is exact, since `A`_k = U_k Σ_k V_kᵀ, and keeps rounding noise outside the range of `A`_k from entering the reduced problem). The pseudoinverse uses only a machine-precision cutoff, max(shape)·eps relative to the largest singular value of U_kᵀ`A`_k,F itself, not a cutoff tied to σ_max of `A`. For example, with `A` = [[1, 10⁻⁸], [0, 0]], `b` = (1, 0) and x₁ fixed at 0 (bounds [(0, 0), (0, None)]), the remaining column has the singular value 10⁻⁸, which belongs to the retained direction of `A`_k, and x₂ = 10⁸ fits `b` exactly. If z_F lies within the bounds (up to tol_b = `tolerance`·max(1, max|`b`|)), accept it. Otherwise, step from the current point towards z_F up to the first blocking bound (interpolation), fix the blocking variables at that bound, and repeat 2.
3. Check the KKT conditions with the gradient g = `A`_kᵀ(`A`_k z − `b`): a variable fixed at its lower bound with g_j < −tol_g, or at its upper bound with g_j > tol_g (tol_g = `tolerance`·max(1, ‖`A`_kᵀ`b`‖_∞)), could improve the fit by moving into the interior. If there is none, verify the full KKT conditions (4.); otherwise, release the one with the largest violation and go to 2.
4. Verify the result by the full KKT conditions on the `A`_k problem: |g_j| ≤ tol_g for the free variables, g_j ≥ −tol_g at a lower bound and g_j ≤ tol_g at an upper bound. tol_g follows `tolerance`, so a tighter `tolerance` is enforced on the free variables as well. The largest violation of the returned zhat is reported in `self.status["kkt_error"]` (without bounds, max|g_j| of zhat0).

Each pass (2 and 3) counts against `iteration_limit`. If the method does not converge (a repeated free set, i.e. a cycle, or `iteration_limit` passes) or the verification fails, the same problem (`A`_k, bounds) is solved by `scipy.optimize.lsq_linear(..., method="bvls")` with a *RuntimeWarning*, and its solution is verified in the same way; if that fails too, the last feasible point is kept with a *RuntimeWarning*. A bound-violating estimate is never returned. The outcome is reported in `self.status["step1"]` (*'converged'*, *'bvls fallback'* or *'not converged: <reason>'*). On convergence (*'converged'* or *'bvls fallback'*), zhat satisfies the KKT conditions of the bounded least-squares problem on `A`_k within tol_g, i.e. it is a minimizer up to this tolerance: the fitted vector `A`_k·zhat is unique, whereas zhat itself is unique only if the free columns have full column rank (the pseudoinverse then takes the minimum-norm free part). The variables at a bound at termination are reported in `self.fixed` and the pass at which each was fixed in `self.fixed_at` (0 = free). Bounds require a diagonal (0/1) `Z`, and coordinates excluded by `Z` (Z_jj = 0) must admit 0.

`self.zhat0` stores the unbounded estimate `A`_k⁺`b`; it is the anchor of Step 2. Without bounds, `self.zhat0` equals `self.zhat`.

#### Residual-Sign Refinement (r > 1)

If `r` > 1 and the problem has model rows, Step 1 is repeated with

```text
Q = diag(−sign(e_M)),   e_M = b_M − M x,
```

where the entries of e_M with |e_i| ≤ `tolerance`·max(1, ‖`b`‖₂) are treated as exact zeros, so that rounding noise does not trigger the iteration. The iteration stops when the NRMSE changes by less than `tolerance` (or is *inf* in two successive iterations, i.e. sd(`b`) = 0), after `r` iterations, or as soon as the number of iterations exceeds `iteration_limit` (i.e., after at most `iteration_limit` + 1 iterations); `self.r` reports the iterations performed (see Example 8). `Q` is appended anew by every solve to `A` without `Q` (the stored specification), so that a fit with `r` = 1 after one with `r` = 3 equals a fresh fit with `r` = 1.

The signs of the residual slacks do not affect x, because y_res is unrestricted. The refinement acts only if e_M ≠ 0, i.e., when the model rows cannot be fitted exactly together with the constraint rows; in a consistent system, the estimate equals that of `r` = 1. When it acts, the misfit of the model rows is moved into y_res: for `Z` = I and nonzero residuals in all model rows, (x, y) minimizes ‖x‖² + ‖y‖² + ‖`b`_M − `M`x‖² subject to `C`x + `S`y = `b`_C (if these rows are consistent), i.e., the constraint rows are then met exactly and x is a Ridge-type compromise rather than the least-squares fit of `M`. Accordingly, `nrmse` and `residual` then refer to the augmented system (and may be close to 0), whereas `nrmse_partial` and `r2_partial` still measure `b`_M − `M`x.

### Step 2: Convex Correction

If `final=True` (the default), Step 2 computes

```text
w*  = A_k zhat,   zhat ∈ argmin ‖b − A_k z‖₂   subject to   lb ≤ z ≤ ub   (Step 1)
ρ   = ‖b − w*‖₂
z*  = argmin (1 − α)‖z − z_a‖₁ + α‖z − z_a‖₂²   subject to   A_k z = w*,   lb ≤ z ≤ ub
```

with the anchor z_a = `self.zhat0` (which equals `self.zhat` without bounds). w* is the fitted vector of the bounded least-squares problem of Step 1: it is unique (the projection of `b` onto the convex set {`A`_k z : lb ≤ z ≤ ub}), equal to `A`_k`A`_k⁺`b` without bounds. Hence ρ (`self.rho`) is the smallest residual norm within the numerical range of `A` (singular values above the cutoff) under the bounds, and Step 2 "corrects only the correctable deviance": if the system is consistent and unbounded, `A`_k z = w* coincides with `A`z = `b`. Since `A`_k vanishes outside the range of `Z`, Step 2 keeps those components at the anchor (0 for a diagonal `Z`). `self.residual` is ‖`b` − `A`z‖₂ with the original `A`; it equals ρ (up to solver accuracy) unless singular values were discarded or Step 2 fell back to zhat.

Properties:

- the program is always feasible, since z_LS satisfies its constraints;
- for α > 0 the objective is strictly convex and z* is unique; for α = 0 (Lasso) several minimizers may exist and the solver returns one of them;
- the residual ‖`b` − `A`_k z*‖₂ = ρ is identical for all α, which only selects among the equally well-fitting solutions;
- for α = 1 and `Z` = I, z* is the minimum-norm solution among all (bounded) least-squares solutions, since zᵀ·zhat0 is constant on {z : `A`_k z = w*};
- without active `bounds`/`slack_sign` and with a diagonal `Z`, zhat already satisfies `A`_k z = w* and is the anchor, so z* = zhat for every α (Step 2 returns the anchor without a solver, see below, for consistent and inconsistent systems alike); Step 2 changes the estimate only when bounds or signed slacks are present.

If the anchor is feasible, i.e. its bound violation and ‖`A`_k z − w*‖₂ for z = z_a and for z_a clipped to the bounds are ≤ `tolerance`·max(1, ‖`b`‖₂), the clipped anchor is returned without calling a solver (the clipped anchor must be checked as well: with `C` = [10⁸, 1], `b` = 1 and x₁ = 0, z_a ≈ (10⁻⁸, 10⁻¹⁶) is within the tolerance of the bound, but clipping x₁ to 0 removes 10⁸·10⁻⁸ = 1 from the fit, so the anchor is not used): the objective is 0 there, so z_a is the minimizer for every α (the unique one for α > 0); `self.status["final"]` is then *'optimal'* and `self.status["solver"]` *'none (anchor feasible)'*. Step 2 is thus exact without bounds and whenever the bounds do not cut off the anchor; solved numerically, such programs had errors of order 10⁻⁵ in z (e.g. `C` = [1, 2], `b` = 5, 0 ≤ x₁ ≤ 1, x₂ ≥ 0, α = 1: OSQP returned *optimal_inaccurate*, and the CLARABEL retry, whose tolerances of ~10⁻⁸ apply to the squared objective ‖z − z_a‖₂², gave z ≈ (0.99997, 2.00001) instead of z_a = (1, 2)). Otherwise, the solver is ECOS for α = 0, OSQP for α = 1 and CLARABEL for 0 < α < 1, unless `solver=` is passed. OSQP is run with tight tolerances, `eps_abs` = `eps_rel` = 10⁻¹⁰ and `max_iter` = 100000 (solution polishing is on by default in CVXPY), unless these options are passed: with the CVXPY defaults (10⁻⁵), Ridge solutions passed the verification below with residual gaps of up to 4·10⁻⁵ to the bounded least-squares optimum, and some benchmark cases stopped at the iteration limit. If OSQP (as the default solver) fails, returns *optimal_inaccurate* or fails verification, the problem is solved once more by CLARABEL before falling back (if the retry fails, a verified *optimal_inaccurate* OSQP solution is kept); `self.status["solver"]` reports the solver actually used and `self.status["retry"]` the reason of the retry (*None* without a retry). On 800 random bounded problems (the four regimes of the BVLS benchmark of the test suite, seeds 1000–1199, α = 1), this reduced the largest residual gap to the BVLS optimum from 4.0·10⁻⁵ to 6.4·10⁻¹⁰ and the fallbacks to `zhat` from 15 to 0 (3 problems were solved by the CLARABEL retry, and 158 had a feasible anchor). The solution is verified with the tolerance tol_v = sqrt(`tolerance`)·max(1, ‖`b`‖₂) (≈ 1.2·10⁻⁴·max(1, ‖`b`‖₂) by default): the bound violation of the solver output must be ≤ tol_v, and, after clipping to the bounds, ‖`A`_k z − w*‖₂ ≤ tol_v. If the solver (and the retry) fails, returns a status other than *optimal* or *optimal_inaccurate*, or the solution fails verification, z = zhat is returned with a *RuntimeWarning* stating the reason. The same check (bound violation and ‖`A`_k z − w*‖₂ ≤ tol_v) is applied once more to the returned z on every path — solver, polish and feasible anchor — so that `self.status["final"]` is never *'optimal'* with `self.status["eq_error"]` > tol_v (*'fallback: final verification failed: …'* otherwise). CVXPY's *UserWarning* "Solution may be inaccurate" is suppressed: an *optimal_inaccurate* solution is verified like any other, and other inaccurate statuses lead to the retry or the fallback. The verification establishes approximate feasibility, not optimality of the Step 2 objective; for α = 1, optimality is established by the polish below. The outcome is reported in `self.status`, together with tol_v and the achieved errors of the returned z (`"bound_error"`, `"eq_error"` = ‖`A`_k z − w*‖₂), and the residual of the returned z in `self.residual`.

For α = 1, a verified solver solution is polished on its active set (in u = D⁻¹z with `scale=True`). The variables within tol_v of a finite bound (the signed slacks at 0 included) are fixed at the bound, and the remaining equality-constrained projection, min ‖z_F − z_a,F‖₂ subject to `A`_k[:, F] z_F = w* − `A`_k[:, ¬F] z_¬F, is solved exactly in the basis U_k of the range of `A`_k: z_F = z_a,F + P⁺(r − P z_a,F) with P = U_kᵀ`A`_k[:, F], r = U_kᵀ(w* − `A`_k[:, ¬F] z_¬F) and the pseudoinverse cutoff of the reduced problems of Step 1. The polished point is accepted only if its bound violation and ‖`A`_k z − w*‖₂ (after clipping to the bounds) are ≤ `tolerance`·max(1, ‖`b`‖₂) and the KKT sign conditions hold for the fixed variables within the same tolerance: the bound multipliers implied by the projection must be ≥ 0 at a lower and ≤ 0 at an upper bound (if the equality multipliers are not unique, a small linear program searches for multipliers with the correct signs). Otherwise, the active set is detected once more with the tolerance `tolerance`·max(1, ‖`b`‖₂), and if this fails as well, the verified solver solution is kept. An accepted point satisfies the optimality conditions of the Step 2 program up to rounding errors. `self.status["polish"]` reports the outcome (*True* or *False*; *None* if Step 2 is skipped); the polish is not applied for α < 1 or to a feasible anchor, which is already exact. On the 800 problems above, all 642 solver solutions were polished: the largest deviation of z from the exact projection (computed by enumeration of the active sets) fell from 4.0·10⁻⁸ to 4.6·10⁻¹³ (deviations above 10⁻¹²: from 65 to 0), and the largest residual gap to the BVLS optimum from 6.4·10⁻¹⁰ to 8.8·10⁻¹⁴.

If α is not given, α = 1 (Ridge) is used: z* is then unique and, for `Z` = I, the minimum-norm (bounded) least-squares solution, which needs no tuning. α must be a single value in [0, 1]; the selection of α by NRMSE was removed in 5.0.0, because all α attain the same residual. α can instead be chosen by cross-validation over the model rows with `cv_alpha()`.

### Choosing α by Cross-Validation

`cv_alpha()` splits the model rows (`M`) at random into K folds (the constraint rows `C` are never held out), refits the model without each fold for every candidate α (with the fitted settings), predicts the held-out `b`_M by `M`x, and reports the mean and standard deviation of the prediction RMSE over the folds. Ties are resolved in favour of α = 1. The comparison is only informative if α can change the estimate, so no refit is made and `informative` is *False* (with `best` = 1 and a *RuntimeWarning*) if there are no model rows, fewer model rows than folds, `final=False`, or no active bounds or signed slacks with a diagonal `Z` (Step 2 then returns zhat for every α). After the refits, the result is also uninformative if all α give the same mean RMSE (relative difference ≤ `tolerance`).

Every fold fit is checked: it is valid only if Step 1 converged (`"step1"` *'converged'* or *'bvls fallback'*) and Step 2 returned a verified solution (not a fallback to zhat). A failed fold fit is retried once with CLARABEL. An α with a failed fold is invalid: it is excluded from the selection, its mean and standard deviation are *nan*, and it is reported in `valid` and `failures` (α, fold, reason) with a *RuntimeWarning*. Candidates with a failed fold are excluded, never averaged over different fold subsets, so that all compared means rest on the same folds. If fewer than two α remain valid, the result is not informative (`best` = 1, or *nan* if no α or α = 1 itself is invalid).

The scores are prediction diagnostics under the chosen fold design. Random folds of the model rows assume that holding out rows is appropriate for the application: time-dependent data need a suitable split design (e.g. blocked or forward folds, which `cv_alpha()` does not provide), and constraint rows derived from the outcomes of the model rows (e.g. totals computed from the observed `b`_M) leak the held-out outcomes into the fit and should be avoided or rebuilt per fold.

### Diagnostics

- **Condition numbers:** `kappaC` = κ(C_canon), where C_canon = [`C` `S` 0] are the constraint rows of `A`; `kappaB` = κ(`A`·C_canon⁺); `kappaA` = κ(`A`). κ = σ_max / σ_min over the singular values σ > `cond_tolerance`·σ_max (default 1e-14), which keeps the results reproducible across platforms.
- **Goodness of fit:** NRMSE = ‖`b` − `A`z‖₂ / (√n·sd(`b`)), with the population standard deviation; `nrmse_partial` and `r2_partial` = 1 − ‖`b`_M − `M`x‖² / ‖`b`_M − mean(`b`_M)‖² use the model rows only. A zero standard deviation yields *inf* (NRMSE) or *nan* (R²).
- **RMSA and correlogram:** the total root mean square alignment of the constraint rows c_i of C_canon, RMSA = sqrt(2 / (k(k−1)) · Σ_{i<j} cos²(c_i, c_j)), and the row-wise RMSA_i = sqrt(1 / (k−1) · Σ_{j≠i} cos²(c_i, c_j)). `corr()` deletes each constraint row in turn, refits the model with the fitted settings, and reports the changes of κ, NRMSE and the norms of zhat, z and x (k refits, which can be slow for many constraint rows). Refits whose Step 1 did not converge or whose Step 2 fell back to zhat are counted in `self.rmsa_failures` and reported by one summary *RuntimeWarning* (count and reasons) after the loop; the same applies to the refits of `ttest(simulate=True)` (`"failures"` in its result).
- **t-test:** `ttest()` builds a sample of `sample_size` NRMSE values under H₀ and runs a one-sample t-test of H₀: *mean of the sampled NRMSE = observed NRMSE*, t = (NRMSE − mean) / (sd/√N), df = N − 1. The bootstrap (default) resamples the fitted residuals with replacement; their expected sum of squares equals the observed one, so the null mean is close to the observed NRMSE by construction, and the test is a check of the resampling variability rather than a goodness-of-fit test. The simulation (`simulate=True`) replaces `b` (or only `b`_M, with `partial=True`) by draws from a standard normal, uniform or Laplace distribution and re-estimates the model with the fitted settings; it tests whether the observed NRMSE equals the mean NRMSE obtained when the right-hand side is pure noise. The test compares the observed value with the mean of the simulated values, not with their spread, i.e. it does not locate the observed value within the null sample. A larger `sample_size` increases the power of the test for a fixed nonzero difference between the observed value and the null mean; the p-values of finite samples need not decrease monotonically with `sample_size`, and they do not tend to zero if the difference is zero.
- **Sensitivity bands:** z ∓ |z|·κ(`A`)·‖`b` − `A`z‖₂ / ‖`b`‖₂ (and the corresponding parts for x and y). They are a heuristic, condition-weighted measure of sensitivity: they have zero width for exact fits or zero components and imply no statistical coverage.

## Examples

Each example below is self-contained; the output shown was produced by running it with pyclsp 5.0.0 (numpy 64-bit floats; the last digits of solver-dependent values may differ across platforms and solver versions).

### 1. Quick Example: CMLS (RP)

```python
import   numpy as     np
from     clsp  import CLSP

# CMLS (RP), based on known stationary points for y = D @ x + e, x to be estimated

seed     = 123456789
rng      = np.random.default_rng(seed)

# sample (dataset)
k        = 500                                               # number of observations in D
p        = 6                                                 # number of regressors
c        = 1                                                 # sum of coefficients
D        = np.empty((k, p))
D[:, 0 ] = 1.0                                               # constant
D[:, 1:] = rng.normal(size=(k, p - 1))                       # D.,j ~ N(0,1), 2 <= j <= p
b_true   = rng.normal(size=p)                                # b_true ~ N(0,1)
b_true   = (b_true / b_true.sum()) * c
e        = rng.normal(size=(k, 1))                           # e ~ N(0,1)
y        = (D @ b_true).reshape(-1, 1) + e                   # y_t = D @ b_true + e_t

# model
b        = np.vstack([
               np.asarray([c]),                              # c (the sum of coefficients)
               np.zeros((k - 2, 1)),                         # zeros
               np.zeros((k - 1, 1)),                         # zeros
               y                                             # values of y_t
           ])
C        = np.vstack([
               np.ones((1, p)),                              # a row of ones
               np.diff(D, n=2, axis=0),                      # the 2nd differences
               np.diff(D, n=1, axis=0)                       # the 1st differences
           ])
S        = np.block([
               [np.zeros((  1, k-2))],                       # a zero vector
               [np.diag(np.sign(np.diff(y.ravel(), n=2)))],  # a diagonal sign matrix
               [np.zeros((k-1, k-2))],                       # a zero matrix
           ])
model    = CLSP().solve(
               problem="cmls", b=b, C=C, S=S, M=D,
               r=1,                                          # a solution without refinement
               alpha=1.0                                     # unique, minimum-norm (α = 1)
           )

# results
print("true beta (x_M):")
print(np.round(np.asarray(b_true).flatten(), 4))
print("beta hat (x_M hat):")
print(np.round(model.x.flatten(), 4))
print("sum of beta hat (c = 1 in the first row of C):")
print(np.round(model.x.sum(), 4))

model.summary(display=True)

print("  Bootstrap t-test (NRMSE_partial):")
for kw, val in model.ttest(sample_size=30,                   # NRMSE_partial sample
                           seed=seed, distribution="normal", # seed and distribution
                           partial=True).items():
    print(f"    {kw}: {float(val):.6f}")
```

Output:

```text
true beta (x_M):
[ 0.1687 -0.3593  0.5969 -0.0628 -0.1742  0.8307]
beta hat (x_M hat):
[ 0.2024 -0.1121  0.2095 -0.0396 -0.0296  0.2611]
sum of beta hat (c = 1 in the first row of C):
0.4917
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                     none
  Slack sign (S ≥ 0):            False
  Column scaling:                False
  Fixed variables:                   0

Numerical Stability:
  kappaC:                     138.6109
  kappaB:                      22.4354
  kappaA:                     127.8287
  rank:                            504

Goodness of Fit:
  r2_partial:                   0.2933
  nrmse:                        0.9028
  nrmse_partial:                0.8406
  z_lower:        min= -296.2716  max=   -0.0982  mean=  -79.7331  sd=   61.1772
  z_upper:        min=    0.0999  max=  291.1651  mean=   78.6599  sd=   60.0930
  x_lower:        min=  -29.7780  max=   -3.4348  mean=  -16.2993  sd=   10.9407
  x_upper:        min=    3.3756  max=   30.3003  mean=   16.4632  sd=   11.2232
  y_lower:        min= -296.2716  max=   -0.0982  mean=  -80.4974  sd=   61.1349
  y_upper:        min=    0.0999  max=  291.1651  mean=   79.4092  sd=   60.0520
  Bootstrap t-test (NRMSE_partial):
    p_one_left: 0.311434
    p_one_right: 0.688566
    p_two_sided: 0.622867
    nrmse: 0.840638
    mean_null: 0.842912
    std_null: 0.025063
    failures: 0.000000
```

The first row of `C` (a row of ones with right-hand side c = 1) is a least-squares row like all other rows of `A`, so the sum of the coefficients is fitted jointly with the 998 difference rows and the 500 observations rather than imposed: here, it equals 0.4917. To impose a restriction exactly, use `bounds` (sign and box restrictions on x); for linear restrictions, see Example 3.

### 2. AP (TM) with Non-Negativity Bounds

```python
import   numpy as     np
from     clsp  import CLSP

# AP (TM), based on known row and column sums of X ∈ ℝ^{3×3}, X to be estimated

# model
m, p     = 3, 3                                              # dimensions of X
b        = np.array([1, 10, 10,                              # row sums
                     1, 10, 10])                             # column sums
model_u  = CLSP().solve(problem="ap", b=b, m=m, p=p)         # x unbounded
model_b  = CLSP().solve(problem="ap", b=b, m=m, p=p,
                        bounds=(0, None))                    # x ≥ 0

# results
print("X hat (unbounded):")
print(np.round(model_u.x, 4))
print("X hat (bounds=(0, None)):")
print(np.round(model_b.x, 4))
print("fixed components of z:", np.flatnonzero(model_b.fixed),
      "at pass", model_b.fixed_at[model_b.fixed])
print("row sums:   ", np.round(model_b.x.sum(axis=1), 4))
print("column sums:", np.round(model_b.x.sum(axis=0), 4))
print("residual:   ", f"{model_b.residual:.6f}")
```

Output:

```text
X hat (unbounded):
[[-1.6667  1.3333  1.3333]
 [ 1.3333  4.3333  4.3333]
 [ 1.3333  4.3333  4.3333]]
X hat (bounds=(0, None)):
[[0.   0.5  0.5 ]
 [0.5  4.75 4.75]
 [0.5  4.75 4.75]]
fixed components of z: [0] at pass [1]
row sums:    [ 1. 10. 10.]
column sums: [ 1. 10. 10.]
residual:    0.000000
```

Without bounds, the minimum-norm estimate contains a negative cell (x[0,0] = −1.6667). With `bounds=(0, None)`, x[0,0] (component 0 of z) is fixed at 0 in the first pass of Step 1 and the remaining cells are re-estimated, so that the estimate is non-negative and reproduces the row and column sums [1, 10, 10] exactly. Step 2 (with the default α = 1) returns the same matrix here, which is the minimum-norm non-negative solution of the row and column sums.

### 3. Inequality Constraints via `S` and `slack_sign`

```python
import   numpy as     np
from     clsp  import CLSP

# CMLS (RP) with an inequality restriction beta_1 + beta_2 ≤ 1 encoded in S

seed     = 1
rng      = np.random.default_rng(seed)

# sample (dataset)
k        = 50                                                # number of observations
D        = np.column_stack([np.ones(k),                      # constant
                            rng.normal(size=k)])             # regressor ~ N(0,1)
b_true   = np.array([0.8, 0.6])                              # beta_1 + beta_2 = 1.4
y        = D @ b_true + 0.1 * rng.normal(size=k)

# model: beta_1 + beta_2 + s = 1, where S_11 = +1 encodes ≤ if s ≥ 0
C        = np.array([[1.0, 1.0]])
S        = np.array([[1.0]])
b        = np.concatenate([[1.0], y])
for label, weight, slack_sign in [("free slack     ",   1.0, False),
                                  ("s ≥ 0          ",   1.0, True ),
                                  ("s ≥ 0, row x1e3", 1.0e3, True )]:
    model = CLSP().solve(problem="cmls", C=weight * C, S=weight * S, M=D,
                         b=np.concatenate([[weight * b[0]], y]),
                         slack_sign=slack_sign, alpha=1.0)
    print(f"{label}: beta hat = {np.round(model.x.ravel(), 4)}, "
          f"sum = {model.x.sum():.4f}, s = {model.y[0, 0]:.4f}, "
          f"kappaA = {model.kappaA:.1f}")
```

Output:

```text
free slack     : beta hat = [0.7887 0.5955], sum = 1.3843, s = -0.3843, kappaA = 7.3
s ≥ 0          : beta hat = [0.781  0.5857], sum = 1.3668, s = 0.0000, kappaA = 7.3
s ≥ 0, row x1e3: beta hat = [0.6197 0.3803], sum = 1.0000, s = 0.0000, kappaA = 464.7
```

With a free slack (`slack_sign=False`), the inequality row is absorbed by s and has no effect: the estimate equals ordinary least squares, and s < 0 means that the row is satisfied as "≥". With `slack_sign=True`, s is fixed at 0 in Step 1, and the row becomes an equality *least-squares* row on its violated side: it pulls the sum towards 1 but, as all rows of `A`, is traded off against the 50 model rows (sum = 1.3668). An inequality row is thus free on its feasible side and soft on its infeasible side. Scaling the row (`C`, `S` and `b`_C) by a large weight makes it nearly binding (sum ≈ 1), at the cost of a larger κ(`A`); this is a penalty approach, not a hard constraint.

### 4. Step 2: α, ρ, Residual and Status

```python
import   numpy as     np
from     clsp  import CLSP

# AP (TM) with inconsistent totals (row sums: 21, column sums: 22) and x ≥ 0

b        = np.array([1, 10, 10,                              # row sums
                     1, 10, 11])                             # column sums
for alpha in (0.0, 0.5, 1.0):                                # Lasso, Elastic Net, Ridge
    model = CLSP().solve(problem="ap", b=b, m=3, p=3,
                         bounds=(0, None), alpha=alpha)
    st    = model.status
    print(f"alpha = {alpha}: step1 = {st['step1']}, final = {st['final']}, "
          f"solver = {st['solver']}, eq_error ≤ tol_v: "
          f"{st['eq_error'] <= st['tol_v']}")
    print(f"  rho = {model.rho:.4f}, residual = {model.residual:.4f}")
    print(np.round(model.x, 4))
print("anchor zhat0 (unbounded Step 1 estimate):")
print(np.round(model.zhat0[:9].reshape(3, 3), 4))
```

Output:

```text
alpha = 0.0: step1 = converged, final = optimal, solver = ECOS, eq_error ≤ tol_v: True
  rho = 0.4082, residual = 0.4082
[[0.     0.4789 0.6878]
 [0.4167 4.6772 5.0728]
 [0.4167 4.6772 5.0728]]
alpha = 0.5: step1 = converged, final = optimal, solver = CLARABEL, eq_error ≤ tol_v: True
  rho = 0.4082, residual = 0.4082
[[0.     0.4167 0.75  ]
 [0.4167 4.7083 5.0417]
 [0.4167 4.7083 5.0417]]
alpha = 1.0: step1 = converged, final = optimal, solver = OSQP, eq_error ≤ tol_v: True
  rho = 0.4082, residual = 0.4082
[[0.     0.4167 0.75  ]
 [0.4167 4.7083 5.0417]
 [0.4167 4.7083 5.0417]]
anchor zhat0 (unbounded Step 1 estimate):
[[-1.7222  1.2778  1.6111]
 [ 1.2778  4.2778  4.6111]
 [ 1.2778  4.2778  4.6111]]
```

The row sums (21) and column sums (22) cannot both be met, so the smallest achievable residual under x ≥ 0 is ρ = 0.4082. All three values of α attain it (`residual` = `rho`); α only selects among the equally well-fitting non-negative solutions by their L1 and/or L2 distance to the unbounded anchor `zhat0`, which is why the Lasso solution (α = 0) differs from the others. `status` reports the outcome of the bounded first step (`"step1"`), the second-step outcome (`"final"`) and the solver used, and whether the achieved error ‖`A`_k z − w*‖₂ (`"eq_error"`) is within the verification tolerance (`"tol_v"`); it also contains the source of w* (`"rho"`), the largest KKT violation of the first step (`"kkt_error"`), the reason of a CLARABEL retry (`"retry"`, *None* here), whether the α = 1 solution was polished on its active set (`"polish"`) and the largest bound violation of z (`"bound_error"`).

### 5. Diagnostics: `summary()`, `corr()`, `ttest()`

```python
import   numpy as     np
from     clsp  import CLSP

# CMLS (RP) with three linear restrictions on beta, followed by diagnostics

seed     = 42
rng      = np.random.default_rng(seed)

# sample (dataset)
k        = 40                                                # number of observations
D        = np.column_stack([np.ones(k),                      # constant
                            rng.normal(size=(k, 2))])        # regressors ~ N(0,1)
b_true   = np.array([1.0, 0.7, 0.3])
y        = D @ b_true + 0.5 * rng.normal(size=k)

# model
C        = np.array([[0.0, 1.0, 1.0],                        # beta_2 + beta_3 = 1
                     [1.0, 1.0, 0.0],                        # beta_1 + beta_2 = 1.7
                     [1.0, 1.0, 1.0]])                       # beta_1 + beta_2 + beta_3 = 2
b        = np.concatenate([[1.0, 1.7, 2.0], y])
model    = CLSP().solve(problem="cmls", C=C, M=D, b=b, alpha=1.0)

# results
print("beta hat:", np.round(model.x.ravel(), 4))
print("C @ beta hat:", np.round(C @ model.x.ravel(), 4))
print("x_lower:", np.round(model.x_lower.ravel(), 4))
print("x_upper:", np.round(model.x_upper.ravel(), 4))

print("Correlogram:")
for kw, val in model.corr().items():                         # one refit per C row
    print(f"  {kw:<13}", np.round(val, 4))

model.summary(display=True)                                  # includes RMSA after corr()

for label, kwargs in [("bootstrap, NRMSE        ", dict()),
                      ("bootstrap, NRMSE_partial", dict(partial=True)),
                      ("simulation, NRMSE_partial", dict(partial=True,
                                                         simulate=True))]:
    res = model.ttest(sample_size=50, seed=seed, **kwargs)
    print(f"{label}: nrmse = {res['nrmse']:.4f}, mean_null = "
          f"{res['mean_null']:.4f}, p_two_sided = {res['p_two_sided']:.4g}")
```

Output:

```text
beta hat: [0.8897 0.6921 0.305 ]
C @ beta hat: [0.997  1.5818 1.8868]
x_lower: [0.4614 0.3589 0.1582]
x_upper: [1.318  1.0252 0.4518]
Correlogram:
  constraint    [1 2 3]
  rmsa_i        [0.677  0.677  0.8165]
  rmsa_dkappaC  [-2.6029 -2.6029 -4.0964]
  rmsa_dkappaB  [-2.4188 -1.3432 -3.0651]
  rmsa_dkappaA  [ 0.0015 -0.0168 -0.0282]
  rmsa_dnrmse   [0.     0.0052 0.011 ]
  rmsa_dzhat    [-0.0001 -0.0041 -0.0047]
  rmsa_dz       [-0.0001 -0.0041 -0.0047]
  rmsa_dx       [-0.0001 -0.0041 -0.0047]
Estimator Configuration:
  Generalized inverse:   Moore-Penrose
  Iterations (r):                    1
  Tolerance:                 1.490e-08
  Final correction:               True
  Regularization (α):           1.0000
  Bounds (x):                     none
  Slack sign (S ≥ 0):            False
  Column scaling:                False
  Fixed variables:                   0

Numerical Stability:
  kappaC:                       5.8284
  kappaB:                       4.6823
  kappaA:                       1.5661
  rank:                              3
  rmsa:                         0.7265
  i:              min=    0.6770  max=    0.8165  mean=    0.7235  sd=    0.0805
  dkappaC:        min=   -4.0964  max=   -2.6029  mean=   -3.1007  sd=    0.8622
  dkappaB:        min=   -3.0651  max=   -1.3432  mean=   -2.2757  sd=    0.8698
  dkappaA:        min=   -0.0282  max=    0.0015  mean=   -0.0145  sd=    0.0150
  dnrmse:         min=    0.0000  max=    0.0110  mean=    0.0054  sd=    0.0055
  dzhat:          min=   -0.0047  max=   -0.0001  mean=   -0.0030  sd=    0.0025
  dz:             min=   -0.0047  max=   -0.0001  mean=   -0.0030  sd=    0.0025
  dx:             min=   -0.0047  max=   -0.0001  mean=   -0.0030  sd=    0.0025

Goodness of Fit:
  r2_partial:                   0.7354
  nrmse:                        0.4972
  nrmse_partial:                0.5144
  z_lower:        min=    0.1582  max=    0.4614  mean=    0.3262  sd=    0.1542
  z_upper:        min=    0.4518  max=    1.3180  mean=    0.9317  sd=    0.4406
  x_lower:        min=    0.1582  max=    0.4614  mean=    0.3262  sd=    0.1542
  x_upper:        min=    0.4518  max=    1.3180  mean=    0.9317  sd=    0.4406
  y_lower:        min=       nan  max=       nan  mean=       nan  sd=       nan
  y_upper:        min=       nan  max=       nan  mean=       nan  sd=       nan
bootstrap, NRMSE        : nrmse = 0.4972, mean_null = 0.5015, p_two_sided = 0.5736
bootstrap, NRMSE_partial: nrmse = 0.5144, mean_null = 0.5144, p_two_sided = 0.9953
simulation, NRMSE_partial: nrmse = 0.5144, mean_null = 0.9836, p_two_sided = 3.446e-60
```

The true coefficients satisfy all three restrictions, but the estimate does not satisfy them exactly (`C @ beta hat`), since the constraint rows are least-squares rows. `corr()` adds the RMSA block to the summary (shown with the labels `i`, `dkappaC`, ..., i.e. `rmsa_i`, `rmsa_dkappaC`, ...); the `y_lower`/`y_upper` lines are *nan* because the model has no slack variables. The bootstrap t-tests have null means close to the observed values by construction. The simulated test compares the observed NRMSE_partial with the mean NRMSE_partial obtained when `b`_M is pure N(0, 1) noise (0.98): the observed fit is better than that, and the test says nothing more.

### 6. Choosing α by Cross-Validation: `cv_alpha()`

```python
import   warnings
import   numpy as     np
from     clsp  import CLSP

# AP (TM) 4×4 with known totals and 8 noisy known cells, α chosen by cross-validation

seed     = 7
rng      = np.random.default_rng(seed)

# data: true X with empty cells, its row and column sums, and 8 known cells
m, p     = 4, 4                                              # dimensions of X
X_true   = rng.uniform(0, 3, size=(m, p))
X_true[rng.random((m, p)) < 0.3] = 0.0                       # empty cells
cells    = rng.choice(m * p, size=8, replace=False)          # known cells
M        = np.eye(m * p)[cells]                              # selection rows
draw_b   = lambda: np.concatenate([X_true.sum(axis=1),       # row sums
                                   X_true.sum(axis=0),       # column sums
                                   X_true.ravel()[cells] +   # noisy cells
                                   0.3 * rng.normal(size=8)])

# not informative: without bounds, Step 2 returns zhat for every α (no refits)
model    = CLSP().solve(problem="ap", m=m, p=p, M=M, b=draw_b())
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)          # "not informative"
    res  = model.cv_alpha(folds=4, seed=seed)
print("unbounded:", res["informative"], "-", res["reason"])

# x ≥ 0: a small Monte Carlo study, α chosen by 4-fold CV in each replication
selected, informative, err_cv, err_1 = [], 0, [], []
for rep in range(20):
    model = CLSP().solve(problem="ap", m=m, p=p, M=M, b=draw_b(),
                         bounds=(0, None))                   # α = 1 (default)
    x_1   = model.x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = model.cv_alpha(folds=4, seed=rep, refit=True)  # refit with best α
    if res["informative"] and informative == 0:              # first informative
        print(f"x ≥ 0, replication {rep + 1}:", res["reason"])
        print("  alpha:       ", res["alpha"])
        print("  cv_rmse_mean:", np.round(res["cv_rmse_mean"], 4))
        print("  best:        ", res["best"])
    informative += res["informative"]
    selected.append(model.alpha)
    err_cv.append(np.sqrt(np.mean((model.x - X_true) ** 2)))
    err_1.append(np.sqrt(np.mean((x_1 - X_true) ** 2)))
values, counts = np.unique(selected, return_counts=True)
print("informative replications:", informative, "of 20")
print("selected α:", dict(zip(values.tolist(), counts.tolist())))
print(f"RMSE of X hat: α by CV = {np.mean(err_cv):.4f}, "
      f"α = 1 = {np.mean(err_1):.4f}")
```

Output:

```text
unbounded: False - no bounds or slack signs and a diagonal Z, Step 2 returns zhat for every α
x ≥ 0, replication 2: smallest mean RMSE
  alpha:        [0.0, 0.25, 0.5, 0.75, 1.0]
  cv_rmse_mean: [0.2948 0.2943 0.2943 0.2943 0.2943]
  best:         1.0
informative replications: 18 of 20
selected α: {0.0: 6, 0.25: 1, 1.0: 13}
RMSE of X hat: α by CV = 0.2472, α = 1 = 0.2465
```

Without bounds, `cv_alpha()` makes no refit: the second step returns zhat for every α, so the folds cannot distinguish them, and the result is marked as not informative with `best` = 1. With x ≥ 0, α can change the estimate, but only in the directions left free by the totals, the known cells and the bounds; here, the mean RMSEs differ in most replications, but only slightly, and the α selected by cross-validation does not improve the estimate of X over the default α = 1. In a simulation, `informative` and `reason` should be recorded together with `best`, since a result that is not informative returns α = 1 by construction.

### 7. Subspace Restriction with `Z`

```python
import   numpy as     np
from     clsp  import CLSP

# Linear regression with 4 regressors, the 4th coefficient excluded by a diagonal Z

seed     = 3
rng      = np.random.default_rng(seed)

# sample (dataset)
k        = 30                                                # number of observations
D        = np.column_stack([np.ones(k),                      # constant
                            rng.normal(size=(k, 3))])        # regressors ~ N(0,1)
y        = D @ np.array([1.0, 0.5, -0.5, 0.0]) + 0.2 * rng.normal(size=k)

# model: M = D only (a general problem without C rows)
Z        = np.diag([1.0, 1.0, 1.0, 0.0])                     # Z_44 = 0 excludes x_4
model_I  = CLSP().solve(M=D, b=y)                            # Z = I (Moore-Penrose)
model_Z  = CLSP().solve(M=D, b=y, Z=Z)                       # Bott-Duffin
ols      = np.linalg.lstsq(D[:, :3], y, rcond=None)[0]       # OLS without the 4th column

# results
for label, model in [("Z = I               ", model_I),
                     ("Z = diag(1, 1, 1, 0)", model_Z)]:
    print(f"{label}: x = {np.round(model.x.ravel(), 4)}, "
          f"rank = {model.rank}, residual = {model.residual:.4f}, "
          f"inverse = {model.summary()['inverse']}")
print("OLS on columns 1-3  : x =", np.round(ols, 4))
```

Output:

```text
Z = I               : x = [ 1.0081  0.4884 -0.4712  0.044 ], rank = 4, residual = 1.1449, inverse = Moore-Penrose
Z = diag(1, 1, 1, 0): x = [ 1.0073  0.4878 -0.4778  0.    ], rank = 3, residual = 1.1652, inverse = Bott-Duffin
OLS on columns 1-3  : x = [ 1.0073  0.4878 -0.4778]
```

A diagonal `Z` with `Z`_jj = 0 excludes component j of z from the estimation: here, x_4 is exactly 0, the effective rank drops from 4 to 3, the residual is slightly larger, and the other coefficients equal the ordinary least-squares estimate without the fourth column, i.e., a diagonal 0/1 `Z` acts like deleting columns of `A` while keeping the dimension of z. `summary()` reports the Bott–Duffin inverse for any `Z` ≠ I. A non-diagonal projector restricts the Step 1 estimate to a general subspace (the range of `Z`), but cannot be combined with `bounds`, `slack_sign` or `scale`. Without bounds and with a diagonal `Z`, Step 2 returns zhat for every α (see [Step 2](#step-2-convex-correction)).

### 8. Residual-Sign Refinement (`r` > 1)

```python
import   numpy as     np
from     clsp  import CLSP

# General problem: the constraint row x_1 + x_2 = 1 conflicts with the model rows x_1 = 1, x_2 = 2

C        = np.array([[1.0, 1.0]])                            # x_1 + x_2 = 1
M        = np.eye(2)                                         # x_1 = 1, x_2 = 2
b        = np.array([1.0, 1.0, 2.0])
for r in (1, 3, 10):
    model = CLSP().solve(C=C, M=M, b=b, r=r)
    print(f"r = {r:<2}: iterations = {model.r}, x = {np.round(model.x.ravel(), 4)}, "
          f"y_res = {np.round(model.y.ravel(), 4)}, C @ x = {(C @ model.x).item():.4f}")
    print(f"        nrmse = {model.nrmse:.4f}, nrmse_partial = "
          f"{model.nrmse_partial:.4f}, step1 = {model.status['step1']}")

# the signs of the residual slacks do not affect x: flip Q and solve by pinv
A_flip   = model.A.copy()
A_flip[1:, 2:] *= -1                                         # Q -> -Q
print("x with -Q:", np.round((np.linalg.pinv(A_flip) @ b)[:2], 4))
```

Output:

```text
r = 1 : iterations = 1, x = [0.3333 1.3333], y_res = [], C @ x = 1.6667
        nrmse = 1.4142, nrmse_partial = 1.3333, step1 = converged
r = 3 : iterations = 3, x = [0.25 0.75], y_res = [-0.75 -1.25], C @ x = 1.0000
        nrmse = 0.0000, nrmse_partial = 2.0616, step1 = converged
r = 10: iterations = 3, x = [0.25 0.75], y_res = [-0.75 -1.25], C @ x = 1.0000
        nrmse = 0.0000, nrmse_partial = 2.0616, step1 = converged
x with -Q: [0.25 0.75]
```

With `r` = 1, all three rows are fitted in least squares, and the constraint row is not met (`C @ x` = 1.6667). With `r` > 1, the residual slacks `Q` absorb the misfit of the model rows: the constraint row is met exactly, x = (0.25, 0.75) minimizes ‖x‖² + ‖`b`_M − `M`x‖² subject to x_1 + x_2 = 1, `nrmse` (of the augmented system) is 0.0000, and `nrmse_partial` of the model rows increases from 1.3333 to 2.0616. The iteration stops after 3 iterations, when the NRMSE no longer changes, also for `r` = 10 (`self.r`). Flipping the signs of `Q` leaves x unchanged, since y_res is unrestricted. The refinement is thus not a better fit of the model rows but a different estimator, in which x is a Ridge-type compromise (see [Residual-Sign Refinement](#residual-sign-refinement-r--1)).

### 9. Truncation of Small Singular Values: `rcond`

```python
import   numpy as     np
from     clsp  import CLSP

# General problem with two nearly collinear constraint rows (singular values ≈ 2 and 5e-7)

C        = np.array([[1.0, 1.0       ],
                     [1.0, 1.0 + 1e-6]])
b        = np.array([2.0, 2.001])
for label, rcond in [("default", None),                      # cutoff sqrt(max(A.shape)·eps)
                     ("1e-4   ", 1e-4)]:                     # cutoff 1e-4·σ_max
    model = CLSP().solve(C=C, b=b, rcond=rcond)
    print(f"rcond = {label}: rank = {model.rank}, sv_discarded = "
          f"{model.sv_discarded}, x = {np.round(model.x.ravel(), 4)}")
    print(f"  rho = {model.rho:.6f}, residual = {model.residual:.6f}, "
          f"kappaA = {model.kappaA:.4g}")
```

Output:

```text
rcond = default: rank = 2, sv_discarded = [], x = [-998. 1000.]
  rho = 0.000000, residual = 0.000000, kappaA = 4e+06
rcond = 1e-4   : rank = 1, sv_discarded = [4.99999875e-07], x = [1.0002 1.0003]
  rho = 0.000706, residual = 0.000706, kappaA = 4e+06
```

With the default cutoff (sqrt(max(`A`.shape)·eps) ≈ 2.1·10⁻⁸ relative to σ_max ≈ 2), the small singular value 5·10⁻⁷ is kept, and the exact solution x = (−998, 1000) of the nearly singular system is returned (ρ = 0); it changes by about 1000 per 0.001 change of `b`_2. With `rcond=1e-4`, the singular value is discarded (`rank` = 1, `sv_discarded`), and x ≈ (1, 1) is the minimum-norm solution of the truncated system `A`_k z = `b`, with ρ = 0.000706 > 0, although `A`z = `b` has an exact solution. Here, `residual` equals `rho`, because z lies in the retained row space; both refer to the cutoff in use, so ρ is "the smallest residual" only within the numerical range of `A` defined by `rcond`. `kappaA` is computed from `A` itself with `cond_tolerance` and is not affected by `rcond`.

### 10. AP (TM) with Grouped Totals and a Zero Diagonal

```python
import   numpy as     np
from     clsp  import CLSP

# AP (TM) 4×4 with grouped totals: sums of row pairs (i = 2) and column pairs (j = 2)

b        = np.array([10, 20,                                 # sums of rows 1-2 and 3-4
                     12, 20])                                # sums of columns 1-2 and 3-4
model    = CLSP().solve(problem="ap", m=4, p=4, i=2, j=2, b=b)
print("C rows x columns:", model.C_idx)
print("X hat (grouped totals, unbounded):")
print(np.round(model.x, 4))
print("row-pair sums:   ", np.round(model.x.reshape(2, 8).sum(axis=1), 4))
print("column-pair sums:", np.round(model.x.reshape(4, 2, 2).sum(axis=(0, 2)), 4))

# AP (TM) 3×3 with x ≥ 0: a zero diagonal as soft model rows vs. exact zeros via bounds
b        = np.array([10, 1, 1,                               # row sums
                     10, 1, 1])                              # column sums
model_s  = CLSP().solve(problem="ap", m=3, p=3, b=b, bounds=(0, None),
                        zero_diagonal=True)                  # soft x_kk = 0
lohi     = [(0, 0) if r == c else (0, None)                  # exact x_kk = 0
            for r in range(3) for c in range(3)]
model_h  = CLSP().solve(problem="ap", m=3, p=3, b=b, bounds=lohi)
for label, model in [("zero_diagonal=True", model_s),
                     ("bounds lo = hi = 0", model_h)]:
    print(f"{label}:")
    print(np.round(model.x, 4))
    print("  row sums:", np.round(model.x.sum(axis=1), 4),
          "column sums:", np.round(model.x.sum(axis=0), 4))
```

Output:

```text
C rows x columns: (4, 16)
X hat (grouped totals, unbounded):
[[0.8125 0.8125 1.8125 1.8125]
 [0.8125 0.8125 1.8125 1.8125]
 [2.0625 2.0625 3.0625 3.0625]
 [2.0625 2.0625 3.0625 3.0625]]
row-pair sums:    [10.5 20.5]
column-pair sums: [11.5 19.5]
zero_diagonal=True:
[[3.2 2.6 2.6]
 [2.6 0.  0. ]
 [2.6 0.  0. ]]
  row sums: [8.4 2.6 2.6] column sums: [8.4 2.6 2.6]
bounds lo = hi = 0:
[[0.     3.6667 3.6667]
 [3.6667 0.     0.    ]
 [3.6667 0.     0.    ]]
  row sums: [7.3333 3.6667 3.6667] column sums: [7.3333 3.6667 3.6667]
```

With `i` = `j` = 2, `C` consists of two row-pair sums and two column-pair sums (4 rows for 16 cells), and `b` starts with these four totals. The totals are inconsistent (30 vs. 32), so they are fitted in least squares (10.5, 20.5 and 11.5, 19.5) like all rows of `A`. In the 3 × 3 problem, non-negative cells cannot meet the totals with a zero diagonal (row 1 must sum to 10 without x_11, but its other cells lie in columns 2 and 3, which sum to 1 each). `zero_diagonal=True` adds the rows x_kk = 0 as model rows (with zeros appended to `b` automatically), which are traded off against the totals: x_11 = 3.2. Exact zeros require bounds with lo = hi = 0 on the diagonal cells; the misfit is then carried by the totals alone.

### 11. Reading the Results

```python
import   numpy as     np
from     clsp  import CLSP

# CMLS (RP) with non-negative coefficients and a (soft) sum of 1, followed by the results

seed     = 11
rng      = np.random.default_rng(seed)

# sample (dataset)
k        = 25                                                # number of observations
D        = rng.normal(size=(k, 3))                           # regressors ~ N(0,1)
y        = D @ np.array([0.7, 0.4, -0.1]) + 0.1 * rng.normal(size=k)

# model
model    = CLSP().solve(problem="cmls", C=np.ones((1, 3)), M=D,
                        b=np.concatenate([[1.0], y]), bounds=(0, None))

# results
print("x     :", np.round(model.x.ravel(),     4))           # final estimate (x part of z)
print("zhat  :", np.round(model.zhat.ravel(),  4))           # Step 1 (bounded)
print("zhat0 :", np.round(model.zhat0.ravel(), 4))           # Step 1 (unbounded), the anchor
print("fixed :", np.flatnonzero(model.fixed),                # components of z at a bound
      "at pass", model.fixed_at[model.fixed])
print(f"rho = {model.rho:.4f}, residual = {model.residual:.4f}, "
      f"rank = {model.rank} of {model.A.shape[1]}")
for key in ("step1", "rho", "final", "solver", "retry", "polish",
            "tol_v", "kkt_error", "bound_error", "eq_error"):
    val   = model.status[key]
    print(f"  {key:<12}", f"{val:.1e}" if isinstance(val, float) else val)
```

Output:

```text
x     : [0.6641 0.3575 0.    ]
zhat  : [0.6641 0.3575 0.    ]
zhat0 : [ 0.6755  0.3609 -0.0653]
fixed : [2] at pass [1]
rho = 0.5537, residual = 0.5537, rank = 3 of 3
  step1        converged
  rho          bvls
  final        optimal
  solver       OSQP
  retry        None
  polish       True
  tol_v        4.1e-04
  kkt_error    6.3e-15
  bound_error  0.0e+00
  eq_error     6.2e-16
```

| Attribute | Meaning |
|---|---|
| `x` | final estimate of the unknowns, the first n components of `z` (an `m` × `p` matrix for AP (TM) problems); the slacks are in `y` |
| `zhat` | Step 1 estimate; with `bounds` or `slack_sign`, a bounded least-squares minimizer on `A`_k (here equal to `x`: with full column rank, the bounded least-squares minimizer is unique, so Step 2 has no other solution to choose) |
| `zhat0` | unbounded Step 1 estimate `A`_k⁺`b`, the anchor of Step 2 (here, it violates the bound: −0.0653) |
| `fixed`, `fixed_at` | components of z at a bound at the termination of Step 1, and the pass at which they were fixed (0 = free) |
| `rho` | ‖`b` − w*‖₂, the smallest residual within the numerical range of `A` under the bounds (*None* if `final=False`) |
| `residual` | ‖`b` − `A`z‖₂ of the returned z with the original `A` (equal to `rho` unless singular values were discarded or Step 2 fell back to zhat) |
| `rank` | effective rank of `A`·`Z` (the rank of `A`_k), out of the number of columns of `A` |
| `status["step1"]` | *'converged'*, *'bvls fallback'* (scipy's BVLS solved the bounded problem) or *'not converged: <reason>'* (the last feasible point was kept) |
| `status["rho"]` | source of w*: *'pinv'* (no bounds), *'bvls'* (bounds) or *'skipped'* (`final=False`) |
| `status["final"]` | CVXPY status of Step 2 (*'optimal'* or *'optimal_inaccurate'*; *'optimal'* for a feasible anchor), *'fallback: <reason>'* (z = zhat) or *'skipped'* |
| `status["solver"]`, `status["retry"]` | Step 2 solver actually used (*'none (anchor feasible)'* if the feasible anchor was returned without a solver), and the reason of a retry with CLARABEL (*None* without a retry) |
| `status["polish"]` | whether the α = 1 solution was polished on its active set, i.e. is the exact Step 2 solution up to rounding (*False* if not applied or rejected, *None* if Step 2 is skipped) |
| `status["tol_v"]` | verification tolerance sqrt(`tolerance`)·max(1, ‖`b`‖₂) of Step 2 |
| `status["kkt_error"]` | largest KKT violation of zhat on `A`_k (at most tol_g on convergence) |
| `status["scale"]` | whether the columns were scaled (`scale=True`; `kkt_error` and `eq_error` are then computed in u = D⁻¹z) |
| `status["bound_error"]`, `status["eq_error"]` | largest bound violation of the returned z, and ‖`A`_k z − w*‖₂ |

Values at the level of rounding errors (`kkt_error`, `eq_error`) differ across platforms. A fit is usable as intended if `status["step1"]` is *'converged'* or *'bvls fallback'* and `status["final"]` does not start with *'fallback'*; this is the check applied to the refits of `corr()`, `ttest(simulate=True)` and `cv_alpha()`.

## User Reference

For comprehensive information on the estimator's capabilities, advanced configuration options, and implementation details, please refer to the docstrings provided in each of the individual .py source files. These docstrings contain complete descriptions of available methods, their parameters, expected input formats, and output structures.

To ensure cross-platform reproducibility, all CLSP implementations use a modified condition number function based on singular values, with a relative cutoff equal to `cond_tolerance * the largest singular value`.

### The `CLSP` Class

```python
self.__init__()
```

Stores the solution, goodness-of-fit statistics, and ancillary parameters.

The class has four core methods: `solve()`, `corr()`, `ttest()` and `cv_alpha()`, and a `summary()` (alias `summarize()`) method. The settings of `solve()` (`final`, `rcond`, `tolerance`, `iteration_limit`, `Z`, `bounds`, `slack_sign`, `scale`) are stored in the object and kept by later calls unless given again; `alpha` is not kept (*None* means 1). The fit specification (`problem`, `C`, `S`, `M`, `m`, `p`, `i`, `j`, `zero_diagonal`) is stored as well: a later `solve()` without `C`, `M` or (`m`, `p`) starts from the stored `A` without `Q` and applies the requested `r`, so that it equals a fresh fit with the same inputs.

**Selected attributes:**
`self.A`             : *np.ndarray*<br>
design matrix `A` = [`C` | `S` | 0; `M` | 0 | `Q`], where `Q` is either empty (`r` = 1) or diag(-sign(e_M)) (`r` > 1).

`self.A_base`        : *np.ndarray*<br>
design matrix of the fit specification without `Q`, from which every `solve()` starts (the matrix built by `canonize()`, also used by the refits of `corr()`, `ttest()` and `cv_alpha()`).

`self.spec`          : *dict*<br>
inputs of the fit specification: `problem`, `C`, `S`, `M`, `m`, `p`, `i`, `j`, `zero_diagonal`.

`self.C_idx`         : *tuple*<br>
shape (rows, columns) of the `C` block in `A`; the number of columns equals the length of x.

`self.S_idx`         : *tuple*<br>
column range (first, last + 1) of the `S` block (slack variables) in `A`.

`self.b`             : *np.ndarray*<br>
vector of the right-hand side.

`self.Z`             : *np.ndarray*<br>
projector of the Bott–Duffin inverse (the identity for the Moore–Penrose inverse).

`self.rcond`         : *float* or *bool*<br>
relative cutoff on the singular values of `A`·`Z` that defines `A`_k (initially *False*, see `solve()`).

`self.rank`          : *int*<br>
effective rank of `A`·`Z` (of `A`·`Z`·D if `scale` is *True*), i.e. the number of singular values above the cutoff (the rank of `A`_k).

`self.sv_discarded`  : *np.ndarray*<br>
singular values of `A`·`Z` (of `A`·`Z`·D if `scale` is *True*) at or below the cutoff, treated as zero (empty if none).

`self.bounds`        : *tuple*, *list* or *None*<br>
bounds on `x` as supplied: *None* (no bounds), a single (lo, hi) pair, or a sequence of (lo, hi) pairs (*None* = unbounded side).

`self.x_lb`          : *np.ndarray*<br>
normalized lower bounds on `x` (-inf if unbounded).

`self.x_ub`          : *np.ndarray*<br>
normalized upper bounds on `x` (inf if unbounded).

`self.slack_sign`    : *bool*<br>
whether the slack variables of the `S` block are restricted to be ≥ 0 (S_ii = +1 encodes ≤, S_ii = -1 encodes ≥).

`self.scale`         : *bool*<br>
whether the columns of `A`·`Z` are equilibrated in both steps (initially *False*, see `solve()`).

`self.zhat`          : *np.ndarray*<br>
vector of the first-step estimate.

`self.zhat0`         : *np.ndarray*<br>
vector of the unbounded first-step estimate `A`_k⁺`b` (for the final `Q` if `r` > 1), at which the second step is anchored. It equals `self.zhat` without bounds.

`self.r`             : *int*<br>
number of refinement iterations performed in the first step.

`self.r_max`         : *int*<br>
requested number of refinement iterations `r`, reused by the diagnostic refits in `corr()` and `ttest(simulate=True)`.

`self.fixed`         : *np.ndarray*<br>
boolean mask over `z` of the components at a bound at the termination of the first step (all *False* without bounds).

`self.fixed_at`      : *np.ndarray*<br>
first-step pass at which each component of `z` was fixed (0 = free; a component released by the KKT check gets 0 again).

`self.final`         : *bool*<br>
whether the second step is solved (initially *True*).

`self.alpha`         : *float*<br>
regularization parameter α of the second step (supplied, 1 by default, or the best α of `cv_alpha(refit=True)`).

`self.z`             : *np.ndarray*<br>
vector of the final solution. If the second step is disabled or falls back, it equals `self.zhat`.

`self.rho`           : *float* or *None*<br>
smallest residual norm ρ = ‖`b` - w*‖₂ within the numerical range of `A` (singular values above the cutoff) under the bounds, where w* = `A`_k`zhat` is the fitted vector of the (bounded) least-squares problem of the first step, the target of the second step (not necessarily the residual of the returned `z`). If the second step is disabled, it equals *None*.

`self.residual`      : *float*<br>
residual norm ‖`b` - `A``z`‖₂ of the returned `z` with the original `A`. It equals `self.rho` (up to solver accuracy) unless singular values were discarded or the second step fell back to `zhat`.

`self.status`        : *dict*<br>
status of the last solve: {`"step1"`: outcome of the bounded first step (*'converged'*, *'bvls fallback'* or *'not converged: <reason>'*; *'converged'* without bounds), `"rho"`: source of w*/ρ (*'pinv'* without bounds, *'bvls'* with bounds, or *'skipped'*), `"final"`: CVXPY status of the second step, *'fallback: <reason>'* or *'skipped'*, `"solver"`: name of the second-step solver actually used (*'none (anchor feasible)'* if the feasible anchor `zhat0` was returned without a solver, *None* if skipped), `"retry"`: reason of a retry with CLARABEL after a failed or inaccurate OSQP solution (*None* without a retry), `"polish"`: whether the α = 1 solution was polished on its active set (*False* if not applied or rejected, *None* if skipped), `"tol_v"`: verification tolerance sqrt(`tolerance`)·max(1, ‖`b`‖₂) (*None* if skipped), `"kkt_error"`: largest KKT violation of the first-step estimate on `A`_k (max|g_j| of the free variables, sign violations at the bounds; in the scaled variables u if `scale` is *True*), `"scale"`: whether the columns were scaled, `"bound_error"`: largest bound violation of the returned `z`, `"eq_error"`: ‖`A`_k`z` - w*‖₂ of the returned `z` (with u = D⁻¹`z` in place of `z` if scaled; *None* if skipped)}.

`self.x`             : *np.ndarray*<br>
`m` x `p` matrix or vector containing the variable component of `z`.

`self.y`             : *np.ndarray*<br>
vector containing the slack component of `z` (the `S` slacks followed by the residual slacks of `Q`, if any).

`self.kappaC`        : *float*<br>
spectral κ() for *C_canon*.

`self.kappaB`        : *float*<br>
spectral κ() for *B* = *`A`C_canon^+*.

`self.kappaA`        : *float*<br>
spectral κ() for `A`.

`self.rmsa`          : *float*<br>
total root mean square alignment (RMSA), computed by `corr()` (*None* before).

`self.rmsa_i`, `self.rmsa_dkappaC`, `self.rmsa_dkappaB`, `self.rmsa_dkappaA`, `self.rmsa_dnrmse`, `self.rmsa_dzhat`, `self.rmsa_dz`, `self.rmsa_dx` : *list[float]*<br>
row-wise results of `corr()` (empty before).

`self.rmsa_failures` : *int*<br>
number of refits of `corr()` whose first step did not converge or whose second step fell back to `zhat` (0 before).

`self.r2_partial`    : *float*<br>
R^2 for the `M` block in `A` (*nan* without `M` rows).

`self.nrmse` : *float*<br>
root mean square error calculated from `A` and normalized by standard deviation (NRMSE).

`self.nrmse_partial` : *float*<br>
root mean square error calculated from the `M` block in `A` and normalized by standard deviation (NRMSE; *nan* without `M` rows).

`self.nrmse_ttest`   : *list[float]*<br>
NRMSE sample under H₀ generated by `ttest()`.

`self.nrmse_ttest_key` : *tuple* or *None*<br>
key (`sample_size`, `seed`, `distribution`, `partial`, `simulate`) of `self.nrmse_ttest`; the sample is reused only if the key matches.

`self.nrmse_ttest_failures` : *int*<br>
number of failed refits of `ttest(simulate=True)` for `self.nrmse_ttest` (0 for the bootstrap).

`self.cv_alpha_result` : *dict* or *None*<br>
result of the last `cv_alpha()` (*None* before and after every `solve()`).

`self.z_lower`       : *np.ndarray*<br>
lower bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `z` - |`z`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.z_upper`       : *np.ndarray*<br>
upper bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `z` + |`z`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.x_lower`       : *np.ndarray*<br>
lower bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `x` - |`x`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.x_upper`       : *np.ndarray*<br>
upper bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `x` + |`x`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.y_lower`       : *np.ndarray*<br>
lower bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `y` - |`y`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.y_upper`       : *np.ndarray*<br>
upper bound of the heuristic condition-weighted sensitivity band based on κ(`A`), `y` + |`y`|·κ(`A`)·‖`b` - `A``z`‖₂ / ‖`b`‖₂; no statistical coverage is implied.

`self.seed`          : *int*<br>
random seed of the Monte Carlo and bootstrap diagnostics (default *123456789*).

### Solver Method: `solve()`

```python
self.solve(problem, C, S, M, b, m, p, i, j, zero_diagonal, r, Z, rcond, tolerance, iteration_limit, final, alpha, cond_tolerance, bounds, slack_sign, scale)
```

Solves the Convex Least Squares Programming (CLSP) problem. All arguments are passed by keyword.

This method performs a two-step estimation:<br>
(1) a pseudoinverse-based solution using either the Moore–Penrose or Bott–Duffin inverse, optionally iterated for refinement and, if `bounds` are set, computed by bounded-variable least squares;<br>
(2) a convex-programming correction using Lasso, Ridge, or Elastic Net regularization (if enabled).

Please note that the rows of `C` are least-squares rows (soft), like the rows of `M`: in an inconsistent system, the fit of the `C` rows is traded off against the fit of the `M` rows. Only `bounds` and the sign of the `S` slacks (`slack_sign`) are enforced exactly; a signed slack does not make its row a hard equality or inequality. A hard-constraint mode (`C`x + `S`y = `b`_C imposed exactly) is not implemented. See [How CLSP Works](#how-clsp-works) for the mathematical details.

**Parameters:**
`problem` : *str*, optional<br>
    Structural template for matrix construction. One of:<br>
    - *'ap'* or *'tm'* : allocation (tabular) matrix problem (AP).<br>
    - *'cmls'* or *'rp'* : constrained-model least squares (regression) problem; requires both `C` and `M`.<br>
    - anything else: general CLSP problem (user-defined `C` and/or `M`).

`C`, `S`, `M` : *np.ndarray* or *None*<br>
    Blocks of the design matrix `A` = [`C` | `S`; `M` | `Q`]. If `C` and/or `M` are provided, the matrix `A` is constructed accordingly (please note that for AP, `C` is constructed automatically and known values are specified in `M`; a supplied `C` with `m`·`p` columns is appended below the row and column sums).

`b` : *np.ndarray* or *None*<br>
    Right-hand side vector. Must have as many rows as `A` (please note that for AP, it should start with row sums, followed by column sums, the rows of a supplied `C`, and the rows of `M`). Required. If neither `C`, `M` nor (`m`, `p`) are provided, the stored specification (`A` without `Q`, `m`, `p`) is reused and a supplied `b` replaces the stored `b`.

`m`, `p` : *int* or *None*<br>
    Dimensions of X ∈ ℝ^{m×p}, relevant for AP. `x` is the row-major vectorization of X and is returned as an `m` x `p` matrix.

`i`, `j` : *int*, default = *1*<br>
    Grouping sizes for row and column sum constraints in AP (each row-sum row covers `i` consecutive rows of X, each column-sum row covers `j` consecutive columns; `m` and `p` must be divisible by `i` and `j`).

`zero_diagonal` : *bool*, default = *False*<br>
    If *True*, adds model rows x_kk = 0 for the diagonal of X in AP (least-squares rows, appended to `M` and `b`_M in this order; no row is merged or removed, so a known diagonal cell in `M` and its row x_kk = 0 are both fitted, e.g. x₁₁ = 3 and x₁₁ = 0 are traded off like any two observations). For an exact zero, use `bounds` with lo = hi = 0 for these cells.

`r` : *int*, default = *1*<br>
    Number of refinement iterations for the pseudoinverse-based estimator (≥ 1; used only if `M` rows are present). When `r` > 1, the slack block `Q` is set to diag(-sign(e_M)), where e_M = `b`_M - `M`x is the model-row residual, whose entries with |e_i| ≤ `tolerance` · max(1, ‖`b`‖₂) are treated as exact zeros, so that rounding noise does not trigger the iteration. The signs of the `Q` slacks do not affect the pseudoinverse estimate of `x`; the iteration acts only when the model rows (`M`) are not fitted exactly, e.g., when they conflict with the constraints (`C`), and then moves the model-row misfit into the residual slacks (see [Residual-Sign Refinement](#residual-sign-refinement-r--1)).

`Z` : *np.ndarray* or *None*<br>
    A symmetric idempotent matrix (projector) defining the subspace for Bott–Duffin pseudoinversion. If *None*, the stored matrix is kept (initially the identity matrix, reducing the Bott–Duffin inverse to the Moore–Penrose case). With `bounds` or `slack_sign`, `Z` must be diagonal (0/1).

`rcond` : *float*, *bool* or *None*, default = *None*<br>
    Relative cutoff on the singular values of `A`·`Z`. `A`_k is `A`·`Z` with the singular values σ ≤ `rcond`·σ_max set to zero; it is used in the first step (`zhat0` = `A`_k⁺`b`, equivalent to the Bott–Duffin form (`Z`·`A`ᵀ`A`·`Z`)⁺`Z`·`A`ᵀ`b` without squaring the condition number of `A`·`Z`, and the bounded least-squares problem), for w* and ρ, and in the constraint of the second step (see `self.rank` and `self.sv_discarded`).<br>
    If *False*, the cutoff sqrt(max(`A`.shape) · eps) is applied, which reproduces the effective cutoff of the normal-equations form and is robust to the structural rank deficiency of AP (TM) problems. If True, a cutoff equal to `tolerance` is applied. If set to a float, it specifies the relative cutoff below which small singular values are treated as zero. If *None*, the stored value is kept (initially *False*).

`tolerance` : *float*, default = *square root of machine epsilon*<br>
    Convergence tolerance for NRMSE change between refinement iterations (the iteration also stops if NRMSE is *inf* in two successive iterations, i.e. sd(`b`) = 0). It also scales the rounding threshold for e_M, the bound and KKT tolerances of the bounded first step, the tie threshold of `cv_alpha()`, and the verification tolerance of the second step, tol_v = sqrt(`tolerance`) · max(1, ‖`b`‖₂). If *None*, the stored value is kept.

`iteration_limit` : *int*, default = *50*<br>
    Limit of the refinement loop (`r` > 1), which stops as soon as the number of iterations exceeds `iteration_limit`, i.e., after at most `iteration_limit` + 1 iterations; with `bounds` or `slack_sign`, also the maximum number of passes of the bounded first step (in each refinement iteration), after which scipy's BVLS is used. If *None*, the stored value is kept.

`final` : *bool* or *None*, default = *None*<br>
    If *True*, a convex programming problem is solved to refine `zhat`. The resulting solution `z` minimizes a weighted L1/L2 norm around the unbounded first-step estimate `zhat0` = `A`_k⁺`b` (= `zhat` without bounds) subject to `A`_k`z` = w* and to `bounds` and `slack_sign` (if set), where w* = `A`_k`zhat` is the fitted vector of the (bounded) least-squares problem of the first step, i.e., ρ = ‖`b` - w*‖₂ is the smallest residual within the numerical range of `A` under the bounds. This coincides with `Az` = `b` when the system is consistent and unbounded. For `Z` = I and α = 1, `z` is the minimum-norm solution among all (bounded) least-squares solutions, since `z`ᵀ`zhat0` is constant on {`z` : `A`_k`z` = w*}.<br>
    If the anchor `zhat0` already satisfies these constraints within `tolerance` · max(1, ‖`b`‖₂), it is returned without a solver (`self.status["final"]` = *'optimal'*, `self.status["solver"]` = *'none (anchor feasible)'*), since the objective is 0 there. Otherwise, the solution is verified with tol_v = sqrt(`tolerance`) · max(1, ‖`b`‖₂): the bound violation of the solver output must be ≤ tol_v and, after clipping to the bounds, ‖`A`_k`z` - w*‖₂ ≤ tol_v. If the solver fails, its status is neither *optimal* nor *optimal_inaccurate*, or the solution fails verification, `z` = `zhat` is returned with a *RuntimeWarning* stating the reason; with the default solver for α = 1 (OSQP), a failed, inaccurate or unverified solution is first retried with CLARABEL (`self.status["solver"]`, `self.status["retry"]`); if the retry fails, a verified *optimal_inaccurate* OSQP solution is kept. A successful retry issues no warning; it is recorded in `self.status["retry"]` and printed by `summary()`. CVXPY's *UserWarning* "Solution may be inaccurate" is suppressed (the solution is verified). tol_v and the achieved errors of the returned `z` are reported in `self.status` (`"tol_v"`, `"bound_error"`, `"eq_error"`); the verification establishes approximate feasibility, not optimality.<br>
    For α = 1, a verified solver solution is polished: the components within tol_v (then within `tolerance` · max(1, ‖`b`‖₂)) of a finite bound are fixed, and the projection of `zhat0` on the others is solved exactly by a pseudoinverse in the basis U_k of the range of `A`_k. The polished point is kept if its bound violation, ‖`A`_k`z` - w*‖₂ and the sign violations of its bound multipliers (KKT) are ≤ `tolerance` · max(1, ‖`b`‖₂); otherwise the solver solution is kept (`self.status["polish"]`).<br>
    If *None*, the stored value is kept (initially *True*).

`alpha` : *float* or *None*, default = *None*<br>
    Regularization parameter (weight) in the final convex program, a single value in [0, 1]:<br>
    - `α = 0`: Lasso (L1 norm)<br>
    - `α = 1`: Tikhonov Regularization/Ridge (L2 norm)<br>
    - `0 < α < 1`: Elastic Net<br>
    If *None*, α = 1: the solution is then unique and, for `Z` = I, the minimum-norm (bounded) least-squares solution. A *list*, *tuple* or *np.ndarray* raises a *CLSPError* (the selection by NRMSE was removed in 5.0.0, since all α attain the same residual; use `cv_alpha()`), as do NaN, inf, non-numeric and out-of-range values.<br>
    The endpoints are exact (no rounding to 0 or 1). The final convex program is solved with ECOS (α = 0), OSQP (α = 1; `eps_abs` = `eps_rel` = 1e-10, `max_iter` = 100000 unless given), or CLARABEL (0 < α < 1), unless the anchor is feasible (see `final`); another CVXPY solver can be selected via `solver=` in `**kwargs`. `alpha` is not used if `final` is *False*.

`cond_tolerance` : *float* or *None*, default = *None*<br>
    Singular-value cutoff for the custom condition number function.<br>
    If *None*, the implementation uses an internal relative cutoff of `1e-14`.

`bounds` : *tuple*, *list* or *None*, default = *None*<br>
    Bounds on the variables `x`, enforced in the first step by bounded-variable least squares on `A`_k (Lawson–Hanson, Stark–Parker): from a feasible start, least-squares steps on the free columns are taken up to the first blocking bound, and the fixed variable with the largest KKT violation (g_j < -tol_g at a lower bound, g_j > tol_g at an upper bound, g = `A`_kᵀ(`A`_k`z` - `b`), tol_g = `tolerance` · max(1, ‖`A`_kᵀ`b`‖_∞)) is released until none is left, and the result is verified by the full KKT conditions (|g_j| ≤ tol_g for the free variables, `self.status["kkt_error"]`; see [Bounds and Signed Slacks](#bounds-and-signed-slacks-bounded-variable-least-squares)). A cycle, `iteration_limit` passes or a failed verification switch to `scipy.optimize.lsq_linear(..., method="bvls")` with a *RuntimeWarning*; if that fails too, the last feasible point is kept. The outcome is reported in `self.status["step1"]`. The bounds are imposed as constraints in the second step. One of:<br>
    - *None* : no bounds (a previously set value is kept).<br>
    - *(lo, hi)* : the same bounds for all `x`.<br>
    - *[(lo, hi), ...]* : one pair per `x` variable (or a single pair).<br>
    *None* (or ±*np.inf*) inside a pair denotes an unbounded side, and (*None*, *None*) gives the same results as no bounds; lo ≤ hi is required. Bounds require a diagonal (0/1) `Z`. The coordinates excluded by `Z` (`Z`_jj = 0) are 0 in the first step, so their bounds must admit 0 (lo ≤ 0 ≤ hi), otherwise a *CLSPError* is raised. Since *None* keeps the stored bounds, previously set bounds are removed by passing (*None*, *None*) or by using a new `CLSP` object.

`slack_sign` : *bool* or *None*, default = *None*<br>
    If *True*, the slack variables of the `S` block must be ≥ 0, i.e., S_ii = +1 encodes ≤ and S_ii = -1 encodes ≥ (enforced in the same way as `bounds`, requiring a diagonal (0/1) `Z`). The sign of the slack does not make the row a hard inequality or equality: the constraint rows remain least-squares rows, fitted exactly on their feasible side and in least squares on their infeasible side (see Example 3). The residual slacks (`Q`) are never constrained. If *None*, a previously set value is kept (initially *False*).

`scale` : *bool* or *None*, default = *None*<br>
    If *True*, the columns of `A`·`Z` are equilibrated: both steps are solved for `A`·`Z`·D with D = diag(1/‖(`A`·`Z`)_j‖₂) (1 for zero columns) in the variables u = D⁻¹`z` (with the bounds divided accordingly), and `z` = D·u is returned. `rcond`, `rank` and `sv_discarded` then refer to the singular values of `A`·`Z`·D, and the minimum-norm selection (`zhat0`, Step 2 with α = 1) and the Step 2 norms are taken in u, i.e., in ‖D⁻¹`z`‖₂, so that the estimate of an underdetermined problem can differ from the unscaled one. Requires a diagonal (0/1) `Z`. If *None*, the stored value is kept (initially *False*); the diagnostic refits use the stored value. See [Limitations](#limitations).

`*args`, `**kwargs` : optional<br>
    CVXPY arguments passed to the CVXPY solver of the second step (e.g., `solver=`, solver tolerances).

**Returns:**
*self*

**Errors and warnings:**<br>
A *CLSPError* (`clsp.errors.CLSPError`) is raised for malformed or inconsistent inputs: a missing `b`; neither `C`, `M` nor (`m`, `p`); missing `m`/`p` or indivisible `i`/`j` for AP; missing `C` or `M` for CMLS; non-conformable blocks; non-finite (NaN or inf) entries in `C`, `S`, `M`, `b` or `Z`; `r` < 1; a `Z` that is not symmetric, idempotent and conformable; invalid `bounds`; `bounds`/`slack_sign` or `scale` with a non-diagonal `Z`; bounds excluding 0 on a coordinate excluded by `Z`; invalid `alpha` (including a list, tuple or array of candidates); a failed pseudoinverse (non-convergent SVD or non-finite `zhat`).<br>
A *RuntimeWarning* is issued if the bounded first step does not converge (cycle, `iteration_limit` or failed KKT verification; scipy's BVLS is then used) or if the second step falls back to `zhat`.

### Correlogram Method: `corr()`

```python
self.corr(reset, threshold)
```

Computes the structural correlogram of the CLSP constraint part.

This method performs a row-deletion sensitivity analysis on the canonical constraint matrix `[C` | `S`], denoted as *C_canon*, and evaluates the marginal effect of each constraint row on numerical stability, angular alignment, and estimator sensitivity.

For each row `i` in `C_canon`, it computes:<br>
    - The Root Mean Square Alignment (`RMSA_i`) with all other rows `j` ≠ `i`.<br>
    - The change in condition numbers κ(`C`), κ(`B`), and κ(`A`) when row `i` is deleted.<br>
    - The effect on estimation quality: changes in `nrmse`, `zhat`, `z`, and `x` when row `i` is deleted.

Additionally, it computes the total `rmsa` statistic across all rows, summarizing the overall angular alignment of *C_canon*.

Each deletion is followed by a refit with the fitted settings (`final`, `alpha`, `r`, `rcond`, `Z`, `tolerance`, `iteration_limit`, `bounds`, `slack_sign`, `scale`), i.e., `k` refits for `k` constraint rows. Warnings raised by the individual refits are suppressed; instead, the refits whose first step did not converge (neither *'converged'* nor *'bvls fallback'*) or whose second step fell back to `zhat` are counted in `self.rmsa_failures` and reported by one summary *RuntimeWarning* with their count and reasons.

**Parameters:**
`reset` : *bool*, default = *False*<br>
    If *True*, forces recomputation of all diagnostic values (the results are preserved for eventual reproduction after the method is called). The stored values are invalidated by every `solve()`.

`threshold` : *float*, default = *0*<br>
    If positive, limits the output to constraints with `RMSA_i` ≥ `threshold`.

**Returns:**
*dict* of *list*<br>
    A dictionary containing per-row diagnostic values:<br>
    {<br>
        `"constraint"`   : `[1, 2, ..., k]`,  # 1-based indices<br>
        `"rmsa_i"`       : list of `RMSA_i` values,<br>
        `"rmsa_dkappaC"` : list of Δκ(`C`) after deleting row `i`,<br>
        `"rmsa_dkappaB"` : list of Δκ(`B`) after deleting row `i`,<br>
        `"rmsa_dkappaA"` : list of Δκ(`A`) after deleting row `i`,<br>
        `"rmsa_dnrmse"`  : list of Δ`nrmse` after deleting row `i`,<br>
        `"rmsa_dzhat"`   : list of Δ‖`zhat`‖_2 after deleting row `i`,<br>
        `"rmsa_dz"`      : list of Δ‖`z`‖_2 after deleting row `i`,<br>
        `"rmsa_dx"`      : list of Δ‖`x`‖_F|2 after deleting row `i`,<br>
    }

### T-Test Method: `ttest`

```python
self.ttest(reset, sample_size, seed, distribution, partial, simulate)
```

Perform bootstrap or Monte Carlo t-tests on the NRMSE statistic from the CLSP estimator.

This function either (a) resamples residuals via a nonparametric bootstrap to generate an empirical NRMSE sample, or (b) produces synthetic right-hand side vectors `b` from a user-defined or default distribution and re-estimates the model. It then runs a one-sample t-test of whether the mean of the resampled or simulated NRMSE values equals the observed NRMSE (see [Diagnostics](#diagnostics) for the interpretation).

**Parameters:**
`reset` : *bool*, default = *False*<br>
    If *True*, forces recomputation of the NRMSE null distribution (under H₀) (the results are preserved for eventual reproduction after the method is called). The null distribution is also recomputed unless (`sample_size`, `seed`, `distribution`, `partial`, `simulate`) equal those of the stored sample, and after every `solve()`.

`sample_size` : *int*, default = *50*<br>
    Size of the Monte Carlo simulated sample under H₀ (an integer ≥ 2).

`seed` : *int* or *None*, optional<br>
    Optional random seed to override the default. If *None*, the stored seed and the current state of the random generator are used.

`distribution` : *str* or *None*, default = *'normal'*<br>
    Distribution for generating simulated `b` vectors. One of (standard): *'normal'*, *'uniform'* (on [-1, 1]), or *'laplace'*. Used only if `simulate` is *True*.

`partial` : *bool*, default = *False*<br>
    If True, runs the t-test on the partial NRMSE: during simulation, the C-block entries are preserved and the M-block entries are simulated; the bootstrap resamples the model-row residual e_M = `b`_M - `M`x, the quantity of `nrmse_partial`. Without `M` rows, a *RuntimeWarning* is issued and the full NRMSE is used.

`simulate` : *bool*, default = **False**<br>
    If True, performs a parametric Monte Carlo simulation by generating synthetic right-hand side vectors `b` (re-estimated with the fitted settings, `sample_size` refits). Failed refits (first step not converged, second step fallback) are counted (`"failures"`) and reported by one summary *RuntimeWarning*. If False (default), executes a nonparametric bootstrap procedure on residuals without re-estimation.

**Returns:**
*dict*<br>
    Dictionary with test results and null distribution statistics:<br>
    {<br>
        `'p_one_left'`  : P(nrmse ≤ null mean),<br>
        `'p_one_right'` : P(nrmse ≥ null mean),<br>
        `'p_two_sided'` : 2-sided t-test p-value,<br>
        `'nrmse'`       : observed value,<br>
        `'mean_null'`   : mean of the null distribution (under H₀),<br>
        `'std_null'`    : standard deviation of the null distribution (under H₀),<br>
        `'failures'`    : number of failed refits (0 for the bootstrap)<br>
    }<br>
    The p-values are *nan* (with one *RuntimeWarning*) if the observed NRMSE or the null sample is not finite (sd(`b`) = 0, or sd(`b`_M) = 0 for `partial=True`; `mean_null` and `std_null` are then *nan* for a non-finite null sample) or if the null sample has zero variance. A *CLSPError* is raised for an invalid `sample_size` or `distribution`.

### Cross-Validation Method: `cv_alpha()`

```python
self.cv_alpha(alphas, folds, seed, refit)
```

Selects α for the second step by K-fold cross-validation over the `M` rows.

The `M` rows are split at random into `folds` folds; the `C` rows are never held out. For each fold and each candidate α, the model is refitted without the held-out rows with the fitted settings (`final`, `r`, `rcond`, `Z`, `tolerance`, `iteration_limit`, `bounds`, `slack_sign`, `scale`), `b`_M of the held-out rows is predicted by `M`x, and the root mean squared prediction error (RMSE) is recorded (`len(alphas)` · `folds` refits). Warnings raised by the individual refits are suppressed; instead, every fold fit is checked: it is valid only if its first step is *'converged'* or *'bvls fallback'* and its second step is not a fallback to `zhat`. A failed fold fit is retried once with CLARABEL; the second-step solver and the retry of every fold fit are logged in `fits`. An α with a failed fold is invalid: its mean and standard deviation are *nan*, it is excluded from the selection (candidates are never averaged over different fold subsets), and it is reported with a *RuntimeWarning*. The scores assume that holding out random `M` rows is appropriate (see [Choosing α by Cross-Validation](#choosing-α-by-cross-validation)).

The method first checks whether the comparison can be informative. If there are no `M` rows, fewer `M` rows than `folds`, `final` is *False*, or no bounds or signed slacks are active and `Z` is diagonal (the second step then returns `zhat` for every α), it makes no refit and returns `informative` = *False* and `best` = 1.0 with a *RuntimeWarning*. After the refits, the result is also not informative if the mean RMSE of every α equals that of α = 1 within a relative `tolerance` ("all α give the same predictions"). If fewer than two α are valid, the result is not informative ("no valid α" or "one valid α") with a *RuntimeWarning*, and `best` is 1.0, or *nan* if no α or α = 1 itself is invalid. Ties are resolved in favour of α = 1 (or the largest valid candidate).

**Parameters:**
`alphas` : *tuple* or *list*, default = *(0, 0.25, 0.5, 0.75, 1)*<br>
    Candidate values of α in [0, 1].

`folds` : *int*, default = *5*<br>
    Number of folds K (an integer ≥ 2).

`seed` : *int* or *None*, optional<br>
    Optional random seed to override the default (as in `ttest()`). If *None*, the stored seed and the current state of the random generator are used.

`refit` : *bool*, default = *False*<br>
    If *True* and the result is informative, the model is refitted with the best α (`self.alpha`). Otherwise, the model is left unchanged.

**Returns:**
*dict*<br>
    Dictionary with the cross-validation results, also stored in `self.cv_alpha_result` (reset by every `solve()`):<br>
    {<br>
        `'alpha'`        : list of the candidate values of α,<br>
        `'cv_rmse_mean'` : list of the mean RMSE over the folds for each α (*nan* if no refit was made or α is invalid),<br>
        `'cv_rmse_sd'`   : list of the standard deviation of the RMSE over the folds for each α,<br>
        `'best'`         : α with the smallest mean RMSE among the valid α (1.0 if the result is not informative; *nan* if no α or α = 1 is invalid),<br>
        `'informative'`  : whether α changes the predictions,<br>
        `'reason'`       : reason of the result,<br>
        `'valid'`        : list of bool, whether all fold fits of each α are valid,<br>
        `'failures'`     : list of the failed fold fits, {`'alpha'`, `'fold'`, `'reason'`},<br>
        `'fits'`         : list of all fold fits, {`'alpha'`, `'fold'`, `'solver'`, `'retry'`, `'valid'`} (`'retry'`: reason of a retry with CLARABEL or *None*)<br>
    }<br>
    A *CLSPError* is raised for a model that is not fitted, invalid `alphas` or invalid `folds`.

### Summary Method: `summarize` or `summary`

```python
self.summarize(display)
self.summary(display)
```

Return or print a summary for the CLSP estimator.

**Parameters:**
`display` : *bool*, default = **False**<br>
    If True, prints the summary instead of returning a dictionary.

**Returns:**
*dict* (or *None* if `display` is *True*)<br>
    Dictionary of estimator configuration, numerical stability, and goodness of fit statistics (the stored attributes except the matrices and vectors `A`, `A_base`, `b`, `Z`, `zhat`, `zhat0`, `z`, `x`, `y`, `sv_discarded`, the bounds arrays, the index tuples, the masks, the fit specification, the t-test sample and the `cv_alpha()` result; the row-wise and band vectors are reduced to min, max, mean and sd):<br>
    {<br>
        `'inverse'`    : type of generalized inverse used ('Bott-Duffin' or 'Moore-Penrose'),<br>
        ...<br>
        `'final'`      : boolean flag indicating whether the second step was present,<br>
        ...<br>
        `'x_bounds'`   : bounds on x ('none', '[lo, hi]', or the ranges of the lower and upper bounds),<br>
        `'slack_sign'` : boolean flag indicating whether the S-block slacks are restricted to be ≥ 0,<br>
        `'n_fixed'`    : number of components of z fixed at a bound in the first step,<br>
        `'scale'`      : boolean flag indicating whether the columns of `A`·`Z` are scaled,<br>
        `'rho'`, `'residual'`, `'status'` : second-step target, residual and statuses,<br>
        `'rank'`       : effective rank of `A`·`Z`,<br>
        ...<br>
        `'rmsa'`       : total RMSA,<br>
        ...<br>
        `'r2_partial'` : coefficient of determination for the M block within A,<br>
        ...<br>
    }

If `display` is *True*, the printed estimator configuration includes the lines *Bounds (x)*, *Slack sign (S ≥ 0)*, *Column scaling* and *Fixed variables*, and *Step 2 retry* if the last solve retried Step 2 with CLARABEL; the numerical stability block includes `rank`; the RMSA block is printed after `corr()` has been called (see Example 5).

## Warnings and Troubleshooting

pyclsp reports problems in two ways. Invalid inputs raise a *CLSPError* (`clsp.errors.CLSPError`). A numerical step that does not reach its target issues a *RuntimeWarning*, and the result is still returned: the fallback is recorded in `self.status` (see Example 11), which is the reliable way to check a fit programmatically. Python's `warnings.simplefilter("error", RuntimeWarning)` turns these warnings into exceptions. `corr()`, `ttest(simulate=True)` and `cv_alpha()` suppress the warnings of their individual refits and issue one summary warning instead.

### Warnings of `solve()`

| Warning | Meaning | What to do |
|---|---|---|
| *RuntimeWarning*: Bounded least-squares active set did not converge (*reason*); using scipy BVLS | The active-set method of the bounded Step 1 cycled (*cycle*), reached `iteration_limit` passes (*iteration_limit*), or its solution failed the KKT verification (*KKT error … > tol_g = …*). `scipy.optimize.lsq_linear(..., method="bvls")` solved the same problem, and its solution passed the same verification; `status["step1"]` = *'bvls fallback'*. | Usually nothing: the result is a verified bounded least-squares solution. If it occurs often, check the scaling of the columns (`scale=True`, see [Limitations](#limitations)) or, for *iteration_limit*, increase `iteration_limit`. |
| *RuntimeWarning*: Bounded least-squares active set did not converge (*reason*; scipy BVLS: *error*); the last feasible estimate is kept | Both methods failed. zhat is the last feasible point of the active-set method: it satisfies the bounds but is not a verified minimizer; `status["step1"]` = *'not converged: <reason>'*, and `status["kkt_error"]` gives its KKT violation. Step 2 is still run, but its target w* is then not the bounded least-squares fit. | Do not use the estimate as a least-squares solution. Equilibrate the columns with `scale=True` (or rescale the columns of `C`, `S` and `M` and the bounds yourself), or discard near-null directions with a larger `rcond` (Example 9). |
| *RuntimeWarning*: Step 2 failed (*reason*); falling back to zhat | Step 2 was not accepted: the solver raised an error (*SolverError: …*), returned a status other than *optimal* or *optimal_inaccurate* (*status …*), or its solution failed the verification (*verification failed: ‖A_k·z - w\*‖₂ = …, bound violation = …, tol_v = …*; *final verification failed: …* for the check of the returned z); with the default solver for α = 1 (OSQP), this happens only after a retry with CLARABEL. z = zhat, `status["final"]` = *'fallback: <reason>'*, and `residual` is that of zhat. | The Step 1 estimate remains a (bounded) least-squares solution; only the selection by α is lost. Try `scale=True`, another solver (`solve(..., solver="CLARABEL")`, `"ECOS"` or `"SCS"`) or solver options passed as keyword arguments. Without active bounds or signed slacks and with a diagonal `Z`, Step 2 does not change zhat, and `final=False` skips it. |

CVXPY's *UserWarning* "Solution may be inaccurate" is not passed on: an *optimal_inaccurate* solution is accepted only if it passes the verification (`status["final"]` = *'optimal_inaccurate'*), and any other inaccurate status leads to the retry with CLARABEL (default OSQP, α = 1) or to the fallback above. With sd(`b`) = 0, `solve()` issues no warning: NRMSE is *inf*, R² is *nan*, and for `r` > 1 the iteration stops after the second iteration.

### Warnings of the Diagnostics

| Warning (*RuntimeWarning*) | Meaning | What to do |
|---|---|---|
| corr(): *N* of *k* refits failed (*reasons*) | *N* of the *k* row-deletion refits had a Step 1 that did not converge (*step1: …*) or a Step 2 fallback (*final: …*); counted in `self.rmsa_failures`. Their values are still included in the correlogram. | Interpret the affected rows with care; the causes are those of the `solve()` warnings above. |
| ttest(): *N* of *S* refits failed (*reasons*) | The same for the refits of `ttest(simulate=True)`; counted in `"failures"`. Their NRMSE values are still included in the null sample. | As above. |
| No M-block present in A; falling back to full NRMSE t-test. | `partial=True` for a model without model rows. | Use `partial=False`. |
| The NRMSE is not finite (sd(b) = 0); p-values are undefined (nan). | The observed NRMSE or a value of the null sample is *inf*, i.e., sd(`b`) = 0 (sd(`b`_M) = 0 for `partial=True`); the p-values are *nan*, and so are `mean_null` and `std_null` if the null sample is not finite. | Expected for a constant right-hand side; use `residual` instead of the normalized statistics. |
| The null NRMSE sample has zero variance; p-values are undefined (nan). | All values of the (finite) null sample are equal, e.g. for an exact fit, so no t-statistic exists. | Check the model; no test is possible. |
| cv_alpha(): α = [*…*] invalid, *N* failed fold fits (*reasons*) | Fold fits that failed also after the retry with CLARABEL; these α are excluded from the selection (`valid`, `failures`). | Inspect `failures`; the causes are those of the `solve()` warnings above. |
| cv_alpha() is not informative (*reason*); best = *…* | *reason* is one of *no M rows*, *fewer M rows (l) than folds (K)*, *final is False, α is not used*, *no bounds or slack signs and a diagonal Z, Step 2 returns zhat for every α* (no refits are made in these four cases), *no valid α*, *one valid α*, or *all α give the same predictions*. `best` is 1.0 (*nan* if no α or α = 1 itself is invalid) and `informative` is *False*. | Expected when α cannot change the estimate: keep α = 1. Otherwise, reduce `folds` or inspect `failures`. |

### Errors (*CLSPError*)

| Message | Cause and remedy |
|---|---|
| Right-hand side vector b must be provided. | `b` is missing in a fit with new `C`, `M` or (`m`, `p`). |
| At least one of C, M, m, or p must be provided. / At least one of C or M must be provided. | No specification is given and none is stored (a first `solve()` needs `C` and/or `M`, or `m` and `p` for AP (TM)). |
| Both m and p must be specified. | An AP (TM) problem without `m` or `p`; note that any `problem` string containing *'ap'* or *'tm'* selects AP (TM). |
| m = … must be divisible by i = … / p = … must be divisible by j = … | Group sizes that do not divide the dimensions of X. |
| C must have … columns | A `C` supplied to an AP (TM) problem must have `m`·`p` columns. |
| Both C and M must be provided. | A CMLS (RP) problem requires both blocks. |
| C and S must have the same number of rows: … / C and M must have the same number of columns: … | Non-conformable blocks. |
| The matrix A and vector b must have the same number of rows: … | `b` must have one entry per row of `A`: the constraint rows (for AP (TM), the totals and then the rows of a supplied `C`), then the model rows (the zeros of `zero_diagonal` are appended automatically). |
| Number of refinement iterations r must be ≥ 1. | Invalid `r`. |
| Matrix Z must be symmetric, idempotent and match the number of columns in A: … | `Z` acts on all columns of `A` (x and the slacks of `S`), so it is (n + s) × (n + s). |
| bounds must be None, a (lo, hi) pair, or a sequence of … (or 1) (lo, hi) pairs with lo ≤ hi | Malformed `bounds`, NaN, or lo > hi. |
| bounds require a diagonal (0/1) Z / scale requires a diagonal (0/1) Z | `bounds`, `slack_sign=True` or `scale=True` combined with a non-diagonal `Z`. |
| bounds conflict with Z: a coordinate excluded by Z (Z_jj = 0) is 0, but its bounds exclude 0 | A coordinate excluded by `Z` must admit 0. |
| alpha must be a single value in [0, 1]; selection by NRMSE was removed in 5.0.0 — use cv_alpha() / alpha must be a single value in [0, 1], got … | A list, tuple or array of α, or a non-numeric, non-finite or out-of-range α. |
| C, S, M, b and Z must be finite (no NaN or inf) | Non-finite entries in the inputs; they are rejected before the SVD. |
| Pseudoinverse estimate zhat failed: SVD did not converge | The SVD of `A`·`Z` failed (rare for finite inputs). |
| Pseudoinverse estimate zhat failed | The Step 1 estimate is not finite (e.g., by overflow for extremely large entries). |
| sample_size must be an integer ≥ 2, got … / Unsupported distribution: … | Invalid arguments of `ttest()`. |
| cv_alpha() requires a fitted model. / alphas must be a sequence of values in [0, 1], got … / folds must be an integer ≥ 2, got … | Invalid use or arguments of `cv_alpha()`. |

## Limitations

- **Soft constraint rows.** The rows of `C` are least-squares rows like those of `M`: in an inconsistent system, they are traded off against the model rows and are not met exactly (Examples 1, 5 and 10). A hard-constraint mode is not implemented. Exact restrictions are available only as `bounds` on x (box and sign restrictions, lo = hi to fix a value, Examples 2 and 10). A large row weight approximates a linear restriction at the cost of a larger κ(`A`) (Example 3), and `r` > 1 meets the constraint rows exactly only under the conditions stated in [Residual-Sign Refinement](#residual-sign-refinement-r--1), by changing the estimator (Example 8).
- **Signed slacks are not hard inequalities.** `slack_sign=True` restricts the sign of the slacks of `S`, so that an inequality row is free on its feasible side; on its infeasible side, it remains a least-squares row (Example 3).
- **Step 2 needs active bounds.** Without active `bounds` or `slack_sign` and with a diagonal `Z` (in particular `Z` = I), Step 2 returns zhat for every α (the feasible anchor, without a solver), for consistent and inconsistent systems alike; α and `cv_alpha()` then have no effect (Example 6).
- **Numerical range.** All steps use `A`_k, so ρ, `rank` and the estimate depend on the cutoff `rcond`: a discarded singular value is treated as exactly zero, even if the full system has an exact solution, and ρ is the smallest residual only within the numerical range of `A` (Example 9). `kappaA` and the sensitivity bands are computed from `A` itself.
- **Which minimizer.** With bounds, Step 1 returns a bounded least-squares minimizer on `A`_k: its fitted vector is unique, but zhat itself is unique only if the free columns have full column rank. The minimum-norm (bounded) least-squares solution is obtained only by Step 2 with α = 1 and `Z` = I (exact up to rounding if `status["polish"]` is *True* or the anchor is feasible, otherwise within the solver accuracy); other α select by the L1/L2 distance to the unbounded anchor zhat0, not by the norm of z.
- **Scaling.** The tolerances are relative (to max|`b`|, ‖`A`_kᵀ`b`‖_∞, ‖`b`‖₂ and σ_max), but by default the columns are not scaled, and columns of very different magnitude are hard for both steps. In 30 random bounded regression problems (`bounds=(0, None)`) for each of three ranges of column scales, spanning 10¹², 10¹⁶ and 10²⁰, the bounded Step 1 did not end with *'converged'* in 0, 8 and 9 problems, and Step 2 fell back to zhat in 1, 11 and 8 problems; with `scale=True` (or with columns rescaled to unit norm before the fit), there were no such failures. The fallbacks keep a feasible Step 1 estimate and are reported (see [Warnings and Troubleshooting](#warnings-and-troubleshooting)). If no singular values are discarded, scaling does not change the (bounded) least-squares fit, but it changes the minimum-norm selection of underdetermined problems, which is then made in the scaled variables (see `scale` in [`solve()`](#solver-method-solve)).
- **Heuristic diagnostics.** The t-tests compare the observed NRMSE with the mean of a resampled or simulated null sample (for the bootstrap, close to the observed value by construction); the sensitivity bands are condition-weighted heuristics without statistical coverage; the RMSA and the correlogram describe the geometry of the constraint rows and the effect of deleting them, not statistical significance. The diagnostic refits that failed are counted but still included (see [Diagnostics](#diagnostics)).
- **Cross-validation assumptions.** `cv_alpha()` holds out random model rows and never constraint rows: this assumes that random row splits suit the application (blocked or forward folds for time-dependent data are not provided), and constraint rows computed from the observed `b`_M leak the held-out outcomes. Small differences between the candidates need not translate into a better estimate (Example 6).
- **Dense linear algebra and problem size.** All matrices are dense NumPy arrays, including the projector `Z` of size (n + s) × (n + s), which is stored also when it is the identity. Every solve computes an SVD of `A`·`Z` (in every refinement iteration), checks the idempotency of `Z` by the dense product `Z`·`Z`, and passes `A`_k to CVXPY, so time and memory grow quickly with the number of unknowns; `corr()` (k refits), `ttest(simulate=True)` (`sample_size` refits) and `cv_alpha()` (`len(alphas)`·`folds` refits) multiply the cost. Indicative timings of `solve()` with `bounds=(0, None)` and `final=True` (α = 1, OSQP) for AP (TM) problems with inconsistent row and column sums of a random non-negative X (about 30 % empty cells, column sums scaled by 1.05) and m noisy known cells as model rows, median of 5 runs (2 for 80 × 80; `corr()`: one run) on an Apple-silicon (arm64) Mac with Python 3.12; all fits converged without fallbacks. The figures are indicative only and vary between runs by about 10 %:

| X | n | `A` | `solve()` | memory of `Z` | `corr()` |
|---|---|---|---|---|---|
| 10 × 10 | 100 | 30 × 100 | 0.006 s | 0.08 MB | 0.08 s (20 refits) |
| 20 × 20 | 400 | 60 × 400 | 0.03 s | 1.3 MB | 1.7 s (40 refits) |
| 40 × 40 | 1,600 | 120 × 1,600 | 0.3 s | 20 MB | 26 s (80 refits) |
| 80 × 80 | 6,400 | 240 × 6,400 | 4 s | 328 MB | not measured |

## What's New in 5.0.0

Version 5.0.0 revises both steps of the estimator; the full list of changes is in [CHANGELOG.md](CHANGELOG.md).

- Stable first step by SVD of `A`·`Z`, with one numerical-rank policy: `A`_k (`A`·`Z` with the singular values below `rcond`·σ_max set to zero) is used in both steps and in the reduced problems of the bounded first step; new attributes `rank` and `sv_discarded`.
- New `bounds` (on x) and `slack_sign` (S-slacks ≥ 0), enforced by bounded-variable least squares (Lawson–Hanson, Stark–Parker) in the first step, verified by the full KKT conditions (`status["kkt_error"]`), with a scipy BVLS fallback, and as constraints in the second step; new attributes `fixed`, `fixed_at`, `zhat0`, `x_lb`, `x_ub`, `S_idx`.
- Reformulated and verified second step, `A`_k`z` = w* anchored at `zhat0`, with the new attributes `rho`, `residual` and `status` (including the first-step outcome, the verification tolerance and the achieved errors), explicit *RuntimeWarning* fallbacks, the anchor returned without a solver when it is feasible, tight OSQP tolerances with a CLARABEL retry for α = 1 (`status["solver"]`, `status["retry"]`; a successful retry issues no warning and is printed by `summary()` as *Step 2 retry*; if the retry fails, a verified *optimal_inaccurate* OSQP solution is kept), CLARABEL for the Elastic Net, no CVXPY *UserWarning* "Solution may be inaccurate" for a verified solution, and, for α = 1, an exact polish of the solver solution on its active set, checked by the KKT conditions (`status["polish"]`).
- α = 1 by default; a single α in [0, 1]; new `cv_alpha()` for the cross-validation of α over the model rows, which checks every fold fit, excludes candidates with a failed fold (`valid`, `failures`) and logs the solver and retry of every fold fit (`fits`).
- `corr()` and `ttest(simulate=True)` count failed refits (`rmsa_failures`, `"failures"`) and report them by one summary *RuntimeWarning*; with sd(`b`) = 0, `ttest()` returns *nan* p-values with one *RuntimeWarning* and no numpy warnings.
- Optional column equilibration `scale=True`: both steps are solved for `A`·`Z`·D, D = diag(1/‖(`A`·`Z`)_j‖₂), in u = D⁻¹`z` (the minimum-norm selection is then made in ‖D⁻¹`z`‖₂), while the diagnostics use the original `A` and `z`; `rank` and `sv_discarded` refer to `A`·`Z`·D; requires a diagonal `Z`; reported in `status["scale"]` and printed by `summary()` as *Column scaling*. The default (*False*) leaves the results bitwise unchanged.
- `Q` built from the model-row residual with a rounding tolerance; `r` > 1 no longer alters consistent systems, and with sd(`b`) = 0 it stops after the second iteration without numpy warnings.
- Non-finite (NaN or inf) entries in `C`, `S`, `M`, `b` or `Z` raise a *CLSPError* before the SVD.
- Stored fit specification: every `solve()` starts from `A` without `Q`; settings kept between calls (`final`, `rcond`, `bounds`, `slack_sign`, `scale`), `solve(b=...)` replaces `b`, the diagnostic refits reuse the fitted settings, and the `corr()`/`ttest()`/`cv_alpha()` results are invalidated by every `solve()`.
- `zero_diagonal=True` appends the rows x_kk = 0 to `M` without merging: a known diagonal cell and its row x_kk = 0 are both fitted, and repeated rows of `M` keep their weight.
- Heuristic sensitivity bands with ordered endpoints, scale-invariant degeneracy checks, stricter `alpha` validation, a test suite, and Python ≥ 3.11.

**Breaking changes:** Python ≥ 3.11; the meaning of `rcond` and its use in both steps; the second-step constraint (`A`_k`z` = w* instead of `Az` = `b`); the defaults of `final` and `rcond` (*None*, keep the stored value) and of `alpha` (1 instead of an error rule); the removal of α lists; the construction of `Q`. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use CLSP in your work, please cite the method:

Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

and the version of the package you used (`clsp.__version__`):

pyclsp, version 5.0.0: Modular Two-Step Convex Optimization Estimator for Ill-Posed Problems [Python package]. [https://github.com/econcz/pyclsp](https://github.com/econcz/pyclsp)

```bibtex
@article{bolotov2025clsp,
  author  = {Bolotov, I.},
  title   = {{CLSP}: Linear Algebra Foundations of a Modular Two-Step Convex
             Optimization-Based Estimator for Ill-Posed Problems},
  journal = {Mathematics},
  year    = {2025},
  volume  = {13},
  number  = {21},
  pages   = {3476},
  doi     = {10.3390/math13213476}
}

@software{pyclsp,
  title   = {pyclsp: Modular Two-Step Convex Optimization Estimator for
             Ill-Posed Problems},
  version = {5.0.0},
  url     = {https://github.com/econcz/pyclsp}
}
```

## Acknowledgements

The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

## Bibliography
Bolotov, I. (2025). CLSP: Linear Algebra Foundations of a Modular Two-Step Convex Optimization-Based Estimator for Ill-Posed Problems. *Mathematics*, *13*(21), 3476. [https://doi.org/10.3390/math13213476](https://doi.org/10.3390/math13213476)

## License
MIT License — see the [LICENSE](LICENSE) file.
