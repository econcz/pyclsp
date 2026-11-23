# Changelog

All notable changes to this project are documented in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [5.0.0] – 2026-11-23

Version 5.0.0 revises both steps of the estimator. The preparation of version 5.0.0 — code changes, tests and documentation — was assisted by Claude (Anthropic), an AI assistant.

### Added

- `bounds` parameter of `solve()`: bounds on x as a (lo, hi) pair or one pair per variable (*None* or ±inf = unbounded side; (*None*, *None*) gives the same results as no bounds). In Step 1, they are enforced by bounded-variable least squares on A_k (Lawson–Hanson / Stark–Parker): from a feasible start, least-squares steps on the free columns (pseudoinverse of the columns of A_k in the orthonormal basis U_k of its range, with a machine-precision cutoff relative to these columns only) are taken up to the first blocking bound (interpolation), and the fixed variable with the largest KKT violation of g = A_kᵀ(A_k·z − b) is released. The result is verified by the full KKT conditions (|g_j| ≤ tol_g for the free variables, g_j ≥ −tol_g at a lower and g_j ≤ tol_g at an upper bound, tol_g = `tolerance`·max(1, ‖A_kᵀb‖_∞)). A repeated free set (cycle guard), `iteration_limit` passes or a failed verification switch to `scipy.optimize.lsq_linear(..., method="bvls")` on the same problem with a *RuntimeWarning* (its solution is verified in the same way); if that fails too, the last feasible point is kept with a *RuntimeWarning*. In Step 2, the bounds are imposed as constraints.
- `scale` parameter of `solve()` (column equilibration, default *False*): both steps are solved for A·Z·D with D = diag(1/‖(A·Z)_j‖₂) (1 for zero columns) in the variables u = D⁻¹z (bounds divided accordingly), and z = D·u is returned. The minimum-norm selection (zhat0, Step 2 with α = 1) and the Step 2 norms are then taken in u, i.e. in ‖D⁻¹z‖₂; the diagnostics (residual, NRMSE, κ, bands) use the original A and z; `rank` and `sv_discarded` refer to A·Z·D. Requires a diagonal (0/1) `Z`. The setting is stored (attribute `scale`) and used by the diagnostic refits. With the default, the results are bitwise unchanged.
- `slack_sign` parameter of `solve()`: restricts the slack variables of the `S` block to be ≥ 0 (S_ii = +1 encodes ≤, S_ii = −1 encodes ≥), enforced in the same way as `bounds`; the residual slacks (`Q`) are never constrained.
- Validation of `bounds`: lo ≤ hi, no NaN, 1 or n pairs; bounds or signed slacks require a diagonal (0/1) `Z`, and bounds that exclude 0 on a coordinate excluded by `Z` (Z_jj = 0) raise a *CLSPError*.
- Validation of the inputs: NaN or inf in `C`, `S`, `M`, `b` or `Z` raises a *CLSPError* ("C, S, M, b and Z must be finite (no NaN or inf)") before the SVD.
- New attributes: `S_idx` (column range of the `S` block), `bounds`, `x_lb`, `x_ub`, `slack_sign`, `zhat0` (unbounded Step 1 estimate A_k⁺b, the anchor of Step 2), `rank` (effective rank of A·Z) and `sv_discarded` (singular values of A·Z below the cutoff), `fixed` and `fixed_at` (components at a bound at the termination of Step 1 and the pass at which they were fixed), `r_max` (requested `r`), `rho` (smallest residual norm ‖b − w*‖₂ within the numerical range of A), `residual` (‖b − A·z‖₂ of the returned z with the original A), `status`, `nrmse_ttest_key` (cache key of the t-test sample), `rmsa_failures` and `nrmse_ttest_failures` (failed refits of `corr()` and `ttest(simulate=True)`), `A_base` (A without Q) and `spec` (inputs of the fit specification), `cv_alpha_result`.
- `status` keys: `"step1"` (*'converged'*, *'bvls fallback'* or *'not converged: <reason>'*), `"rho"` (*'pinv'* or *'bvls'*), `"final"` (CVXPY status or *'fallback: <reason>'*), `"solver"` (the solver actually used, *'none (anchor feasible)'* if no solver was needed), `"retry"` (reason of a CLARABEL retry or *None*), `"polish"` (whether the α = 1 solution was polished on its active set), `"tol_v"` (verification tolerance), `"scale"` (whether the columns were scaled) and the achieved errors `"kkt_error"` (largest KKT violation of the Step 1 estimate on A_k), `"bound_error"` (largest bound violation of z) and `"eq_error"` (‖A_k·z − w*‖₂).
- Verification of Step 2: the solution is accepted only if the bound violation of the solver output is ≤ tol_v and, after clipping to the bounds, ‖A_k·z − w*‖₂ ≤ tol_v, tol_v = sqrt(`tolerance`)·max(1, ‖b‖₂). The check is applied once more to the returned z on every path (solver, polish, feasible anchor), so that `status["final"]` is never *'optimal'* with `status["eq_error"]` > tol_v; the polish verifies its point after clipping to the bounds.
- Active-set polish of the Step 2 solution for α = 1: the variables of a verified solver solution within tol_v (then within `tolerance`·max(1, ‖b‖₂)) of a finite bound, the signed slacks at 0 included, are fixed, and the projection of the anchor on the remaining variables subject to A_k·z = w* is solved exactly, u_F = u0_F + P⁺(r − P·u0_F) with P = U_kᵀA_k[:, F] (pseudoinverse cutoff of the reduced problems of Step 1). The polished point is kept only if its bound violation, ‖A_k·z − w*‖₂ and the sign violations of the bound multipliers (KKT; a small linear program if the equality multipliers are not unique) are ≤ `tolerance`·max(1, ‖b‖₂); otherwise the solver solution is kept. Not applied for α < 1 or to a feasible anchor. Recorded in `status["polish"]` (*True*/*False*, *None* if Step 2 is skipped). On 800 random bounded problems (seeds 1000–1199, four regimes, α = 1), the largest deviation of z from the exact projection fell from 4.0·10⁻⁸ to 4.6·10⁻¹³ and the largest residual gap to BVLS from 6.4·10⁻¹⁰ to 8.8·10⁻¹⁴; e.g. C = [1, 2], b = 5, 0 ≤ x₁ ≤ 0.5, x₂ ≥ 0 gives z = (0.5, 2.25) exactly.
- `cv_alpha(alphas, folds, seed, refit)`: K-fold cross-validation of α over the `M` rows (the `C` rows are never held out), returning the mean and standard deviation of the prediction RMSE for each α, the best α (ties: α = 1), and whether the result is informative with a reason; the result key `fits` logs the Step 2 solver and the retry of every fold fit. Without `M` rows, with fewer `M` rows than folds, with `final=False`, or without active bounds or slack signs and with a diagonal `Z`, no refit is made and α = 1 is returned with a *RuntimeWarning*; `refit=True` refits the model with the best α if the result is informative. Every fold fit is checked (Step 1 *'converged'* or *'bvls fallback'*, Step 2 not a fallback) and a failed one is retried once with CLARABEL; an α with a failed fold is invalid (*nan* scores, excluded from the selection, never averaged over a different subset of folds) and reported in the new result keys `valid` and `failures` (α, fold, reason) with a *RuntimeWarning*; with fewer than two valid α, the result is not informative (`best` = 1, or *nan* if no α or α = 1 is invalid).
- `solve(solver=...)` selects another CVXPY solver for Step 2.
- `solve(b=...)` without `C`, `M` or (`m`, `p`) replaces the stored `b` (with a row-count check) and reuses the stored `A`.
- `summary()`: new entries `x_bounds`, `slack_sign` and `n_fixed`, printed as *Bounds (x)*, *Slack sign (S ≥ 0)* and *Fixed variables*, `scale`, printed as *Column scaling*, a *Step 2 retry* line (the reason) if the last solve retried Step 2 with CLARABEL, and `rank`, printed in the numerical stability block; the dictionary also contains the new `rho`, `residual`, `status` and `r_max`.
- `ttest()`: validation of `sample_size` (an integer ≥ 2); *nan* p-values with a *RuntimeWarning* if the null sample has zero variance or an NRMSE is not finite (sd(b) = 0); result key `"failures"` (failed refits of the simulation).
- `clarabel>=0.6` dependency.
- Test suite `tests/test_clsp.py` (regression and acceptance tests with references computed independently of the package, including a BVLS benchmark; runs under pytest or as a script, against the installed package with `CLSP_TEST_INSTALLED=1`).
- CI: a `test` job (Python 3.11 and 3.12) in the PyPI workflow, on which the release build depends, and a `tests.yml` workflow (push and pull request; Python 3.11, 3.12 and 3.13) that builds the wheel and runs the tests against the installed wheel.
- Full MIT license text in `LICENSE` (previously only the title line).
- `CHANGELOG.md`; README sections on how CLSP works (with problem types and notation), eleven runnable examples, an extended API reference, warnings and troubleshooting, limitations, and citation.

### Changed

- Requires Python ≥ 3.11 (the minimum of cvxpy ≥ 1.8.1); classifiers for Python 3.11, 3.12 and 3.13.
- Step 1 is computed from one SVD of A·Z, as zhat0 = A_k⁺b, equivalent to the Bott–Duffin form (Z·AᵀA·Z)⁺Z·Aᵀb without squaring the condition number of A·Z.
- One numerical-rank policy: A_k, A·Z with the singular values σ ≤ `rcond`·σ_max set to zero, is used in the Step 1 estimate, in the bounded least-squares problem (including its reduced free-column problems, without a second cutoff) and its KKT check, for w* and ρ, and in the Step 2 constraint. `rcond` is a relative cutoff on the singular values of A·Z. The default (*False*) applies sqrt(max(A.shape)·eps), which reproduces the effective cutoff of the previous normal-equations form; *True* applies `tolerance`; a float is used as given.
- `Q` (for `r` > 1) is built as diag(−sign(e_M)) from the model-row residual e_M = b_M − M·x instead of the model rows of b − A·zhat, and entries with |e_i| ≤ `tolerance`·max(1, ‖b‖₂) are treated as exact zeros.
- The fit specification (`problem`, `C`, `S`, `M`, `m`, `p`, `i`, `j`, `zero_diagonal`) and the canonical A without Q (`A_base`) are stored; every `solve()` without `C`, `M` or (`m`, `p`) starts from `A_base` and appends Q for the requested `r`, and the refits of `corr()`, `ttest()` and `cv_alpha()` set `A_base`.
- Step 2 is solved subject to A_k·z = w*, the fitted vector A_k·zhat of the (bounded) least-squares problem of Step 1, instead of A·z = b, and is anchored at `zhat0` instead of `zhat`. The program is always feasible, its residual ‖b − A_k·z‖₂ equals ρ for every α, and for α = 1 and Z = I it returns the minimum-norm (bounded) least-squares solution.
- If the anchor `zhat0` (u0 = D⁻¹·zhat0 with `scale=True`) is feasible, i.e. its bound violation and ‖A_k·z − w*‖₂ for z = zhat0 and for zhat0 clipped to the bounds are ≤ `tolerance`·max(1, ‖b‖₂), Step 2 returns the clipped anchor without calling a solver (checking the clipped anchor prevents a large fit error when a bound cuts off a tiny component of a large column, e.g. C = [10⁸, 1], b = 1, x₁ = 0): the objective is 0 there, so it is the minimizer for every α (unique for α > 0). `status["final"]` is then *'optimal'* and `status["solver"]` *'none (anchor feasible)'*. Step 2 is thus exact without bounds and whenever the bounds do not cut off the anchor.
- Step 2 falls back to `zhat` with a *RuntimeWarning* stating the reason whenever the solver raises an error, returns a status other than *optimal*/*optimal_inaccurate*, or the solution fails verification.
- Elastic Net (0 < α < 1) is solved with CLARABEL instead of SCS; ECOS (α = 0) and OSQP (α = 1) are kept. OSQP is run with `eps_abs` = `eps_rel` = 1e-10 and `max_iter` = 100000 (unless given; polishing is the CVXPY default), and a failed, *optimal_inaccurate* or unverified OSQP solution is retried once with CLARABEL before the fallback (not if `solver=` is passed). A successful retry issues no *RuntimeWarning* (it is recorded in `status["retry"]`); if the retry fails, a verified *optimal_inaccurate* OSQP solution is kept.
- CVXPY's *UserWarning* "Solution may be inaccurate" is suppressed around the Step 2 solves: an *optimal_inaccurate* solution is accepted only after verification, and genuine fallbacks keep their *RuntimeWarning*.
- `alpha` is a single value in [0, 1]; *None* means α = 1 (Ridge: the Step 2 solution is unique and, for Z = I, the minimum-norm (bounded) least-squares solution). NaN, inf, non-numeric or out-of-range values raise a *CLSPError*. The endpoints are exact (`alpha == 0`, `alpha == 1` instead of `np.isclose`).
- `final` and `rcond` default to *None* in `solve()` (keep the stored value, initially *True* and *False*); `bounds`, `slack_sign` and `scale` are kept likewise.
- `corr()` and `ttest(simulate=True)` refit with the fitted settings (`final`, `alpha`, `r`, `rcond`, `Z`, `tolerance`, `iteration_limit`, `bounds`, `slack_sign`, `scale`) instead of the defaults. The warnings of the individual refits remain suppressed, but refits whose Step 1 did not converge or whose Step 2 fell back to `zhat` are counted and reported by one summary *RuntimeWarning* (count and reasons).
- Every `solve()` invalidates the stored `corr()` and `ttest()` results; the t-test sample is cached by (`sample_size`, `seed`, `distribution`, `partial`, `simulate`) instead of its length.
- `ttest(partial=True)` with the bootstrap resamples the model-row residual b_M − M·x, the quantity of `nrmse_partial`.
- `nrmse_partial` and `r2_partial` are *nan* for a model without `M` rows.
- Sensitivity bands are z ∓ |z|·κ(A)·‖b − A·z‖₂ / ‖b‖₂ and are documented as heuristic (no statistical coverage); attributes and docstrings no longer call them confidence bands.
- Degeneracy checks of NRMSE, R² and the bands use sd > 0 (or ‖b‖₂ > 0) instead of `np.isclose(·, 0)`, so they are scale-invariant.
- Docstrings and README state that the rows of `C` are least-squares rows, that only `bounds` and the sign of the `S` slacks are enforced exactly (a signed slack does not make its row a hard equality or inequality), that `zero_diagonal` adds soft model rows (exact zeros via equal bounds), and that a hard-constraint mode is not implemented. The README describes the t-test as a comparison of the observed value with the mean of the simulated values (a larger `sample_size` increases the power for a fixed nonzero difference; p-values need not decrease monotonically), and the Step 2 verification tolerance and achieved errors.
- Version 5.0.0 in `pyproject.toml` and `clsp/__init__.py`.

### Fixed

- `r` > 1 changed the estimate of consistent systems, because rounding-level residuals produced arbitrary signs in `Q`.
- `r` > 1 raised a *CLSPError* when `solve()` reused the stored `A` (and hence in diagnostic refits).
- `solve(b=...)` on a stored `A` silently ignored the new `b`.
- A refit with `r` = 1 after a fit with `r` > 1 kept the old `Q` block; `solve(b=...)` on an AP (TM) model returned `x` as a vector instead of an `m` × `p` matrix; a refit without `M` rows kept the previous `nrmse_partial` and `r2_partial`.
- `r` > 1 with `zero_diagonal=True` rebuilt `b` from the already merged `b` (the Q block is now appended to the stored A).
- `zero_diagonal=True` merged the rows of `M` and the rows x_kk = 0 by their coefficients only (`np.unique`): a known diagonal cell in `M` (e.g. x₁₁ = 3) silently removed its soft row x₁₁ = 0, and repeated rows of `M` were reduced to one, which changed their least-squares weight. All rows are now kept.
- `alpha` given as a *tuple* or *np.ndarray*, or as a non-finite scalar, raised an unrelated *TypeError* (now a *CLSPError*).
- α values close to, but different from, 0 or 1 were rounded to Lasso or Ridge.
- Right-hand sides with a small scale (sd(b) below the absolute tolerance of `np.isclose`) were treated as degenerate (NRMSE = inf).
- Band endpoints were reversed for negative components (z·(1 ∓ dz)).
- `corr()` and `ttest()` returned stale results after a new `solve()`, and `ttest()` reused a sample computed with a different `seed`, `distribution`, `partial` or `simulate`.
- The diagnostic refits ignored the fitted settings (e.g., `final=False`, a fixed `alpha`, `r`, `rcond`, `Z`).
- The partial bootstrap resampled rows of b − A·z, which include the residual slacks for `r` > 1, instead of b_M − M·x.
- The warning for `ttest(partial=True)` without `M` rows was suppressed, and the fallback to the full NRMSE was not applied when a stored sample was reused.
- Step 2 solutions with a non-optimal status were accepted without verification, and solver errors fell back to `zhat` silently.
- The default OSQP tolerances (1e-5) left Ridge solutions with residual gaps of up to 4·10⁻⁵ to the bounded least-squares optimum (within tol_v) and stopped some problems at the iteration limit (fallback to `zhat`).
- Step 2 with a feasible anchor was solved numerically, with errors of order 10⁻⁵ in z: e.g. for C = [1, 2], b = 5, 0 ≤ x₁ ≤ 1, x₂ ≥ 0 and α = 1, OSQP returned *optimal_inaccurate*, and the CLARABEL retry, whose tolerances (~10⁻⁸) apply to the squared objective, returned z ≈ (0.99997, 2.00001) instead of (1, 2), which passed the verification.
- Passing `solver=` to `solve()` raised a *TypeError* (multiple values for the keyword argument `solver`).
- `r` > 1 with sd(`b`) = 0 issued numpy divide-by-zero and invalid-value warnings and never stopped early; the NRMSE of the previous iteration now follows the convention of the other NRMSE values (*inf* if sd = 0), and the iteration stops when NRMSE is *inf* in two successive iterations.
- `ttest()` with sd(b) = 0 (sd(b_M) = 0 for `partial=True`) issued numpy's "invalid value encountered in subtract" (the standard deviation of a null sample of *inf* values) followed by the zero-variance warning, and with `simulate=True` returned finite p-values for the observed NRMSE = *inf*; it now returns *nan* p-values (and *nan* `mean_null` and `std_null` for a non-finite null sample) with one *RuntimeWarning* ("The NRMSE is not finite (sd(b) = 0); p-values are undefined (nan).").
- Non-finite inputs raised "Pseudoinverse estimate zhat failed: SVD did not converge" (`C`, `S`, `M`) or "Pseudoinverse estimate zhat failed" (`b`) instead of a clear message.

### Removed

- Support for Python 3.10.
- The normal-equations form of Step 1, (Z·AᵀA·Z)⁺Z·Aᵀb.
- The selection of α by NRMSE: the error rule for `alpha=None` and the lists of `alpha` candidates (all α attain the same residual, so NRMSE does not discriminate between them); use `cv_alpha()`.
- SCS as the default Elastic Net solver (`scs` remains a dependency).

### Breaking

- Python ≥ 3.11 is required.
- `rcond` is applied to the singular values of A·Z (previously to those of Z·AᵀA·Z); a float `rcond` therefore has a different effective cutoff. The default (*False*) is equivalent to the previous default. The truncated A_k is now used in both steps, so ρ is the smallest residual within the numerical range of A.
- Step 2 is constrained by A_k·z = w* instead of A·z = b (both coincide for consistent systems without bounds and discarded singular values) and is anchored at `zhat0`; it keeps the components outside the range of Z at the anchor.
- `alpha=None` means α = 1 instead of the NRMSE error rule, and a *list*, *tuple* or *np.ndarray* of candidates raises a *CLSPError* ("alpha must be a single value in [0, 1]; selection by NRMSE was removed in 5.0.0 — use cv_alpha()").
- Every `solve()` without `C`, `M` or (`m`, `p`) starts from the stored A without Q, so a refit with a different `r` equals a fresh fit.
- `final` and `rcond` default to *None* in `solve()`: a value set in an earlier call is kept instead of being reset to *True*/*False*.
- `Q` is constructed from the model-row residual b_M − M·x (with a rounding tolerance), which changes the results of `r` > 1.
- Invalid `alpha` values (NaN, inf, non-numeric, out of [0, 1]) raise a *CLSPError*; previously, invalid list candidates were skipped silently, an invalid scalar caused a *TypeError*, and out-of-range values were clipped to [0, 1].
- The diagnostic refits and the partial bootstrap use the new definitions, so `corr()` and `ttest()` results can differ from 4.0.1.
- New dependency `clarabel>=0.6`.
- `zero_diagonal=True` appends the rows x_kk = 0 to `M` without merging or sorting: estimates differ if `M` contains a diagonal cell or repeated rows, and the order of the rows of `M` and `b`_M is kept.

## [4.0.1] – 2026-06-06

- Bug fixes.

## [3.5.0] – 2026-04-21

- Cross-platform condition-number tolerance.

## [2.2.0] – 2026-04-13

- Bug fixes for CLSP statistics, `corr()`, and `ttest()`.

## [2.0.0] – 2025-12-21

- GitHub Actions workflow for PyPI publishing.

## [1.8.8] – 2025-11-17

- Refreshed README.

## [1.8.4] – 2025-11-13

- Bug fixes.

## [1.8.2] – 2025-11-06

- Nonparametric bootstrap t-tests and output summary.

## [1.7.0] – 2025-10-25

- Change in the design matrix A for AP (TM) problems.

## [1.6.0] – 2025-10-12

- Regularization for the generalized inverses.

## [1.4.0] – 2025-09-27

- Fix for the Bott–Duffin inverse in iterated Step 1.

## [1.3.0] – 2025-09-14

- Extended alpha handling and partial t-test option.

## [1.1.8] – 2025-08-22

- Minor bug fixes, extension of the error class.

## [1.1.2] – 2025-08-16

- Initial commit.
