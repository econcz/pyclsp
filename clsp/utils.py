import warnings
import copy
import numbers
import numpy       as np
import scipy.stats as stats

def CLSPCanonicalForm(
    self, problem:        str  = "",   C: np.ndarray | None = None,
          S: np.ndarray | None = None, M: np.ndarray | None = None,
          Q: np.ndarray | None = None, b: np.ndarray | None = None,
          m: int        | None = None, p: int        | None = None,
          i: int               = 1,    j: int               = 1,
          zero_diagonal:  bool = False
) -> None:
    """
    Construct the canonical design matrix A = [C | S; M | Q] for CLSP.

    This method assembles the constraint matrix A from user-supplied or
    internally generated components — C, S, M, and Q — and assigns the
    corresponding right-hand side vector b. It is a required pre-step
    before solving a Convex Least Squares Programming (CLSP) problem.

    Depending on the specified problem type, it can generate allocation,
    tabular matrix, or modular constraints and add optional soft model rows
    for a zero diagonal. All missing blocks are padded to ensure
    conformability.

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
        is raised.

    Q : np.ndarray or None
        Externally supplied residual slack matrix used to adjust inequality
        constraints in M. Required only when r > 1. Encodes the sign pattern
        of residuals from the previous iteration and is used to construct the
        [C | S; M | Q] canonical form. Defaults to a conformable zero matrix
        on the first iteration.

    b : np.ndarray or None
        Right-hand side vector. Must have as many rows as A. Required.

    m, p : int or None
        Dimensions of X ∈ ℝ^{m×p}, relevant for allocation problems ('ap').

    i, j : int, default = 1
        Grouping sizes for row and column sum constraints in AP problems.

    zero_diagonal : bool, default = False
        If True, adds soft model rows x_kk = 0 for the diagonal of X in AP
        problems (exact zeros via equal bounds, lo = hi = 0). The rows are
        appended to M without merging: a known diagonal cell in M and its
        row x_kk = 0 are both fitted in least squares.

    Attributes Set
    ---------------
    self.A : np.ndarray
        Canonical design matrix constructed from (C, S, M, Q).

    self.C_idx : tuple
        Tuple (rows, cols) indicating the size of the C block.

    self.S_idx : tuple
        Tuple (first, last + 1) indicating the columns of the S block.

    self.b : np.ndarray
        Conformable right-hand side vector.

    self.A_base : np.ndarray
        The matrix A as constructed, reused by later solves (A without Q
        unless Q is supplied).

    self.spec : dict
        Inputs of the fit specification (problem, C, S, M, m, p, i, j,
        zero_diagonal).

    Raises
    ------
    CLSPError
        If the design matrix A or right-hand side b is malformed, inconsistent,
        or incompatible with the structural assumptions of the problem.
    """
    # (spec) Inputs of the fit specification
    spec   = copy.deepcopy({"problem": problem, "C": C, "S": S, "M": M,
                            "m": m, "p": p, "i": i, "j": j,
                            "zero_diagonal": zero_diagonal})

    # (b) Ensure the right-hand side is defined and set `self.b`
    if b is None:
        raise self.error("Right-hand side vector b must be provided.")
    self.b = b.astype(np.float64).reshape(-1, 1)

    if  C is not None:
        C = C.astype(np.float64)
    if  S is not None:
        S = S.astype(np.float64)
    if  M is not None:
        M = M.astype(np.float64)
    if  Q is not None:
        Q = Q.astype(np.float64)
    # (A) Option 1. AP (TM) problems with an optional zero diagonal
    if 'ap' in problem.lower() or 'tm' in problem.lower():
        if m is None or p is None:
            raise self.error("Both m and p must be specified.")
        if m % i != 0:
            raise self.error(f"m = {m} must be divisible by i = {i}")
        if p % j != 0:
            raise self.error(f"p = {p} must be divisible by j = {j}")
        # construct the C block using Kronecker product
        row_groups = np.kron(np.kron(np.eye(m // i), np.ones((1, i))),
                             np.ones((1, p)))
        col_groups = np.kron(np.ones((1, m)), np.kron(np.eye(p // j),
                                                      np.ones((1, j))))
        if C is not None:
            if C.shape[1] != row_groups.shape[1]:
                raise self.error(f"C must have {row_groups.shape[1]} columns")
        C =     (np.vstack([row_groups, col_groups, C]) if C is not None
            else np.vstack([row_groups, col_groups]))
        # append an optional identity matrix to M, keep all rows (each one
        # is a least-squares row: a known cell and x_kk = 0 are both kept)
        if zero_diagonal:
            M_flag = (M is None or
                      M.size == 0)
            M_diag = np.zeros((min(m, p), m * p))
            for k in range(min(m, p)):
                M_diag[k, k * p + k] = 1
            M      = M_diag.copy() if M_flag else np.vstack([M, M_diag])
            self.b = np.vstack([self.b[0:C.shape[0]],
                                np.zeros((min(m, p), 1)) if M_flag
                                else np.vstack([self.b[C.shape[0]:],
                                np.zeros((min(m, p), 1))])])
            del M_diag

    # (A) Option 2. CMLS and RP problems
    if 'cmls' in problem.lower() or 'rp' in problem.lower():
        if C is None or M is None:
            raise self.error("Both C and M must be provided.")

    # (A) Option 3. General problems
    if C is None and M is None:
        raise self.error("At least one of C or M must be provided.")

    # (A) Convert missing blocks to conformable zero matrices
    n_col = C.shape[1] if C is not None else M.shape[1]
    C     = C          if C is not None else np.zeros((0, n_col))
    M     = M          if M is not None else np.zeros((0, n_col))
    S     = S          if S is not None else np.zeros((C.shape[0], 0))
    Q     = Q          if Q is not None else np.zeros((M.shape[0], 0))
    if C.shape[0] != S.shape[0]:
        raise self.error(f"C and S must have the same number of rows: "
                         f"{C.shape[0]} vs {S.shape[0]}")
    if C.shape[1] != M.shape[1]:
        raise self.error(f"C and M must have the same number of columns: "
                         f"{C.shape[1]} vs {M.shape[1]}")

    # (A) Pad C and Q with zeros and set `self.A`, `self.C_idx`, `self.S_idx`
    self.A     = np.vstack([
        np.hstack([C, S, np.zeros((C.shape[0], Q.shape[1]))]),
        np.hstack([M,    np.zeros((M.shape[0], S.shape[1])),   Q])
    ])
    self.C_idx = C.shape
    self.S_idx = (C.shape[1], C.shape[1] + S.shape[1])

    # (A_base), (spec) Store the specification for later solves
    self.A_base = self.A
    self.spec   = spec

def CLSPCorrelogram(
    self, reset:       bool = False,       threshold:    float = 0,
) -> dict[str, list[float]]:
    """
    Compute the structural correlogram of the CLSP constraint system.

    This method performs a row-deletion sensitivity analysis on the canonical
    constraint matrix [C | S], denoted as C_canon, and evaluates the marginal
    effect of each constraint row on numerical stability, angular alignment,
    and estimator sensitivity.

    For each row i in C_canon, it computes:
    - The Root Mean Square Alignment (RMSA_i) with all other rows j ≠ i.
    - The change in condition numbers κ(C), κ(B), and κ(A) when row i is
      deleted.
    - The effect on estimation quality: changes in NRMSE, zhat, z, and x.

    Additionally, it computes the total RMSA statistic across all rows,
    summarizing the overall angular alignment of the constraint block.

    Failed refits are counted in `self.rmsa_failures` and reported by a
    RuntimeWarning.

    Parameters
    ----------
    reset : bool, default = False
        If True, forces recomputation of all diagnostic values.

    threshold : float, default = 0
        If positive, limits the output to constraints with RMSA_i ≥ threshold.

    Returns
    -------
    dict of list
        A dictionary containing per-row diagnostic values:
        {
            "constraint"   : [1, 2, ..., k],  # 1-based indices,
            "rmsa_i"       : list of RMSA_i values,
            "rmsa_dkappaC" : list of Δκ(C) after deleting row i,
            "rmsa_dkappaB" : list of Δκ(B) after deleting row i,
            "rmsa_dkappaA" : list of Δκ(A) after deleting row i,
            "rmsa_dnrmse"  : list of ΔNRMSE after deleting row i,
            "rmsa_dzhat"   : list of Δ‖zhat‖_2 after deleting row i,
            "rmsa_dz"      : list of Δ‖z‖_2 after deleting row i,
            "rmsa_dx"      : list of Δ‖x‖_F|2 after deleting row i
        }
    """
    # (RMSA) Total RMSA
    if self.rmsa        is None          or reset:
        k                        = self.C_idx[0]
        p                        = self.C_idx[1]
        C_canon                  = self.A[:k]
        norms                    = np.linalg.norm(C_canon, axis=1)
        self.rmsa                = (lambda C:     np.sqrt(np.sum([(np.dot(
                                           C[i] / norms[i], C[j] / norms[j])
                                    ) ** 2 for i  in range(0,     k - 1)
                                           for j  in range(i + 1, k    )
                                   ]) * 2 / k / (k - 1)))(C_canon)

    # (RMSA) Constraint-wise RMSA, changes in condition numbers, and GoF
    if len(self.rmsa_i) != self.C_idx[0] or reset:
        tmp                      = copy.deepcopy(self)
        k                        = self.C_idx[0]
        p                        = self.C_idx[1]
        C_canon                  = self.A[:k]
        norms                    = np.linalg.norm(C_canon, axis=1)
        self.rmsa_i              = [None] * k
        self.rmsa_dkappaC        = [None] * k
        self.rmsa_dkappaB        = [None] * k
        self.rmsa_dkappaA        = [None] * k
        self.rmsa_dnrmse         = [None] * k
        self.rmsa_dzhat          = [None] * k
        self.rmsa_dz             = [None] * k
        self.rmsa_dx             = [None] * k
        failures                 = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*tmp.*")
            for i in range(k):
                tmp.A_base           = np.delete(self.A_base, i, axis=0)
                tmp.b                = np.delete(self.b,      i, axis=0)
                tmp.C_idx            = (k - 1, p)
                tmp.solve(final=self.final,   alpha=self.alpha,
                          r=self.r_max,       rcond=self.rcond,
                          tolerance=self.tolerance,
                          bounds=self.bounds, slack_sign=self.slack_sign,
                          iteration_limit=self.iteration_limit)
                failures            += [_refit_failure(tmp.status)]
                self.rmsa_i[i]       = (lambda C,  i: np.sqrt(np.sum([(np.dot(
                                               C[i] / norms[i], C[j] / norms[j])
                                        ) ** 2 for j  in range(k) if j != i
                                       ]) / (k - 1)))(C_canon, i)
                self.rmsa_dkappaC[i] = (tmp.kappaC - self.kappaC)
                self.rmsa_dkappaB[i] = (tmp.kappaB - self.kappaB)
                self.rmsa_dkappaA[i] = (tmp.kappaA - self.kappaA)
                self.rmsa_dnrmse[i]  = (tmp.nrmse  - self.nrmse)
                self.rmsa_dzhat[i]   = (np.linalg.norm(tmp.zhat)  -
                                        np.linalg.norm(self.zhat))
                self.rmsa_dz[i]      = (np.linalg.norm(tmp.z)     -
                                        np.linalg.norm(self.z))
                self.rmsa_dx[i]      = (np.linalg.norm(tmp.x,  ord='fro' if
                                        tmp.x.ndim > 1 else None) -
                                        np.linalg.norm(self.x, ord='fro' if
                                        self.x.ndim > 1 else None))
        failures                 = [f for f in failures if f is not None]
        self.rmsa_failures       = len(failures)
        if failures:
            warnings.warn(f"corr(): {len(failures)} of {k} refits failed "
                          f"({'; '.join(sorted(set(failures)))})",
                          RuntimeWarning)

    # Return the correlogram
    indices = [i for i, r in enumerate(self.rmsa_i) if r >= threshold]
    return {
        "constraint"  : [i + 1 for i in indices],      # 1-based indexing
        "rmsa_i"      : [self.rmsa_i[i]       for i in indices],
        "rmsa_dkappaC": [self.rmsa_dkappaC[i] for i in indices],
        "rmsa_dkappaB": [self.rmsa_dkappaB[i] for i in indices],
        "rmsa_dkappaA": [self.rmsa_dkappaA[i] for i in indices],
        "rmsa_dnrmse" : [self.rmsa_dnrmse[i]  for i in indices],
        "rmsa_dzhat"  : [self.rmsa_dzhat[i]   for i in indices],
        "rmsa_dz"     : [self.rmsa_dz[i]      for i in indices],
        "rmsa_dx"     : [self.rmsa_dx[i]      for i in indices],
    }

def CLSPTTest(
    self, reset:       bool = False,       sample_size:  int   = 50,
          seed:        int  | None = None, distribution: str   | None = None,
          partial:     bool = False,       simulate:     bool  = False,
) -> dict[str, float]:
    """
    Perform bootstrap or Monte Carlo t-tests on the NRMSE statistic from
    the CLSP estimator.

    This function either (a) resamples residuals via a nonparametric
    bootstrap to generate an empirical NRMSE sample, or (b) produces
    synthetic right-hand side vectors `b` from a user-defined or default
    distribution and re-estimates the model. It tests whether the observed
    NRMSE significantly deviates from the null distribution of resampled or
    simulated NRMSE values.

    Parameters
    ----------
    reset : bool, default = False
        If True, forces recomputation of the NRMSE null distribution. It is
        also recomputed after a new solve or if any other argument changes.

    sample_size : int, default = 50
        Size of the Monte Carlo simulated sample under H₀ (≥ 2).

    seed : int or None, default = None
        Optional random seed to override the default.

    distribution : str or None, default = None
        Distribution for generating synthetic b vectors. One of:
        'normal', 'uniform', 'laplace'. Defaults to standard normal.

    partial : bool, default = False
        If True, runs the t-test on the partial NRMSE: during simulation,
        the C-block entries are preserved and the M-block entries are
        simulated; the bootstrap resamples e_M = b_M - M·x.

    simulate : bool, default = False
        If True, performs a parametric Monte Carlo simulation by generating
        synthetic right-hand side vectors `b`. If False (default), executes
        a nonparametric bootstrap procedure on residuals without
        re-estimation.

    Returns
    -------
    dict
        Dictionary with test results and null distribution statistics:
        {
            'p_one_left'  : P(nrmse ≤ null mean),
            'p_one_right' : P(nrmse ≥ null mean),
            'p_two_sided' : 2-sided t-test p-value,
            'nrmse'       : observed value,
            'mean_null'   : mean of null distribution,
            'std_null'    : std of null distribution,
            'failures'    : number of failed refits (simulate=True)
        }
        The p-values are nan, with a RuntimeWarning, if an NRMSE is not
        finite (sd(b) = 0) or the null sample has zero variance. Failed
        refits are reported by a RuntimeWarning.
    """
    # Set the seed, RNG configuration, and distribution
    if not isinstance(sample_size, numbers.Integral) or sample_size < 2:
        raise self.error(f"sample_size must be an integer ≥ 2, got "
                         f"{sample_size!r}")
    if seed is not None:
        self.seed = seed
        self.rng  = np.random.default_rng(self.seed)
    if distribution is None:
        distribution = 'normal'
    dist_fn = {
        'normal' : lambda n: self.rng.normal(loc=0,  scale=1, size=(n, 1)),
        'uniform': lambda n: self.rng.uniform(-1, 1, size=(n, 1)),
        'laplace': lambda n: self.rng.laplace(loc=0, scale=1, size=(n, 1))
    }.get(distribution.lower())
    if dist_fn is None:
        raise self.error(f"Unsupported distribution: {distribution}.")
    if partial and self.A.shape[0] == self.C_idx[0]:
        warnings.warn(
        "No M-block present in A; falling back to full NRMSE t-test.",
        RuntimeWarning)
        partial = False
    key = (sample_size, self.seed, distribution.lower(), bool(partial),
           bool(simulate))                             # cache key

    # (t-test) Bootstrap-resampled or simulated NRMSE distribution under H0
    if self.nrmse_ttest_key != key or reset:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            warnings.filterwarnings("ignore", message=".*tmp.*")
            self.nrmse_ttest = [None] * sample_size
            failures         = []
            # (re)generate a nonparametric bootstrap sample
            if not simulate:
                residuals = (self.A @ self.z - self.b
                             if not partial else       # e_M = b_M - M·x
                             self.b[self.C_idx[0]:] -
                             self.A[self.C_idx[0]:, :self.C_idx[1]] @
                             self.z[:self.C_idx[1]])
                for i in range(sample_size):
                    bootstrap = residuals[self.rng.choice(len(residuals),
                                          size=len(residuals), replace=True)]
                    b         = (self.b    if not partial else
                                 self.b[self.C_idx[0]:])
                    self.nrmse_ttest[i] = (lambda residuals, sd:
                              np.linalg.norm(residuals) / np.sqrt(sd.shape[0]) /
                              np.std(sd) if np.std(sd) > 0                else
                              np.inf)(bootstrap.ravel(), b)
            # (re)generate a parametric Monte Carlo sample
            else:
                tmp = copy.deepcopy(self)
                for i in range(sample_size):
                    tmp.b     = (dist_fn(self.b.shape[0])
                                 if not partial else
                                 np.vstack([self.b[:self.C_idx[0]],
                                 dist_fn(self.b[self.C_idx[0]:].shape[0])
                                 ]))                   # simulate b_M only
                    tmp.solve(final=self.final,   alpha=self.alpha,
                              r=self.r_max,       rcond=self.rcond,
                              tolerance=self.tolerance,
                              bounds=self.bounds, slack_sign=self.slack_sign,
                              iteration_limit=self.iteration_limit)
                    self.nrmse_ttest[i] = (tmp.nrmse
                                           if not partial else
                                           tmp.nrmse_partial)
                    failures           += [_refit_failure(tmp.status)]
        failures                  = [f for f in failures if f is not None]
        self.nrmse_ttest_key      = key
        self.nrmse_ttest_failures = len(failures)
        if failures:
            warnings.warn(f"ttest(): {len(failures)} of {sample_size} refits "
                          f"failed ({'; '.join(sorted(set(failures)))})",
                          RuntimeWarning)

    # Return the t-test (nan p-values if undefined)
    nrmse      = self.nrmse if partial == False else self.nrmse_partial
    nrmse_null = np.array(self.nrmse_ttest, dtype=np.float64)
    finite     = bool(np.all(np.isfinite(nrmse_null)))
    mean_null  = np.mean(nrmse_null)         if finite else np.nan
    std_null   = np.std(nrmse_null, ddof=1)  if finite else np.nan
    if not (finite and np.isfinite(nrmse)):
        warnings.warn("The NRMSE is not finite (sd(b) = 0); p-values are "
                      "undefined (nan).", RuntimeWarning)
        p_left  = p_right = p_two = np.nan
    elif std_null > 0:
        t_stat  = ((nrmse - mean_null) /
                   (std_null / np.sqrt(sample_size)))
        p_left  = stats.t.cdf(t_stat, df=sample_size - 1)
        p_right = 1 - p_left
        p_two   = stats.t.sf(abs(t_stat), df=sample_size - 1) * 2
    else:
        warnings.warn("The null NRMSE sample has zero variance; p-values "
                      "are undefined (nan).", RuntimeWarning)
        p_left  = p_right = p_two = np.nan
    return {
        "p_one_left" : p_left,
        "p_one_right": p_right,
        "p_two_sided": p_two,
        "nrmse"      : nrmse,
        "mean_null"  : mean_null,
        "std_null"   : std_null,
        "failures"   : self.nrmse_ttest_failures
    }

def CLSPCVAlpha(
    self, alphas:      tuple | list = (0, 0.25, 0.5, 0.75, 1),
          folds:       int          = 5,
          seed:        int  | None  = None,
          refit:       bool         = False,
) -> dict[str, object]:
    """
    Select alpha for Step 2 by K-fold cross-validation over the M rows.

    The M rows are split at random into `folds` folds; the C rows are never
    held out. For each fold and each α in `alphas`, the model is refitted
    without the held-out rows, b_M of the held-out rows is predicted by M·x,
    and the root mean squared prediction error (RMSE) is recorded. A failed
    fold fit is retried once with CLARABEL; an α with a failed fold is
    invalid (nan scores, a RuntimeWarning).

    The result is not informative (best = 1.0, a RuntimeWarning, no refit)
    if there are no M rows, fewer M rows than folds, `final` is False, no
    bounds or slack signs are active and Z is diagonal, all α give the same
    mean RMSE, or fewer than two α are valid (best = nan if α = 1 is
    invalid). Ties are resolved in favour of α = 1.

    Parameters
    ----------
    alphas : tuple or list, default = (0, 0.25, 0.5, 0.75, 1)
        Candidate values of α in [0, 1].

    folds : int, default = 5
        Number of folds (≥ 2).

    seed : int or None, default = None
        Optional random seed to override the default.

    refit : bool, default = False
        If True and the result is informative, the model is refitted with
        the best α. Otherwise, the model is left unchanged.

    Returns
    -------
    dict
        Dictionary with the cross-validation results (also stored in
        `self.cv_alpha_result`):
        {
            'alpha'        : candidate values of α,
            'cv_rmse_mean' : mean RMSE over the folds for each α,
            'cv_rmse_sd'   : standard deviation of the RMSE for each α,
            'best'         : α with the smallest mean RMSE (1.0 if the
                             result is not informative),
            'informative'  : whether α changes the predictions,
            'reason'       : reason of the result,
            'valid'        : whether all folds of each α are valid,
            'failures'     : failed fold fits, {'alpha', 'fold', 'reason'},
            'fits'         : fold fits, {'alpha', 'fold', 'solver',
                             'retry', 'valid'} (retry: reason of a
                             CLARABEL retry or None)
        }
    """
    # Set the seed, RNG configuration, and candidates
    if self.A_base is None or self.z is None:
        raise self.error("cv_alpha() requires a fitted model.")
    if (not isinstance(alphas, (list, tuple, np.ndarray))   or
        np.size(alphas) == 0                                or
        not all(isinstance(a, numbers.Real) and 0.0 <= a <= 1.0
                for a in np.ravel(alphas))):
        raise self.error(f"alphas must be a sequence of values in [0, 1], "
                         f"got {alphas!r}")
    if not isinstance(folds, numbers.Integral) or folds < 2:
        raise self.error(f"folds must be an integer ≥ 2, got {folds!r}")
    if seed is not None:
        self.seed = seed
        self.rng  = np.random.default_rng(self.seed)
    alphas  = [float(a) for a in np.ravel(alphas)]
    k, n    = self.C_idx
    l       = self.A_base.shape[0] - k                 # number of M rows
    bounded = (np.any(np.isfinite(self.x_lb)) or np.any(np.isfinite(self.x_ub))
               or (self.slack_sign and self.S_idx[1] > self.S_idx[0]))
    reason  = ("no M rows"                                if l == 0      else
               f"fewer M rows ({l}) than folds ({folds})" if l < folds   else
               "final is False, α is not used"            if not self.final
                                                                         else
               "no bounds or slack signs and a diagonal Z, Step 2 returns "
               "zhat for every α"                         if not bounded and
               np.allclose(self.Z, np.diag(np.diag(self.Z)),
                           atol=self.tolerance)                          else
               None)

    # (cv_alpha_result) RMSE of the held-out M rows for each fold and α
    rmse    = np.full((len(alphas), folds), np.nan)
    valid   = np.ones(len(alphas), dtype=bool)
    fails   = []
    fits    = []                                       # solver log
    if reason is None:
        tmp = copy.deepcopy(self)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for f, h in enumerate(np.array_split(self.rng.permutation(l),
                                                 folds)):
                tmp.A_base = np.delete(self.A_base, k + h, axis=0)
                tmp.b      = np.delete(self.b,      k + h, axis=0)
                for a, alpha in enumerate(alphas):
                    retry      = None                  # CLARABEL retry
                    for kw in ({}, {"solver": "CLARABEL"}):
                        tmp.solve(final=self.final,   alpha=alpha,
                                  r=self.r_max,       rcond=self.rcond,
                                  tolerance=self.tolerance,
                                  bounds=self.bounds,
                                  slack_sign=self.slack_sign,
                                  iteration_limit=self.iteration_limit, **kw)
                        error  = _refit_failure(tmp.status)
                        if error is None:
                            break
                        retry  = retry or f"{tmp.status['solver']}: {error}"
                    fits      += [{"alpha": alpha, "fold": f,
                                   "solver": tmp.status["solver"],
                                   "retry": retry or tmp.status["retry"],
                                   "valid": error is None}]
                    if error is not None:              # invalid α
                        valid[a] = False
                        fails   += [{"alpha": alpha, "fold": f,
                                     "reason": error}]
                        continue
                    e          = (self.b[k + h, 0] -   # e = b_M - M·x
                                  self.A_base[k + h, :n] @ tmp.z[:n, 0])
                    rmse[a, f] = np.sqrt(np.mean(e ** 2))
    rmse[~valid] = np.nan                              # no partial scores
    mean    = np.mean(rmse, axis=1)
    i_one   = int(np.argmax(np.where(valid, alphas, -np.inf)))
    if fails:
        warnings.warn(f"cv_alpha(): α = "
                      f"{[a for a, v in zip(alphas, valid) if not v]} "
                      f"invalid, {len(fails)} failed fold fits ("
                      f"{'; '.join(sorted({v['reason'] for v in fails}))})",
                      RuntimeWarning)
    if reason is None and np.sum(valid) < 2:
        reason = "no valid α" if not np.any(valid) else "one valid α"
    if reason is None and (np.nanmax(np.abs(mean - mean[i_one])) <=
                           self.tolerance * abs(mean[i_one])):
        reason = "all α give the same predictions"
    best    = (np.nan                if reason is not None and
                                        (not np.any(valid) or 1.0 in
                                         [v["alpha"] for v in fails])  else
               1.0                   if reason is not None             else
               alphas[i_one]         if mean[i_one] <= np.nanmin(mean) +
                                        self.tolerance *
                                        abs(np.nanmin(mean))           else
               alphas[int(np.nanargmin(mean))])
    if reason is not None:
        warnings.warn(f"cv_alpha() is not informative ({reason}); "
                      f"best = {best}", RuntimeWarning)

    # Refit with the best α and return the result
    if refit and reason is None:
        self.solve(final=self.final, alpha=best, r=self.r_max)
    self.cv_alpha_result = {
        "alpha"       : alphas,
        "cv_rmse_mean": mean.tolist(),
        "cv_rmse_sd"  : np.std(rmse, axis=1, ddof=1).tolist(),
        "best"        : float(best),
        "informative" : reason is None,
        "reason"      : reason or "smallest mean RMSE",
        "valid"       : valid.tolist(),
        "failures"    : fails,
        "fits"        : fits
    }
    return self.cv_alpha_result

def CLSPSummary(
    self, display:     bool = False
) -> dict[str, object] | None:
    """
    Return or print a summary for the CLSP estimator.

    Parameters
    ----------
    display : bool, default = False
        If True, prints the summary instead of returning a dictionary.
    """
    # Define an ancillary np.ndarray summarizing function
    def _summary(a):
        if  a is None or len(np.atleast_1d(a)) == 0 or np.all(np.isnan(a)):
            return dict(min=np.nan, max=np.nan, mean=np.nan, sd=np.nan)
        v = np.asarray(a, dtype=float)
        return dict( min=np.nanmin(v), max=np.nanmax(v),
                    mean=np.nanmean(v), sd=np.nanstd(v, ddof=1))
    summary                     = {k: v for k, v in self.__dict__.items()
                                   if k not in {"A", "C_idx", "S_idx", "b",
                                                "A_base", "spec",
                                                "sv_discarded",
                                                "cv_alpha_result",
                                                "Z", "zhat" , "zhat0",
                                                "z", "x", "y",
                                                "bounds", "x_lb", "x_ub",
                                                "fixed", "fixed_at",
                                                "nrmse_ttest",
                                                "nrmse_ttest_key",
                                                "nrmse_ttest_failures"} and
                                        not callable(v)}
    if self.zhat is not None:
        summary  = {'inverse'   : ("Bott-Duffin" if self.Z is not None and
                                                    not np.allclose(self.Z,
                                                    np.eye(self.Z.shape[0]),
                                                    rtol=self.tolerance,
                                                    atol=self.tolerance) else
                                   "Moore-Penrose"),          **summary}
        summary["x_bounds"]     = ("none" if self.bounds is None         else
                                   f"[{self.x_lb[0]:g}, {self.x_ub[0]:g}]"
                                   if (np.all(self.x_lb == self.x_lb[0]) and
                                       np.all(self.x_ub == self.x_ub[0])) else
                                   f"lb∈[{np.min(self.x_lb):g}, "
                                   f"{np.max(self.x_lb):g}], "
                                   f"ub∈[{np.min(self.x_ub):g}, "
                                   f"{np.max(self.x_ub):g}]")
        summary["n_fixed"]      = (int(self.fixed.sum())
                                   if self.fixed is not None else 0)
    if self.rmsa is not None:
        summary["rmsa_i"]       = _summary(summary.get("rmsa_i"))
        summary["rmsa_dkappaC"] = _summary(summary.get("rmsa_dkappaC"))
        summary["rmsa_dkappaB"] = _summary(summary.get("rmsa_dkappaB"))
        summary["rmsa_dkappaA"] = _summary(summary.get("rmsa_dkappaA"))
        summary["rmsa_dnrmse"]  = _summary(summary.get("rmsa_dnrmse"))
        summary["rmsa_dzhat"]   = _summary(summary.get("rmsa_dzhat"))
        summary["rmsa_dz"]      = _summary(summary.get("rmsa_dz"))
        summary["rmsa_dx"]      = _summary(summary.get("rmsa_dx"))
    if self.z    is not None:
        summary["z_lower"]      = _summary(summary.get("z_lower"))
        summary["z_upper"]      = _summary(summary.get("z_upper"))
        summary["x_lower"]      = _summary(summary.get("x_lower"))
        summary["x_upper"]      = _summary(summary.get("x_upper"))
        summary["y_lower"]      = _summary(summary.get("y_lower"))
        summary["y_upper"]      = _summary(summary.get("y_upper"))

    # (summary) Return the summary
    if not display:
        return summary or None

    # (summary) Print the result
    def _print(d, k, p=None, n=None):
        k = k if isinstance(k, list) else [k]
        n = n if isinstance(n, list) else [n]
        p = p or ''
        for i in range(len(k)):
            v = d.get(f"{p}{k[i]}")
            l = n[i] if i < len(n) and n[i] else k[i]
            if not isinstance(v, dict):                # print value
                fmt = "  {:<20} {:>15}"
                print(fmt.format(f"{l}:", _format(v)))
            else:                                      # print list
                fmt = "  {:<14}  min={:>10}  max={:>10}  mean={:>10}  sd={:>10}"
                print(fmt.format(f"{l}:", _format(v.get('min')),
                                          _format(v.get('max')),
                                          _format(v.get('mean')),
                                          _format(v.get('sd'))))
    def _format(v, width=10):
        fmt = f"{{:>{width}}}"
        if          (not isinstance(v, (int, float, np.number)) or
                         isinstance(v, int)):
            return fmt.format(str(v))
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return fmt.format("nan")
        dp       = max(3, width // 2 - 1)
        ep       = max(2, dp - 1)
        return fmt.format(f"{v:.{dp}f}" if (abs(v) >= 10**-(dp + 1)      and
                                            abs(v) <  10**(width - dp)   and
                                            len(f"{v:.{dp}f}") <= width) else
                          f"{v:.{ep}e}")
    print("Estimator Configuration:")
    _print(summary, "inverse",    n="Generalized inverse")
    _print(summary, "r",          n="Iterations (r)")
    _print(summary, "tolerance",  n="Tolerance")
    _print(summary, "final",      n="Final correction")
    _print(summary, "alpha",      n="Regularization (α)")
    if (self.status or {}).get("retry") is not None:
        _print(self.status, "retry",  n="Step 2 retry")
    _print(summary, "x_bounds",   n="Bounds (x)")
    _print(summary, "slack_sign", n="Slack sign (S ≥ 0)")
    _print(summary, "scale",      n="Column scaling")
    _print(summary, "n_fixed",    n="Fixed variables")
    print("\nNumerical Stability:")
    _print(summary, ["kappaC", "kappaB", "kappaA", "rank"])
    if summary.get("rmsa")    is not None:
        _print(summary, "rmsa")
        _print(summary, ["i", "dkappaC", "dkappaB", "dkappaA",
                         "dnrmse", "dzhat", "dz", "dx"], p="rmsa_")
    print("\nGoodness of Fit:")
    _print(summary, ["r2_partial", "nrmse", "nrmse_partial"])
    _print(summary, ["z_lower", "z_upper", "x_lower", "x_upper",
                     "y_lower", "y_upper"])

def _refit_failure(
    status:    dict
) -> str | None:
    """
    Return the reason of a failed diagnostic refit (None if it succeeded).

    A refit fails if Step 1 did not converge ("converged" or "bvls
    fallback") or Step 2 fell back to zhat.

    Parameters
    ----------
    status : dict
        `self.status` of the refit.

    Returns
    -------
    str or None
        "step1: <status>", "final: <status>" or None.
    """
    return (f"step1: {status['step1']}"
            if status["step1"] not in {"converged", "bvls fallback"} else
            f"final: {status['final']}"
            if str(status["final"]).startswith("fallback")          else
            None)
