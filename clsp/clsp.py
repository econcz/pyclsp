import warnings
import copy
import numbers
import numpy        as np
import scipy.linalg as la
import scipy.stats  as stats
import cvxpy        as cp
from types          import MethodType
from .errors        import CLSPError
from .utils         import CLSPCanonicalForm, CLSPCorrelogram, CLSPTTest
from .utils         import CLSPSummary,       CLSPCVAlpha
from .solver        import CLSPSolve

class CLSP:
    def __init__(self):
        """
        Instantiate a CLSP (Convex Least Squares Programming) object.

        This class initiates and stores the results of a two-stage estimation
        process for constrained least-squares problems under a canonical matrix
        structure. The approach is designed for underdetermined, ill-posed, or
        structurally constrained systems.

        Stage 1 estimates the solution z from the system A·z = b using a
        pseudoinverse method. Depending on the projection matrix Z, this may
        be the Moore–Penrose inverse (Z = I) or the Bott–Duffin inverse
        (Z ≠ I). If r > 1, the pseudoinverse solution is iteratively refined
        using an updated slack matrix Q to improve numerical stability and
        feasibility. Optional bounds on x and signs of the S-block slacks are
        enforced by bounded-variable least squares (Lawson–Hanson,
        Stark–Parker). All steps use A_k, A·Z with the singular values below
        the cutoff `rcond` set to zero.

        Stage 2 optionally refines the estimate using a convex optimization
        step centered around the pseudoinverse estimate ẑ. Regularization is
        governed by the user-specified parameter α:
            • α = 0      → Lasso (ℓ₁ norm),
            • α = 1      → Ridge (ℓ₂ norm),
            • 0 < α < 1  → Elastic Net (ℓ₁ and ℓ₂ combined).

        The class also supports diagnostic routines such as Monte Carlo
        hypothesis testing, row-wise deletion sensitivity analysis and the
        cross-validation of α over the M rows.

        Attributes
        ----------
        self.A : np.ndarray or None
            Canonical design matrix with block structure A = [C | S; M | Q].

        self.A_base : np.ndarray or None
            Canonical design matrix of the fit specification (without Q),
            reused by later solves.

        self.spec : dict or None
            Inputs of the fit specification (problem, C, S, M, m, p, i, j,
            zero_diagonal).

        self.C_idx : tuple
            Pair of integers defining the row and column ranges of the
            C block inside A. Used for matrix slicing and partitioning.

        self.S_idx : tuple
            Pair of integers defining the column range of the S block
            inside A.

        self.b : np.ndarray or None
            Right-hand side vector for the linear system A·z = b.

        self.Z : np.ndarray or None
            Projection matrix for Bott–Duffin inversion. Must be symmetric
            and idempotent. Defaults to identity for Moore–Penrose.

        self.rcond : float or bool
            Regularization parameter for the Moore-Penrose and Bott-Duffin
            inverses (relative cutoff on the singular values of A·Z).

        self.tolerance : float
            Convergence tolerance for NRMSE change between iterations.

        self.iteration_limit : int
            Maximum number of iterations allowed in the refinement loop.

        self.bounds : tuple, list or None
            Bounds on x as supplied: None, a (lo, hi) pair, or a sequence
            of (lo, hi) pairs (None = unbounded side).

        self.x_lb : np.ndarray or None
            Lower bounds on x (-inf if unbounded).

        self.x_ub : np.ndarray or None
            Upper bounds on x (inf if unbounded).

        self.slack_sign : bool
            Whether the S-block slack variables must be ≥ 0
            (S_ii = +1 encodes ≤, S_ii = -1 encodes ≥).

        self.scale : bool
            Whether the columns of A·Z are equilibrated in Steps 1 and 2
            (A·Z·D, D = diag(1/‖(A·Z)_j‖₂), u = D⁻¹·z).

        self.zhat : np.ndarray or None
            Unregularized pseudoinverse estimate of z from Step 1.

        self.zhat0 : np.ndarray or None
            Unbounded pseudoinverse estimate A_k⁺·b, the center of Step 2.
            Equals zhat without bounds.

        self.rank : int or None
            Effective rank of A·Z (the rank of A_k), of A·Z·D if scaled.

        self.sv_discarded : np.ndarray or None
            Singular values of A·Z (A·Z·D if scaled) below the cutoff
            (possibly empty).

        self.fixed : np.ndarray or None
            Mask of the components of z fixed at a bound in Step 1.

        self.fixed_at : np.ndarray or None
            Step 1 pass at which each component of z was fixed
            (0 = not fixed).

        self.final : bool
            Whether to run the second-stage convex refinement. If True,
            the estimate is corrected via convex programming.

        self.alpha : float or None
            Regularization parameter (1 if not given):
            - α = 0: Lasso (L1 norm),
            - α = 1: Ridge (L2 norm),
            - 0 < α < 1: Elastic Net.

        self.z : np.ndarray or None
            Final estimate of z after correction (if final is True).
            If skipped, z = zhat.

        self.rho : float or None
            Smallest residual norm ‖b - w*‖₂ within the numerical range
            of A under the bounds, w* = A_k·zhat (the target of Step 2).

        self.residual : float or None
            Residual norm ‖b - A·z‖₂ with the original A. Equals `rho`
            unless singular values were discarded or Step 2 fell back.

        self.status : dict or None
            Statuses of the last solve: "step1" (bounded Step 1), "rho"
            (w* fit), "final" (Step 2), "solver" (Step 2 solver used;
            "none (anchor feasible)" if the feasible anchor zhat0 was
            returned without a solver), "retry" (reason of a CLARABEL
            retry), "polish" (α = 1 solution polished on its active set;
            None if Step 2 is skipped), "tol_v" (verification tolerance),
            the achieved errors "kkt_error" (Step 1 KKT), "bound_error"
            (bounds) and "eq_error" (‖A_k·z - w*‖₂), and "scale" (column
            scaling).

        self.x : np.ndarray or None
            Variable component extracted from z, reshaped into m × p.

        self.y : np.ndarray or None
            Slack component of z representing inequality residuals.

        self.r : int
            Number of refinement iterations performed during Step 1.
            Iteration stops when NRMSE stabilizes or the limit is reached.

        self.r_max : int
            Requested number of refinement iterations, reused by `corr()`
            and `ttest(simulate=True)`.

        self.kappaC : float or None
            Condition number of the constraint block C (upper-left block
            in A).

        self.kappaB : float or None
            Condition number of the projected estimator B^(r) = A·pinv(C),
            calculated during refinement.

        self.kappaA : float or None
            Condition number of the full canonical matrix A^(r) after
            refinement step r.

        self.rmsa : float or None
            Total RMSA (Root Mean Square Adjustment) over all rows.

        self.rmsa_i : list of float
            Change in RMSA caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dkappaC : list of float
            Change in κ(C) caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dkappaB : list of float
            Change in κ(B) caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dkappaA : list of float
            Change in κ(A) caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dnrmse : list of float
            Change in NRMSE caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dzhat : list of float
            Change in ‖zhat‖_2 caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dz : list of float
            Change in ‖z‖_2 caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_dx : list of float
            Change in ‖x‖_F|2 caused by removing one row from [C | S]
            and re-estimating the CLSP solution.

        self.rmsa_failures : int
            Number of failed refits of `corr()` (Step 1 not converged or
            Step 2 fallback).

        self.r2_partial : float or None
            R² statistic computed over the M block (nan without M rows).
            Reflects partial goodness-of-fit in structured systems.

        self.nrmse : float or None
            Normalized RMSE between A·z and b, computed over full system.

        self.nrmse_partial : float or None
            NRMSE computed only over M block rows (nan without M rows).

        self.nrmse_ttest : list of float
            NRMSE samples generated via simulation under the null, used
            for empirical t-testing.

        self.nrmse_ttest_key : tuple or None
            Key (sample_size, seed, distribution, partial, simulate) of
            `nrmse_ttest`. Reset with all diagnostics by every solve.

        self.nrmse_ttest_failures : int
            Number of failed refits of `nrmse_ttest` (simulate=True).

        self.cv_alpha_result : dict or None
            Result of `cv_alpha()`. Reset by every solve.

        self.z_lower : np.ndarray or None
            Lower bound of the sensitivity band on z,
            z - |z|·κ(A)·‖b - A·z‖₂ / ‖b‖₂ (heuristic).

        self.z_upper : np.ndarray or None
            Upper bound of the sensitivity band on z. Symmetric to z_lower.

        self.x_lower : np.ndarray or None
            Lower bound of the sensitivity band on x (see z_lower).

        self.x_upper : np.ndarray or None
            Upper bound of the sensitivity band on x. Symmetric to x_lower.

        self.y_lower : np.ndarray or None
            Lower bound of the sensitivity band on y (see z_lower).

        self.y_upper : np.ndarray or None
            Upper bound of the sensitivity band on y. Symmetric to y_lower.

        self.seed : int
            Random seed used for reproducible Monte Carlo diagnostics.

        self.rng : np.random.Generator
            Random number generator initialized using `seed`.

        self.distribution : callable
            Function that generates random samples for simulation.
            Must accept a single integer argument `n`.

        Methods
        -------
        self.error : method
            Raise a CLSPError with optional diagnostics or contextual details.
        
        self.canonize : method
            Construct matrix A = [C | S; M | Q] and define its partitions.
        
        self.solve : method
            Main estimation routine performing pseudoinverse and convex steps.
        
        self.corr : method
            Compute RMSA and condition diagnostics via leave-one-out row
            deletion on [C | S]. Populates all `self.rmsa_*` lists.
        
        self.ttest : method
            Monte Carlo t-test for NRMSE using user-defined distribution.
            Returns one-sided and two-sided p-values.

        self.cv_alpha : method
            K-fold cross-validation of α over the M rows. Returns the mean and
            sd of the prediction RMSE for each α and the best α.
        """
        # Variables
        self.A              : np.ndarray | None = None  # design, [C|S;M|Q]
        self.A_base         : np.ndarray | None = None  # design without Q
        self.spec           : dict       | None = None  # fit specification
        self.C_idx          : tuple[int  | None,  int  | None]                 \
                            = (None, None)              # indices of the C block
        self.S_idx          : tuple[int  | None,  int  | None]                 \
                            = (None, None)              # columns of the S block
        self.b              : np.ndarray | None = None  # right-hand side
        self.Z              : np.ndarray | None = None  # B-D subspace matrix
        self.rcond          : float      | bool = False # regularization
        self.tolerance      : float = float(np.sqrt(np.finfo(float).eps))
        self.iteration_limit: int               = 50    # limit  of iterations
        self.bounds         : tuple | list | None                              \
                            = None                      # bounds on x
        self.x_lb           : np.ndarray | None = None  # lower bounds on x
        self.x_ub           : np.ndarray | None = None  # upper bounds on x
        self.slack_sign     : bool              = False # S slacks ≥ 0
        self.scale          : bool              = False # column scaling
        self.r              : int               = 0     # number of iterations
        self.r_max          : int               = 1     # requested iterations
        self.zhat           : np.ndarray | None = None  # first-step estimate
        self.zhat0          : np.ndarray | None = None  # unbounded estimate
        self.rank           : int        | None = None  # effective rank of A·Z
        self.sv_discarded   : np.ndarray | None = None  # discarded sing. values
        self.fixed          : np.ndarray | None = None  # fixed components of z
        self.fixed_at       : np.ndarray | None = None  # pass of fixing
        self.final          : bool              = True  # inclusion of Step 2
        self.alpha          : float      | None = None  # regularization (α)
        self.z              : np.ndarray | None = None  # final solution
        self.rho            : float      | None = None  # min. residual norm
        self.residual       : float      | None = None  # residual norm of z
        self.status         : dict       | None = None  # solver statuses
        self.x              : np.ndarray | None = None  # variable part of z
        self.y              : np.ndarray | None = None  # slack part of z
        self.kappaC         : float      | None = None  # spectral κ(C_canon)
        self.kappaB         : float      | None = None  # spectral κ(B^(r))
        self.kappaA         : float      | None = None  # spectral κ(A^(r))
        self.rmsa           : float      | None = None  # total RMSA
        self.rmsa_i         : list[float]       = []    # list of row RMSA
        self.rmsa_dkappaC   : list[float]       = []    # list of Δκ(C)
        self.rmsa_dkappaB   : list[float]       = []    # list of Δκ(B)
        self.rmsa_dkappaA   : list[float]       = []    # list of Δκ(A)
        self.rmsa_dnrmse    : list[float]       = []    # list of ΔNRMSE
        self.rmsa_dzhat     : list[float]       = []    # list of Δ‖zhat‖_2
        self.rmsa_dz        : list[float]       = []    # list of Δ‖z‖_2
        self.rmsa_dx        : list[float]       = []    # list of Δ‖x‖_F|2
        self.rmsa_failures  : int               = 0     # failed refits
        self.r2_partial     : float      | None = None  # R^2   for the M block
        self.nrmse          : float      | None = None  # NRMSE for A
        self.nrmse_partial  : float      | None = None  # NRMSE for the M block
        self.nrmse_ttest    : list[float]       = []    # list of NRMSE
        self.nrmse_ttest_key: tuple      | None = None  # key of nrmse_ttest
        self.nrmse_ttest_failures: int          = 0     # failed refits
        self.cv_alpha_result: dict       | None = None  # result of cv_alpha()
        self.z_lower        : np.ndarray | None = None  # lower band
        self.z_upper        : np.ndarray | None = None  # upper band
        self.x_lower        : np.ndarray | None = None  # lower band
        self.x_upper        : np.ndarray | None = None  # upper band
        self.y_lower        : np.ndarray | None = None  # lower band
        self.y_upper        : np.ndarray | None = None  # upper band
        self.seed           : int               = 123456789  # Monte Carlo
        self.rng            = np.random.default_rng(self.seed)
        self.distribution   = lambda n: self.rng.normal(loc=0, scale=1, size=n)
        # Methods
        self.error          = CLSPError
        self.canonize       = MethodType(CLSPCanonicalForm, self)
        self.solve          = MethodType(CLSPSolve,         self)
        self.corr           = MethodType(CLSPCorrelogram,   self)
        self.ttest          = MethodType(CLSPTTest,         self)
        self.cv_alpha       = MethodType(CLSPCVAlpha,       self)
        self.summarize      = MethodType(CLSPSummary,       self)
        self.summary        = MethodType(CLSPSummary,       self)

    def __repr__(self):
        """
        Return a formatted string representation of the CLSP object
        """
        if self.z is None:
            return "<CLSP: z=None, x=None, y=None>"

        z_str = (np.array2string(self.z, max_line_width=80))
        x_str = (np.array2string(self.x, max_line_width=80)
                 if self.x is not None else "None")
        y_str = (np.array2string(self.y, max_line_width=80)
                 if self.y is not None else "None")

        return f"<CLSP:\n z={z_str},\n x={x_str},\n y={y_str}>"
