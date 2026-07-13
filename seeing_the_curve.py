import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from openbb import obb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
matplotlib.use("Qt5Agg")



START_DATE = "2019-01-01"
END_DATE   = "2025-12-31"
# OpenBB column name → tenor label
TENOR_MAP = {
    "year_2":  "2Y",
    "year_5":  "5Y",
    "year_7":  "7Y",
    "year_10": "10Y",
    "year_20": "20Y",
    "year_30": "30Y",
}

# Reasonable yield bounds for sanity check (%)
YIELD_FLOOR   = -0.5
YIELD_CEILING =  7.0

#Task 2 regime split date for sub-period PCA comparison
REGIME_SPLIT = "2022-03-16" #First day of Fed's 2022 tightening cycle (25 bps hike), an inflection point for yield behavior.

# Modified durations from Duration Table.
DURATIONS = {
    "2Y":  1.91,
    "5Y":  4.53,
    "7Y":  6.11,
    "10Y": 8.24,
    "20Y": 13.55,
    "30Y": 17.05,
}
# Butterfly tenors — chosen from full-sample PC3 loadings:
# 2Y  (+0.615) and 30Y (+0.403) are the two strongest positive loadings (wings)
# 7Y  (-0.434) is the deepest negative loading (belly)
# Trade-off vs 2s10s30s: 7Y is less liquid than 10Y, but its PC3 loading is
# ~2x larger — we accept the liquidity cost for a cleaner curvature signal.
BUTTERFLY_TENORS = ["2Y", "7Y", "30Y"]
 
# Belly notional (in $mm). 
BELLY_NOTIONAL = 100.0

# Task 4 Estimation/validation split - in-sample 2019-2023, out-of-sample 2024–2025
IS_END       = "2023-12-31"
OOS_START    = "2024-01-01"
# Strategy parameters
LOOKBACK = 60       # 60 days, ~1 quarter, several estimated half-lives
ENTRY_Z  = 2      # meaningful dislocation, clears costs, adequate trade count
EXIT_Z   = 0.5     # capture fast portion of OU decay; 1.0-sigma hysteresis
COST_BPS = 0.2     # per-leg bid/ask on on-the-run Treasuries


"""
Task 1 — Data Retrieval & Sanity Check
=======================================
Pulls daily U.S. Treasury constant-maturity yields via OpenBB
and applies a documented cleaning pipeline.

Usage:
    # Default: drop rows flagged by checks for non-numeric, NaN, weekend, and conflicting duplicates
    tcd = TreasuryCurveData(drop_non_numeric=True, drop_nan=True,
                            drop_non_business=True, drop_conflicting_duplicates=True)
    yields  = tcd.get_clean_yields()
    changes = tcd.get_yield_changes()
"""


class TreasuryCurveData:
    """Fetches, cleans, and exposes U.S. Treasury yield curve data."""

    def __init__(
        self,
        drop_non_numeric:             bool = True,
        drop_nan:                     bool = True,
        drop_non_business:            bool = True,
        drop_conflicting_duplicates:  bool = True,
    ):
        """
        Parameters
        ----------
        drop_non_numeric            : Remove rows containing non-numeric values.
        drop_nan                    : Remove rows containing NaN values.
        drop_non_business           : Remove weekend / non-business-day rows.
        drop_conflicting_duplicates : Remove ALL copies of dates with conflicting data. Identical duplicates are always silently deduped.
        """
        self.drop_non_numeric            = drop_non_numeric
        self.drop_nan                    = drop_nan
        self.drop_non_business           = drop_non_business
        self.drop_conflicting_duplicates = drop_conflicting_duplicates

        self._clean   = None 
        self._changes = None

    # ── 1. Fetch ─────────────────────────────────────────────────────────────

    def _fetch_raw(self) -> pd.DataFrame:
        """
        Pull daily Treasury rates from OpenBB (Federal Reserve H.15).

        OpenBB call:
            obb.fixedincome.government.treasury_rates(
                start_date, end_date, provider="federal_reserve"
            )
        Returns columns: year_2, year_5, year_7, year_10, year_20, year_30.
        """
        result = obb.fixedincome.government.treasury_rates(
            start_date=START_DATE,
            end_date=END_DATE,
            provider="federal_reserve",
            country = "united_states"
        )
        df = result.to_dataframe()
        # Select and rename our six tenors
        df = df[list(TENOR_MAP.keys())].rename(columns=TENOR_MAP)
        ## Ensure DatetimeIndex
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    # ── 2. Inspect ───────────────────────────────────────────────────────────

    def _inspect(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """
        Run all checks on the raw DataFrame without modifying it.
        Returns the coerced DataFrame (floats, original structure intact)
        and a findings dict with one key per check.

        Separation of detection from removal means:
        - Check 1 runs on raw strings, so it catches non-numeric values
          before they silently become NaN.
        - Check 2 then runs on the float-coerced frame; any NaN it finds
          is a genuine missing value, not a carry-over from check 1.
        """
        tenors   = list(TENOR_MAP.values())
        findings = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

        # ── Check 1: non-numeric values ──────────────────────────────────────
        # A cell is non-numeric if coercion to float produces NaN but the
        # original value is not already NaN (i.e. it had a bad string value).
        non_numeric_dates = set()
        for col in tenors:
            mask = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            for d in df.index[mask]:
                findings[1].append(
                    f"{col} on {d.strftime('%Y-%m-%d')}: '{df.loc[d, col]}'"
                )
                non_numeric_dates.add(d)

        # Coerce to float for all subsequent checks; non-numeric become NaN here
        df = df.copy()
        df[tenors] = df[tenors].apply(pd.to_numeric, errors="coerce")

        # ── Check 2: NaN values ──────────────────────────────────────────────
        # Only flag dates not already caught by check 1 (those NaNs were
        # non-numeric values, not genuine missing data — already reported above).
        nan_dates = set()
        for col in tenors:
            mask = df[col].isna()
            for d in df.index[mask]:
                if d not in non_numeric_dates:
                    findings[2].append(f"{col} on {d.strftime('%Y-%m-%d')}")
                    nan_dates.add(d)

        # ── Check 3: missing business dates ──────────────────────────────────
        # pd.bdate_range covers Mon–Fri; US holidays still appear as "missing"
        # but are legitimate absences — flagged for human review, not auto-removed.
        expected      = pd.bdate_range(start=START_DATE, end=END_DATE)
        missing_dates = expected.difference(df.index).strftime("%Y-%m-%d").tolist()
        if missing_dates:
            findings[3] = missing_dates   # full list; brief truncates for display

        # ── Check 4: non-business-day rows ───────────────────────────────────
        weekend_mask = df.index.dayofweek >= 5
        findings[4]  = df.index[weekend_mask].strftime("%Y-%m-%d").tolist()

        # ── Check 5: duplicate dates ─────────────────────────────────────────
        dup_mask = df.index.duplicated(keep=False)
        identical_dupes    = []
        conflicting_dupes  = []
        if dup_mask.any():
            for d in df.index[dup_mask].unique():
                rows = df.loc[d]
                if rows.duplicated(keep=False).all():
                    identical_dupes.append(d.strftime("%Y-%m-%d"))
                else:
                    conflicting_dupes.append(
                        f"{d.strftime('%Y-%m-%d')}:\n{rows.to_string()}"
                    )
        findings[5] = {"identical": identical_dupes, "conflicting": conflicting_dupes}

        # ── Check 6: sanity check — values outside reasonable range ──────────
        # Warning only — no removal option. Flags yields outside
        # [YIELD_FLOOR, YIELD_CEILING] and single-day moves > 50 bps.
        range_warnings = []
        for col in tenors:
            mask = (df[col] < YIELD_FLOOR) | (df[col] > YIELD_CEILING)
            for d in df.index[mask]:
                range_warnings.append(
                    f"{col} on {d.strftime('%Y-%m-%d')}: {df.loc[d, col]:.3f}% "
                    f"(outside [{YIELD_FLOOR}, {YIELD_CEILING}]%)"
                )
        jump_warnings = []
        for col in tenors:
            jumps = df[col].diff().abs()
            mask  = jumps > 0.50
            for d in df.index[mask]:
                jump_warnings.append(
                    f"{col} on {d.strftime('%Y-%m-%d')}: "
                    f"{df[col].diff().loc[d]:+.3f}% single-day move"
                )
        findings[6] = {"range": range_warnings, "jumps": jump_warnings}

        return df, findings

    # ── 3. Apply removals ────────────────────────────────────────────────────

    def _apply_removals(
        self, df: pd.DataFrame, findings: dict
    ) -> tuple[pd.DataFrame, dict]:
        """
        Apply user-configured removals based on inspection findings.
        Returns the cleaned DataFrame and a removals summary dict.
        """
        tenors   = list(TENOR_MAP.values())
        removed  = {1: [], 2: [], 4: [], 5: []}

        # Check 1 — non-numeric rows
        if self.drop_non_numeric and findings[1]:
            bad_dates = set(
                pd.to_datetime(entry.split(" on ")[1].split(":")[0])
                for entry in findings[1]
            )
            removed[1] = sorted(d.strftime("%Y-%m-%d") for d in bad_dates)
            df = df[~df.index.isin(bad_dates)]

        # Check 2 — NaN rows (only genuine NaNs, not the coerced non-numerics)
        if self.drop_nan and findings[2]:
            nan_dates = set(
                pd.to_datetime(entry.split(" on ")[1])
                for entry in findings[2]
            )
            removed[2] = sorted(d.strftime("%Y-%m-%d") for d in nan_dates)
            df = df[~df.index.isin(nan_dates)]

        # Check 4 — non-business-day rows
        if self.drop_non_business and findings[4]:
            removed[4] = findings[4]
            df = df[df.index.dayofweek < 5]

        # Check 5 — duplicates
        # Identical duplicates: always deduplicate silently (keep first)
        if findings[5]["identical"]:
            df = df[~df.index.duplicated(keep="first")]
        # Conflicting duplicates: remove all copies if option is set
        if self.drop_conflicting_duplicates and findings[5]["conflicting"]:
            conflict_dates = [
                pd.to_datetime(entry.split(":\n")[0])
                for entry in findings[5]["conflicting"]
            ]
            removed[5] = [d.strftime("%Y-%m-%d") for d in conflict_dates]
            df = df[~df.index.isin(conflict_dates)]

        df = df.sort_index()
        return df, removed

    # ── 4. Print brief ───────────────────────────────────────────────────────

    def _print_brief(self, findings: dict, removed: dict) -> None:
        """Print the full data issue brief with findings and removal decisions."""

        def _fmt_action(check_key: int, items) -> str: #formatting
            n       = len(items)
            dropped = len(removed.get(check_key, []))
            kept    = n - dropped
            if dropped == n:
                return f"→ all {n} row(s) removed"
            elif dropped > 0:
                return f"→ {dropped} row(s) removed, {kept} kept"
            else:
                return f"→ rows kept (drop_{['non_numeric','nan',None,'non_business','conflicting_duplicates'][check_key-1]}=False)"

        print("\n══ Data Issue Brief ════════════════════════════════")

        # Check 1
        if findings[1]:
            print(f"\n[CHECK 1] Non-numeric values — {len(findings[1])} cell(s):")
            for item in findings[1]:
                print(f"  • {item}")
            print(f"  {_fmt_action(1, findings[1])}")
        else:
            print("\n[CHECK 1] Non-numeric values: none ✓")

        # Check 2
        if findings[2]:
            print(f"\n[CHECK 2] NaN values — {len(findings[2])} cell(s):")
            for item in findings[2]:
                print(f"  • {item}")
            print(f"  {_fmt_action(2, findings[2])}")
        else:
            print("\n[CHECK 2] NaN values: none ✓")

        # Check 3 — info only, no removal
        if findings[3]:
            sample = findings[3][:5]
            more   = f" … and {len(findings[3]) - 5} more" if len(findings[3]) > 5 else ""
            print(f"\n[CHECK 3] Missing business dates — {len(findings[3])} day(s) "
                  f"(likely US holidays, no action taken):")
            print(f"  • {', '.join(sample)}{more}")
        else:
            print("\n[CHECK 3] Missing business dates: none ✓")

        # Check 4
        if findings[4]:
            print(f"\n[CHECK 4] Non-business-day rows — {len(findings[4])} row(s):")
            for item in findings[4]:
                print(f"  • {item}")
            print(f"  {_fmt_action(4, findings[4])}")
        else:
            print("\n[CHECK 4] Non-business-day rows: none ✓")

        # Check 5
        n_identical    = len(findings[5]["identical"])
        n_conflicting  = len(findings[5]["conflicting"])
        if n_identical or n_conflicting:
            print(f"\n[CHECK 5] Duplicate dates:")
            if n_identical:
                print(f"  • {n_identical} identical duplicate(s) — silently deduped ✓")
            if n_conflicting:
                print(f"  • {n_conflicting} conflicting duplicate(s):")
                for item in findings[5]["conflicting"]:
                    print(f"    {item}")
                action = (f"→ all copies removed"
                          if self.drop_conflicting_duplicates
                          else "→ kept (drop_conflicting_duplicates=False)")
                print(f"  {action}")
        else:
            print("\n[CHECK 5] Duplicate dates: none ✓")

        # Check 6 — warnings only, no removal
        r_warns = findings[6]["range"]
        j_warns = findings[6]["jumps"]
        if r_warns or j_warns:
            print(f"\n[CHECK 6] Sanity warnings (informational only — no rows removed):")
            for w in r_warns:
                print(f"  ⚠  Out-of-range: {w}")
            for w in j_warns:
                print(f"  ⚠  Large move:   {w}")
        else:
            print("\n[CHECK 6] Sanity checks: all values in range, no large moves ✓")

        print("\n══════════════════════════════════════════════════\n")

    def get_clean_yields(self) -> pd.DataFrame:
        """Return cleaned yield levels (%). Fetches and cleans on first call."""
        if self._clean is None:
            raw              = self._fetch_raw()
            coerced, findings = self._inspect(raw)
            cleaned, removed  = self._apply_removals(coerced, findings)
            self._print_brief(findings, removed)
            self._clean       = cleaned
        return self._clean.copy()

    def get_yield_changes(self) -> pd.DataFrame:
        """Return daily first differences of yields (%)."""
        if self._changes is None:
            clean           = self.get_clean_yields()
            self._changes   = clean.diff().iloc[1:].dropna()
        return self._changes.copy()

    # ── Plots ────────────────────────────────────────────────────────────────

    def plot_yields(self) -> None:
        df = self.get_clean_yields() * 100
        plt.figure(figsize=(18, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], label=col, linewidth=0.8)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.title("U.S. Treasury Constant-Maturity Yields (2019–2025)")
        plt.ylabel("Yield (%)") 
        plt.xlabel("Date")
        plt.legend() 
        plt.grid(alpha=0.3)
        plt.tight_layout()

    def plot_yield_changes(self) -> None:
        df = self.get_yield_changes() * 10000 
        plt.figure(figsize=(18, 6))
        for col in df.columns:
            plt.plot(df.index, df[col], label=col, linewidth=0.5, alpha=0.7)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MonthLocator())
        ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.title("Daily Yield Changes (First Differences)")
        plt.ylabel("ΔYield (%)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

class TreasuryPCA:
    """
    Runs PCA on a DataFrame of daily yield changes and exposes the results.
    One instance = one PCA fit (full sample, or a single sub-period).
    """

    def __init__(self, n_components: int = 3, standardize: bool = False):
        """
        Parameters:
        n_components : number of PCs to retain (default 3 — level/slope/curvature).
        standardize  : Default False. If True, standardize each tenor to unit variance before PCA            
        """
        self.n_components = n_components
        self.standardize  = standardize

        # Populated by fit()
        self.tenors_             = None   # list of column names
        self.means_              = None   # per-tenor mean (used to demean)
        self.stds_               = None   # per-tenor std (only if standardize=True)
        self.loadings_           = None   # DataFrame (tenors × PCs)
        self.eigenvalues_        = None   # np.ndarray, length n_components
        self.explained_variance_ratio_ = None  # np.ndarray, length n_components
        self.scores_             = None   # DataFrame (dates × PCs)
        self.window_             = None   # (start_date, end_date) tuple, for labelling

    # ── Fit ──────────────────────────────────────────────────────────────────

    def fit(self, changes: pd.DataFrame) -> "TreasuryPCA":
        """
        Fit PCA on a DataFrame of daily yield changes.
        Parameters
        changes : DataFrame indexed by date, columns are tenors, values are daily yield changes 
                (in whatever unit was passed — units cancel in PCA but stay attached to loadings and scores).
        """
        self.tenors_ = list(changes.columns)
        X = changes.values.astype(float)

        if self.standardize:
            scaler      = StandardScaler()
            X_input     = scaler.fit_transform(X)
            self.means_ = scaler.mean_
            self.stds_  = scaler.scale_
        else:
            # Demean only — covariance-matrix PCA
            self.means_ = X.mean(axis=0)
            X_input     = X - self.means_

        pca = PCA(n_components=self.n_components)
        scores = pca.fit_transform(X_input)

        # sklearn's components_ has shape (n_components × n_tenors); we transpose so rows = tenors, cols = PCs — reads naturally
        self.loadings_ = pd.DataFrame(
            pca.components_.T,
            index=self.tenors_,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
        )
        self.eigenvalues_              = pca.explained_variance_
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self.scores_ = pd.DataFrame(
            scores,
            index=changes.index,
            columns=[f"PC{i+1}" for i in range(self.n_components)],
        )
        self.window_ = (changes.index.min(), changes.index.max())
        return self

    # ── Sign alignment ───────────────────────────────────────────────────────

    def align_signs(self, reference: "TreasuryPCA") -> "TreasuryPCA":
        """
        Flip any PC whose dot product with the reference's corresponding PC
        is negative. Eigenvectors are sign-ambiguous by construction; this
        makes across-window loading comparisons meaningful.

        Modifies `loadings_` and `scores_` in place; returns self.
        """
        if self.loadings_ is None:
            raise RuntimeError("Call fit() before align_signs_to().")
        if reference.loadings_ is None:
            raise RuntimeError("Reference PCA has not been fit.")

        for pc in self.loadings_.columns:
            dot = np.dot(self.loadings_[pc].values, reference.loadings_[pc].values)
            if dot < 0:
                self.loadings_[pc] = -self.loadings_[pc]
                self.scores_[pc]   = -self.scores_[pc]
        return self

    # ── Kaiser check (correlation-PCA only) ──────────────────────────────────

    def kaiser_check(self) -> pd.DataFrame | None:
        """
        For standardized PCA only, flag PCs with eigenvalue > 1.
        Returns None if the fit was not standardized.
        Returns a DataFrame with columns:
        - eigenvalue : the eigenvalue of each PC
        - keep_by_kaiser : True if eigenvalue > 1, else False
        """
        if not self.standardize:
            print("Kaiser check skipped: rule only applies to standardized PCA.")
            return None
        return pd.DataFrame({
            "eigenvalue":     self.eigenvalues_,
            "keep_by_kaiser": self.eigenvalues_ > 1.0,
        }, index=self.loadings_.columns)

    # ── Scree plot ───────────────────────────────────────────────────────────

    def scree_plot(self, title_suffix: str = "") -> None:
        """Bar chart of individual variance shares with cumulative line overlay."""
        shares     = self.explained_variance_ratio_ * 100
        cumulative = shares.cumsum()
        idx        = np.arange(1, len(shares) + 1) # 1-based PC index for x-axis

        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(idx, shares, alpha=0.7, label="Individual")
        ax1.set_xlabel("Principal component")
        ax1.set_ylabel("Variance explained (%)")
        ax1.set_xticks(idx)

        ax2 = ax1.twinx()
        ax2.plot(idx, cumulative, "o-", color="firebrick", label="Cumulative")
        ax2.set_ylabel("Cumulative variance (%)")
        ax2.set_ylim(0, 105)

        title = "Scree plot"
        if title_suffix:
            title += f" — {title_suffix}"
        plt.title(title)
        fig.tight_layout()

    # ── Summary ──────────────────────────────────────────────────────────────

    def summary(self, label: str = "") -> pd.DataFrame:
        """One-row-per-PC summary DataFrame for printing / comparison."""
        rows = []
        for i, pc in enumerate(self.loadings_.columns):
            rows.append({
                "PC":              pc,
                "eigenvalue":      self.eigenvalues_[i],
                "var_share_%":     self.explained_variance_ratio_[i] * 100,
                "cum_var_%":       self.explained_variance_ratio_[:i+1].sum() * 100,
            })
        df = pd.DataFrame(rows).set_index("PC")
        if label: # add a top-level column label for multi-indexing when comparing multiple PCAs 
            df.columns = pd.MultiIndex.from_product([[label], df.columns])
        return df.round(3)

"""
Task 3 — Factor-neutral butterfly.

The class represents a three-leg butterfly on a chosen set of tenors and
solves the leg notionals two ways:
  1. Factor-neutral: zero exposure to PC1 and PC2 (uses PCA loadings).
  2. DV01-neutral:   equal-DV01 wings, no factor information used.

It also produces a residual-exposure table quantifying the unintended
factor exposure the naive DV01-neutral trade carries.
"""
class Butterfly:
    """Three-leg butterfly on chosen tenors, with two weight solvers."""

    def __init__(
        self,
        tenors:     list[str],
        durations:  dict[str, float],
        pca:        TreasuryPCA,
    ):
        """
        Parameters
        ----------
        tenors    : three tenor labels [wing_short, belly, wing_long],
                    e.g. ["2Y", "7Y", "30Y"]. Order matters: middle = belly.
        durations : modified duration per tenor, e.g. {"2Y": 1.91, ...}.
        pca       : a fitted TreasuryPCA whose loadings will drive the factor-neutral solve. Must contain rows for `tenors`.
        """
        if len(tenors) != 3:
            raise ValueError("A butterfly needs exactly three tenors.")
        missing = [t for t in tenors if t not in durations]
        if missing:
            raise ValueError(f"Missing durations for {missing}.")
        if pca.loadings_ is None:
            raise ValueError("PCA must be fit before constructing Butterfly.")

        self.tenors     = tenors
        self.durations  = np.array([durations[t] for t in tenors])
        self.loadings   = pca.loadings_.loc[tenors]   # 3 tenors × 3 PCs

    # ── DV01 table ───────────────────────────────────────────────────────────

    def dv01_table(self, notional: float = 100.0) -> pd.DataFrame:
        """
        DV01 per given notional (default $100mm), assuming par pricing.
        At par:  DV01 = ModDuration × Notional × 0.0001
                      = ModDuration × $10,000  (for notional = $100mm)
        """
        dv01 = self.durations * notional * 1e6 * 1e-4   # dollars per bp
        return pd.DataFrame({
            "modified_duration": self.durations,
            f"DV01_per_${notional:.0f}mm ($/bp)": dv01,
        }, index=self.tenors)

    # ── Factor exposures for any set of notionals ──────────────────────

    def _factor_exposures(self, notionals: np.ndarray) -> np.ndarray:
        """
        Portfolio's dollar P&L per 1-unit move of each PC.
        E_k = Σ_i n_i · dv01_i · L_ik
        Returns a length-n_PCs vector.
        """
        dv01_per_1mm = self.durations * 1e6 * 1e-4       # $/bp per $1mm notional
        dv01_dollars = notionals * dv01_per_1mm    # $/bp per leg (notionals in $mm) 
        return dv01_dollars @ self.loadings.values        # matrix multiply: 1×3 @ 3×n_PCs → 1×n_PCs

    # ── Solver 1: factor-neutral ─────────────────────────────────────────────
    def solve_factor_neutral(self) -> pd.Series:
        """
        Solve for weights so the trade is neutral to PC1 and PC2.
        Belly weight fixed at 1 (normalization).
        Returns a Series of weights indexed by tenor.
        """
        dv01_per_1mm = self.durations * 1e6 * 1e-4    # $/bp per $1mm at each tenor

        # 3x3 system Aw = b, w = [w_2Y, w_7Y, w_30Y]
        # Row 0: PC1 exposure = 0 → sum_i w_i * dv01_i * L[i, PC1] = 0
        # Row 1: PC2 exposure = 0 → sum_i w_i * dv01_i * L[i, PC2] = 0
        # Row 2: belly weight = 1 → w_belly = 1
        A = np.zeros((3, 3))
        A[0, :] = dv01_per_1mm * self.loadings["PC1"].values
        A[1, :] = dv01_per_1mm * self.loadings["PC2"].values
        A[2, 1] = 1.0          # belly is index 1
        b = np.array([0.0, 0.0, 1.0])
 
        weights = np.linalg.solve(A, b)
        return pd.Series(weights, index=self.tenors)

    # ── Solver 2: naive DV01-neutral (equal-DV01 wings) ──────────────────────
    def solve_dv01_neutral(self) -> pd.Series:
        """
        Naive fly: each wing carries half the belly's DV01.
        No factor information used — assumes yields move in parallel.
        Belly weight fixed at 1 (same normalization as factor-neutral).
        Returns a Series of weights indexed by tenor.
        """
        dv01_per_1mm  = self.durations * 1e6 * 1e-4
        # Each wing's DV01 = half the belly's DV01
        # wing_weight x dv01_wing = 0.5 x |belly_weight| x dv01_belly
        # => wing_weight = 0.5 x dv01_belly / dv01_wing  (negative: short wings)
        w_short = -0.5 * dv01_per_1mm[1] / dv01_per_1mm[0]
        w_long  = -0.5 * dv01_per_1mm[1] / dv01_per_1mm[2]
 
        weights = np.array([w_short, 1, w_long])
        return pd.Series(weights, index=self.tenors)


    # ── Residual exposure comparison ─────────────────────────────────────────

    def exposure_table(self, belly_mm: float) -> pd.DataFrame:
        """
        2-row x 3-PC table of dollar factor exposures, plus PC1/PC2 as
        a % of PC3 to quantify the naive fly's unintended factor bets.
        """
        w_fn = self.solve_factor_neutral()
        w_dv = self.solve_dv01_neutral()
        belly_mm = 100.0   # or whatever the user passes in
        E_fn = self._factor_exposures(w_fn.values * belly_mm)
        E_dv = self._factor_exposures(w_dv.values * belly_mm)
        pcs = list(self.loadings.columns)
        df  = pd.DataFrame(
            [E_fn, E_dv],
            index=["Factor-neutral", "DV01-neutral"],
            columns=[f"{pc}_exp ($/unit)" for pc in pcs],
        )
        cols = ["PC1_exp ($/unit)", "PC2_exp ($/unit)", "PC3_exp ($/unit)"]
        df = df.loc[:, cols]  # keep only PC1/PC2/PC3 columns
        # Ratio: unintended PC1/PC2 exposure as % of intended PC3 exposure
        df["PC1_as_%_of_PC3"] = 100 * df.iloc[:, 0] / df.iloc[:, 2]
        df["PC2_as_%_of_PC3"] = 100 * df.iloc[:, 1] / df.iloc[:, 2]
        return df.round(2)

"""
Task 4 — Mean-reversion strategy on the factor-neutral butterfly.
Signal   : cumulative PC3 score (with butterfly yield spread as sanity twin)
Rule     : enter when |trailing z-score| > entry_z, exit when |z| < exit_z
P&L      : DV01 engine — position x sum_i(-w_i x dv01_i x dy_i) - costs
"""


class MeanReversionStrategy:
    """
    Frozen mean-reversion strategy. All estimation inputs (PCA loadings,
    butterfly weights, thresholds) are fixed at construction time — the
    same object is applied to both IS and OOS windows via `run(period)`.
    """

    def __init__(
        self,
        yields_bps:      pd.DataFrame,   # yield in bps
        pca:             TreasuryPCA,    # fitted on in-sample window
        butterfly:       Butterfly,      # built from same in-sample PCA
        lookback:        int   = 60,     # z-score trailing window (days)
        entry_z:         float = 1.5,    # |z| threshold to enter
        exit_z:          float = 0.5,    # |z| threshold to exit
        cost_bps:        float = 0.5,    # bid/ask per leg per trade (bps)
        belly_mm:        float = 100.0,  # belly notional in $mm
    ):
        self.yields_bps = yields_bps
        self.pca        = pca
        self.butterfly  = butterfly
        self.lookback   = lookback
        self.entry_z    = entry_z
        self.exit_z     = exit_z
        self.cost_bps   = cost_bps
        self.belly_mm   = belly_mm

        # Frozen weights (Series indexed by tenor) — never re-solved
        self.weights = butterfly.solve_factor_neutral()

        # Per-leg DV01 in $/bp at self.belly_mm notional
        # signed_dv01: carries leg direction (belly +, wings -) — used for P&L; = weight x belly_mm x duration x 1e6 x 1e-4
        # _dv01_per_leg: magnitude only — used for cost calculation
        self._signed_dv01  = (
            self.weights.values
            * self.belly_mm
            * butterfly.durations
            * 1e6 * 1e-4
        )
        self._dv01_per_leg = np.abs(self._signed_dv01)
 

        # Cached signal, instead of recomputing the entire signal from scratch every call.
        self._signal = None

    # ── Signal construction ─────────────────────────────────────────────────
    def build_signal(self) -> pd.DataFrame:
        """
        Build the FULL-history signal DataFrame with columns:
          s3_daily   : daily PC3 score (from PCA projection)
          S          : cumulative PC3 score — the mean-reverting level
          spread_bps : butterfly yield spread (sanity twin of S)
          z          : trailing z-score of S with self.lookback window,
                       shifted by 1 day (no lookahead: z_t uses S up to t-1)
        """
        if self._signal is not None:
            return self._signal.copy()

        tenors = self.butterfly.tenors
        loads  = self.pca.loadings_.loc[tenors, "PC3"].values

        # Daily PC3 score using butterfly tenors only.
        # (Same sign convention as pca.scores_; using only butterfly tenors keeps
        #  the score directly comparable to the yield spread built on same tenors.)
        dy = self.yields_bps[tenors].diff()
        s3_daily = dy.mul(loads, axis=1).sum(axis=1)

        # Cumulative — the level we test for mean reversion and trade on
        S = s3_daily.cumsum()

        # Butterfly yield spread (sanity twin): weighted sum of yield levels
        # Uses signed weights (belly is negative)
        spread_bps = self.yields_bps[tenors].mul(self.weights.values, axis=1).sum(axis=1)

        # Trailing z-score with strict no-lookahead:
        # z_t = (S_{t-1} - trailing_mean_ending_at_{t-1}) / trailing_std_ending_at_{t-1}
        S_lag  = S.shift(1)
        mean_  = S_lag.rolling(self.lookback).mean()
        std_   = S_lag.rolling(self.lookback).std()
        z      = (S_lag - mean_) / std_

        self._signal = pd.DataFrame({
            "s3_daily":   s3_daily,
            "S":          S,
            "spread_bps": spread_bps,
            "z":          z,
        })
        return self._signal.copy()

    # ── Mean-reversion diagnostics ──────────────────────────────────────────
    def test_mean_reversion(self, period: slice) -> pd.Series:
        """
        On the given window, run ADF on S and fit AR(1) to estimate half-life.

        Returns
        -------
        dict with:
          adf_stat, adf_pvalue     : Dickey-Fuller test (H0: unit root)
          ar1_beta                 : coefficient in dS_t = alpha + beta*S_{t-1} + e
          half_life_days           : ln(2) / ln(1/phi) where phi = 1 + beta
        """
        S = self.build_signal()["S"].loc[period].dropna()

        # ADF: statsmodels' regression form is dS_t = alpha + beta*S_{t-1} + ...
        # Using regression='c' (constant, no trend) matches an Ornstein-Uhlenbeck process with a mean
        adf_stat, adf_p, *_ = adfuller(S, regression="c", autolag=None, maxlag=0)

        # AR(1) coefficient via OLS on the same regression
        dS      = S.diff().dropna()
        S_lag   = S.shift(1).dropna().loc[dS.index]
        # beta = cov(dS, S_lag) / var(S_lag)
        beta    = np.cov(dS, S_lag, ddof=1)[0, 1] / np.var(S_lag, ddof=1)
        #S_t = phi * S_{t-1} + alpha + eps; phi = 1 + beta; half-life = ln(2) / ln(1/phi)
        phi     = 1 + beta
        if phi >= 1:
            half_life = np.nan
            mr_test   = "FAIL - unit root or explosive (phi >= 1)"
        elif phi <= 0:
            half_life = np.nan
            mr_test   = "FAIL - oscillating or random walk (phi <= 0)"
        else:
            half_life = np.log(2) / np.log(1 / phi)
            mr_test   = f"PASS - mean reverting, half-life = {half_life:.1f} days"
 
        return pd.Series({
            "adf_stat":       round(adf_stat,  3),
            "adf_pvalue":     round(adf_p,     4),
            "ar1_beta":       round(beta,      5),
            'ar1_phi':        round(phi,       5),
            "half_life_days": round(half_life, 1) if not np.isnan(half_life) else np.nan,
            "mr_test":        mr_test,
        })

    def test_mean_reversion_local(self, period: slice) -> pd.Series:
        """
        Test mean reversion on the DEMEANED signal (S - rolling mean)  
        The strategy actually trades, not the global level of S.
 
        A series can trend globally (failing global ADF) while exhibiting
        local mean reversion around its moving average — which is exactly
        what the z-score rule exploits. 
        """
        sig  = self.build_signal().loc[period]
        S    = sig["S"]
 
        # Demeaned signal: deviation from rolling mean (same window as z-score)
        S_lag        = S.shift(1)
        rolling_mean = S_lag.rolling(self.lookback).mean()
        S_demeaned   = (S - rolling_mean).dropna()
 
        adf_stat, adf_p, *_ = adfuller(S_demeaned, regression="n",autolag=None, maxlag=0)
 
        dS    = S_demeaned.diff().dropna()
        S_lag = S_demeaned.shift(1).dropna().loc[dS.index]
        beta  = np.cov(dS, S_lag, ddof=1)[0, 1] / np.var(S_lag, ddof=1)
        phi   = 1 + beta
 
        if phi >= 1:
            half_life = np.nan
            mr_test   = "FAIL - unit root or explosive (phi >= 1)"
        elif phi <= 0:
            half_life = np.nan
            mr_test   = "FAIL - oscillating or random walk (phi <= 0)"
        else:
            half_life = np.log(2) / np.log(1 / phi)
            mr_test   = f"PASS - local mean reversion, half-life = {half_life:.1f} days"
 
        return pd.Series({
            "adf_stat":       round(adf_stat,  3),
            "adf_pvalue":     round(adf_p,     4),
            "ar1_beta":       round(beta,      5),
            'ar1_phi':        round(phi,       5),
            "half_life_days": round(half_life, 1) if not np.isnan(half_life) else np.nan,
            "mr_test":        mr_test,
        })

    # ── Trading positions based on signal ─────────────────────────────────────────────
    def generate_positions(self, period: slice) -> pd.Series:
        """
        Convert z-scores into positions with hysteresis band and direct reversal:
 
          Flat (current == 0):
            enter long fly to buy belly and sell wings (position = +1) when z < -entry_z
            enter short fly to sell belly and buy wings (position = -1) when z > +entry_z
 
          In a position (current != 0):
            |z| < exit_z  → exit to flat (0): spread back to normal
            # direct reversal (edge case) — if signal flips strongly enough, reverse in one step rather than exit-and-wait.
            current == +1 and z > +entry_z  → flip to -1 (long -> short)
            current == -1 and z < -entry_z  → flip to +1 (short -> long)
            otherwise hold
 
          NaN z (first lookback days before rolling window fills):
            hold current position (0 at start, existing position if already in trade)
 
        Sign convention: +1 = long belly / short wings
        """
        z = self.build_signal()["z"].loc[period]

        pos = pd.Series(0, index=z.index, dtype=int)
        current = 0
        for t, z_t in z.items():
            if pd.isna(z_t):
                pos.loc[t] = current
                continue
            if current == 0:
                if z_t < -self.entry_z:
                    current = +1
                elif z_t > self.entry_z:
                    current = -1
            else:
                if abs(z_t) < self.exit_z:
                    current = 0                          # exit to flat
                elif current == +1 and z_t > self.entry_z:
                    current = -1                         # direct reversal: long -> short
                elif current == -1 and z_t < -self.entry_z:
                    current = +1                         # direct reversal: short -> long
                #else: hold — no action
            pos.loc[t] = current
        return pos

    # ── DV01 P&L engine ─────────────────────────────────────────────────────
    def pnl_calc(self, positions: pd.Series) -> pd.DataFrame:
        """
        Daily P&L in dollars:
 
          gross_pnl_t = position_{t-1} x sum_i(-signed_dv01_i x dy_i,t)
            - position_{t-1}: yesterday's close position (no same-day execution)
            - signed_dv01_i : $/bp per leg, signed by direction (belly +, wings -)
            - dy_i,t        : yield change arriving at t (y_t - y_{t-1}), in bps
 
          cost_t = |position_t - position_{t-1}| x sum_i(dv01_per_leg_i x cost_bps)
            - |diff| = 1 for normal entry/exit  → 1x cost
            - |diff| = 2 for direct reversal    → 2x cost (close + re-open)
            - cost_bps: bid-ask in bps; cost in $ = DV01 ($/bp) x bid-ask (bps)
 
          net_pnl_t = gross_pnl_t - cost_t
        """
        tenors = self.butterfly.tenors
        dy     = self.yields_bps[tenors].diff().loc[positions.index]
        
        # Daily gross P&L: yesterday's position applied to today's yield moves
        raw = -dy.mul(self._signed_dv01, axis=1).sum(axis=1)
        gross = raw * positions.shift(1).fillna(0)

        # Cost proportional to position change magnitude
        # 1 unit of position = 1 x belly_mm; a full round-trip crosses all 3 legs
        pos_change    = positions.diff().abs().fillna(positions.abs().iloc[0]) # fillna handles first-row NaN from .diff() 
        cost_per_unit = self._dv01_per_leg.sum() * self.cost_bps
        cost          = pos_change * cost_per_unit
 
        net = gross - cost
        return pd.DataFrame({
            "position":  positions.astype(int),
            "gross_pnl": gross,
            "cost":      cost,
            "net_pnl":   net,
            "cum_pnl":   net.cumsum(),
        })

    # ── Plots ─────────────────────────────────────────────────────
    def plot_signal(self, period: slice, positions: pd.Series = None) -> None:
        """
        Two-panel plot:
          Top    : cumulative PC3 score S with rolling mean overlay
          Bottom : z-score with entry/exit bands and position shading
        """
        sig = self.build_signal().loc[period]
        z   = sig["z"]
        S   = sig["S"]
 
        # Rolling mean of S (same window as z-score)
        S_lag       = S.shift(1)
        rolling_mean = S_lag.rolling(self.lookback).mean()
 
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
 
        # ── Top: S with rolling mean ─────────────────────────────────────
        ax1.plot(S.index, S*100,            label="Cumulative PC3 score (S)", lw=0.8)
        ax1.plot(S.index, rolling_mean*100, label=f"{self.lookback}d rolling mean",
                 lw=0.8, linestyle="--", color="orange")
        ax1.set_ylabel("Cumulative PC3 score (bps)")
        ax1.set_title("Cumulative PC3 Score vs Rolling Mean")
        ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
 
        # ── Bottom: z-score with bands and position shading ──────────────
        ax2.plot(z.index, z, lw=0.8, color="steelblue", label="z-score")
        ax2.axhline( self.entry_z, color="red",   lw=0.8, linestyle="--", label=f"±{self.entry_z} entry")
        ax2.axhline(-self.entry_z, color="red",   lw=0.8, linestyle="--")
        ax2.axhline( self.exit_z,  color="green", lw=0.8, linestyle=":",  label=f"±{self.exit_z} exit")
        ax2.axhline(-self.exit_z,  color="green", lw=0.8, linestyle=":")
        ax2.xaxis.set_major_locator(MonthLocator())
        ax2.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",fontsize=8)
 
        # Shade periods in position
        if positions is not None:
            pos = positions.reindex(z.index).fillna(0)
            ax2.fill_between(z.index, -4, 4,
                             where=pos > 0, alpha=0.15, color="green", label="long fly")
            ax2.fill_between(z.index, -4, 4,
                             where=pos < 0, alpha=0.15, color="red",   label="short fly")
 
        ax2.set_ylabel("z-score"); ax2.set_ylim(-4, 4)
        ax2.set_title("Z-score with Entry/Exit Bands")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
 
        plt.tight_layout()
 
    def plot_pnl(self, pnl: pd.DataFrame, label: str = "") -> None:
        """
        Two-panel P&L plot:
          Top    : cumulative gross and net P&L
          Bottom : daily cost vs gross P&L (shows cost drag)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 7), sharex=True)
 
        cum_gross = pnl["gross_pnl"].cumsum()
        cum_net   = pnl["cum_pnl"]
        cum_cost  = pnl["cost"].cumsum()
 
        # ── Top: cumulative P&L ──────────────────────────────────────────
        ax1.plot(cum_gross.index, cum_gross / 1e6, label="Cumulative gross P&L ($mm)", lw=0.9)
        ax1.plot(cum_net.index,   cum_net   / 1e6, label="Cumulative net P&L ($mm)",   lw=0.9)
        ax1.plot(cum_cost.index,  -cum_cost  / 1e6, label="Cumulative cost drag ($mm)",
                 lw=0.8, linestyle="--", color="red")
        ax1.axhline(0, color="black", lw=0.5)
        ax1.set_ylabel("P&L ($mm)")
        ax1.set_title(f"Cumulative P&L{' — ' + label if label else ''}")
        ax1.legend(fontsize=9); ax1.grid(alpha=0.3)
 
        # ── Bottom: daily gross vs cost ──────────────────────────────────
        ax2.bar(pnl.index, pnl["gross_pnl"] / 1e3, label="Daily gross P&L ($k)",
                alpha=0.6, width=1)
        ax2.bar(pnl.index, -pnl["cost"] / 1e3,     label="Daily cost drag ($k)",
                alpha=0.8, color="red", width=1)
        ax2.set_ylabel("Daily P&L ($k)")
        ax2.set_title("Daily Gross P&L vs Cost")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_locator(MonthLocator())
        ax2.xaxis.set_major_formatter(DateFormatter("%b %Y"))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", fontsize=8)
 
        plt.tight_layout()
 
    # ── Performance Metrics ─────────────────────────────────────────────────────────────

    def performance_metrics(self, pnl: pd.DataFrame) -> pd.Series:
        """
        Standard backtest metrics. Returns expressed as % of gross notional
        gross_notional = sum(|leg notionals|) = weights.abs().sum() x belly_mm x 1e6
        """
        gross_notional = (self.weights.abs().sum() * self.belly_mm) * 1e6  # dollars

        total_gross = pnl["gross_pnl"].dropna().sum()
        total_cost  = pnl["cost"].dropna().sum()
        total_net   = pnl["net_pnl"].dropna().sum()

        daily_net = pnl["net_pnl"].dropna()
        n = len(daily_net)
        if n == 0:
            return pd.Series(dtype=float)

        # Evaluate net performance
        ##Annualized return, annualized volatility, Sharpe ratio
        ann_ret_pct = (total_net / gross_notional) * (252 / n) * 100
        ann_vol_pct = daily_net.std() * np.sqrt(252) / gross_notional * 100
        sharpe      = ann_ret_pct / ann_vol_pct if ann_vol_pct > 0 else np.nan

        ## Max drawdown on cumulative net P&L
        cum         = daily_net.cumsum()
        peak        = cum.cummax()
        drawdown    = cum - peak
        max_dd      = drawdown.min()  #Max drawdown in dollars
        max_dd_pct  = drawdown.min() / gross_notional * 100 #Max drawdown as % of gross notional 

        # Trades and holding period
        # Count both fresh entries AND direct reversals as separate trades
        pos         = pnl["position"]
        fresh_entry = (pos != 0) & (pos.shift(1).fillna(0) == 0)
        reversal    = (pos != 0) & (pos.shift(1).fillna(0) != 0) & (pos != pos.shift(1))
        entries     = (fresh_entry | reversal).sum()
        # avg_holding: average days a trade stands open
        # compare to half-life — if avg_holding >> half-life, exiting too late
        avg_holding = (pos != 0).sum() / entries if entries > 0 else np.nan
        turnover    = entries / (n / 252)

        # Evaluate cost vs gross P&L decomposition
        ## Annualized gross and cost returns as % of gross notional
        ann_gross_ret_pct = (total_gross / gross_notional) * (252 / n) * 100
        ann_cost_ret_pct = (total_cost / gross_notional) * (252 / n) * 100

        return pd.Series({
            "annual_return_%":   round(ann_ret_pct, 3),
            "annual_gross_return_%":  round(ann_gross_ret_pct, 3),
            "annual_cost_return_%":   round(ann_cost_ret_pct, 3),
            "annual_vol_%":      round(ann_vol_pct, 3),
            "sharpe":            round(sharpe,      3),
            "max_drawdown_mm":    round(max_dd/1e6,      1),
            "max_drawdown_%":    round(max_dd_pct,  3),
            "num_trades":        int(entries),
            "turnover_per_year": round(turnover,    2),
            "avg_holding_days":  round(avg_holding, 1),
            "total_gross_pnl_mm":round(total_gross/1e6,  1),
            "total_cost_mm":     round(total_cost/1e6,  1),
            "total_net_pnl_mm":  round(total_net/1e6,  1),
        })

    def run(self, period: slice) -> dict:
        """
        End-to-end backtest on the given date window.
        Uses the frozen strategy — same weights, lookback, thresholds.
        """
        positions = self.generate_positions(period)
        pnl       = self.pnl_calc(positions)
        metrics   = self.performance_metrics(pnl)
        return {"positions": positions, "pnl": pnl, "metrics": metrics}


def main() -> None:
    # Task 1: data
    print(f'\n────────────────Task 1────────────────') 
    tcd     = TreasuryCurveData()
    yields  = tcd.get_clean_yields()
    changes = tcd.get_yield_changes()   # decimal daily first differences
    tcd.plot_yields()
    tcd.plot_yield_changes()
 
    # Task 2: PCA
    print(f'\n────────────────Task 2────────────────')
    # Work in basis points so eigenvalues have interpretable scale.
    yields_bps = yields * 100
    changes_bps = changes * 10_000
    max_components = min(changes_bps.shape)
 
    # 1. Full-sample PCA — the reference for sign alignment
    pca_full = TreasuryPCA(n_components = max_components, standardize=False).fit(changes_bps)
    print(f"\nFull-sample PCA:")
    print("Loadings:\n", pca_full.loadings_.round(5))
    print("\nSummary:\n", pca_full.summary("Full"))
    pca_full.scree_plot("Full sample")
 
    # 2. Sub-period PCAs, sign-aligned to full sample
    pre  = changes_bps.loc[changes_bps.index <  REGIME_SPLIT]
    post = changes_bps.loc[changes_bps.index >= REGIME_SPLIT]
 
    pca_pre  = TreasuryPCA(n_components = max_components).fit(pre ).align_signs(pca_full)
    pca_post = TreasuryPCA(n_components = max_components).fit(post).align_signs(pca_full)
 
    print(f"\nSub-period PCAs (split at {REGIME_SPLIT}):")
    print(f"Pre  : {len(pre)} obs   Post : {len(post)} obs")
 
    # 3. Side-by-side loading comparison
    loadings_compare = pd.concat(
        {"Full": pca_full.loadings_,
         "Pre":  pca_pre.loadings_,
         "Post": pca_post.loadings_},
        axis=1,
    )
    print("\nLoadings across periods:\n", loadings_compare.round(3))
 
    # 4. Variance-share comparison
    variance_compare = pd.concat(
        [pca_full.summary("Full"),
         pca_pre .summary("Pre"),
         pca_post.summary("Post")],
        axis=1,
    )
    print("\nVariance shares across periods:\n", variance_compare)

    # Task 3: Factor-neutral butterfly 
    print(f'\n────────────────Task 3────────────────')
    fly = Butterfly(tenors = BUTTERFLY_TENORS, durations = DURATIONS, pca = pca_full) # full-sample loadings; 

    # 1. DV01 table (task deliverable — assumes par pricing)
    print("\nDV01 table:\n", fly.dv01_table(notional=100.0))
  
    # 2. Factor-neutral weights
    w_fn = fly.solve_factor_neutral()
    print("\nFactor-neutral weights (PC1 & PC2 exposure = 0):")
    print(w_fn)
 
    # 3. Naive DV01-neutral weights
    w_dv = fly.solve_dv01_neutral()
    print("\nDV01-neutral weights (equal-DV01 wings, no factor info):")
    print(w_dv)
 
    # 4. Residual exposure comparison
    print("\nResidual factor exposures ($/unit factor move, belly = $100mm):")
    print(fly.exposure_table(BELLY_NOTIONAL))

    # Task 4: Strategy using PC3 Score as Signal
    print(f'\n────────────────Task 4────────────────')
    is_slice  = slice(None, IS_END)
    oos_slice = slice(OOS_START, None)

    # 1. Fit PCA & solve butterfly weights on in-sample data only 
    pca_is = TreasuryPCA(n_components=3).fit(changes_bps.loc[is_slice])
    fly_is = Butterfly(BUTTERFLY_TENORS, DURATIONS, pca=pca_is)
 
    strat = MeanReversionStrategy(
        yields_bps = yields_bps,
        pca        = pca_is,
        butterfly  = fly_is,
        lookback   = LOOKBACK,
        entry_z    = ENTRY_Z,
        exit_z     = EXIT_Z,
        cost_bps   = COST_BPS,
        belly_mm   = BELLY_NOTIONAL, #in $mm
    )
 
    # 2. Justify the trade using IS data only
    print(f"\nMean-reversion diagnostics:")
    print(f"In-sample through {IS_END})")
    print(strat.test_mean_reversion(is_slice))
    print(f"Out-of-sample from {OOS_START})")
    print(strat.test_mean_reversion(oos_slice))
    
    # Local mean-reversion test (what the strategy actually trades) ─ Tests S - rolling_mean(S) rather than the global level of S.
    # A trending S can still show local reversion around its moving average.
    print("\nLocal mean-reversion test (demeaned signal):")
    print(f"In-sample through {IS_END})")
    print(strat.test_mean_reversion_local(is_slice))
    print(f"Out-of-sample from {OOS_START})")
    print(strat.test_mean_reversion_local(oos_slice))

    # 3. Backtest both windows with the same frozen strategy
    is_result  = strat.run(is_slice)
    oos_result = strat.run(oos_slice)
 
    print("\nIn-sample backtest performance metrics:")
    print(is_result["metrics"])
    print("\nOut-of-sample backtest performance metrics:")
    print(oos_result["metrics"])
 
    # Signal and P&L plots
    strat.plot_signal(is_slice,  positions=is_result["positions"])
    strat.plot_signal(oos_slice, positions=oos_result["positions"])
    strat.plot_pnl(is_result["pnl"],  label="IS  (2019-2023)")
    strat.plot_pnl(oos_result["pnl"], label="OOS (2024-2025)")

    # 4. Bridge back to Task 2: stale-weight neutrality drift ─────────────────
    # The in-sample weights zero out PC1/PC2 exposure under in-sample loadings. 
    # Under out-of-sample loadings they no longer do — quantify the residual in dollars.
    pca_oos          = TreasuryPCA(n_components=3).fit(changes_bps.loc[oos_slice])
    fly_oos_loadings = Butterfly(BUTTERFLY_TENORS, DURATIONS, pca=pca_oos)
    w_is             = fly_is.solve_factor_neutral().values
 
    stale_exposure   = fly_oos_loadings._factor_exposures(w_is * BELLY_NOTIONAL)  # $/unit factor move
    print("\nStale-weight residual exposure:")
    print("IS weights scored under OOS loadings ($/unit factor move):")
    print(pd.Series(stale_exposure, index=["PC1", "PC2", "PC3"]).round(2))

    plt.show()  # show all plots at the end
 

if __name__ == "__main__":
    main()
 