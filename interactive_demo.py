"""
Seeing the Curve — Interactive Demo

Usage:
    pip install streamlit plotly
    streamlit run demo.py

Assumes all classes (TreasuryCurveData, TreasuryPCA, Butterfly,
MeanReversionStrategy) live in seeing_the_curve.py in the same directory.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from seeing_the_curve import (
    TreasuryCurveData,
    TreasuryPCA,
    Butterfly,
    MeanReversionStrategy,
    TreasuryCurveData,
    TreasuryPCA,
    Butterfly,
    MeanReversionStrategy,
    DURATIONS,
    BUTTERFLY_TENORS,
    IS_END,
    OOS_START,
    REGIME_SPLIT,
)



# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Seeing the Curve", layout="wide")
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 800;
        color: red;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: darkred;
        border-bottom: 3px solid darkred;
    }
    </style>
    """, unsafe_allow_html=True)
st.title("Seeing the Curve")
st.caption("U.S. Treasury Curve PCA & Factor-Neutral Relative-Value Strategy")


# ── Sidebar controls ────────────────────────────────────────────────────────

st.sidebar.header("Parameters")

lookback = st.sidebar.slider("Z-score lookback (days)", 20, 120, 60, step=5)
entry_z  = st.sidebar.slider("Entry |z| threshold", 1.0, 3.0, 2.0, step=0.1)
exit_z   = st.sidebar.slider("Exit |z| threshold",  0.0, 1.5, 0.5, step=0.1)
cost_bps = st.sidebar.slider("Transaction Cost in bps (bid/ask spread)",  0.1, 1.0, 0.2, step=0.05)
belly_mm = st.sidebar.number_input("Belly notional ($mm)", value=100.0, step=10.0)

run_btn = st.sidebar.button("Run Analysis", type="primary", width='stretch')


# ── Cached data loading ─────────────────────────────────────────────────────

@st.cache_data(show_spinner="Fetching Treasury data from OpenBB …")
def load_data():
    tcd     = TreasuryCurveData()
    yields  = tcd.get_clean_yields()
    changes = tcd.get_yield_changes()
    return yields, changes




# ── Run analysis ─────────────────────────────────────────────────────────────

if run_btn or "results" in st.session_state:

    yields, changes = load_data()
    yields_bps  = yields  * 100
    changes_bps = changes * 10000

    # ── PCA (full sample for display, IS for strategy) ───────────────────
    pca_full = TreasuryPCA(n_components=6).fit(changes_bps)
    pca_is   = TreasuryPCA(n_components=3).fit(changes_bps.loc[:IS_END])

    # ── Butterfly ────────────────────────────────────────────────────────
    fly_full = Butterfly(BUTTERFLY_TENORS, DURATIONS, pca=pca_full)
    fly_is   = Butterfly(BUTTERFLY_TENORS, DURATIONS, pca=pca_is)

    # ── Strategy ─────────────────────────────────────────────────────────
    strat = MeanReversionStrategy(
        yields_bps = yields_bps,
        pca        = pca_is,
        butterfly  = fly_is,
        lookback   = lookback,
        entry_z    = entry_z,
        exit_z     = exit_z,
        cost_bps   = cost_bps,
        belly_mm   = belly_mm,
    )

    is_result  = strat.run(slice(None, IS_END))
    oos_result = strat.run(slice(OOS_START, None))

    # Store in session so tabs persist without re-running
    st.session_state["results"] = True

    # ════════════════════════════════════════════════════════════════════
    # TABS
    # ════════════════════════════════════════════════════════════════════

    tab1, tab2, tab3 = st.tabs([
        "1 · Yield Curve & PCA",
        "2 · Butterfly",
        "3 · Strategy",
    ])

    # ── TAB 1: Yield Curve & PCA ─────────────────────────────────────────
    with tab1:
        st.subheader("Daily Yield Levels")
        fig_yields = go.Figure()
        for col in yields_bps.columns:
            fig_yields.add_trace(go.Scatter(
                x=yields_bps.index, y=yields_bps[col] / 100,
                name=col, mode="lines", line=dict(width=1),
            ))
        fig_yields.update_layout(
            yaxis_title="Yield (%)", height=400,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30),
        )
        fig_yields.update_xaxes(
            dtick="M1",           # tick every month
            tickformat="%b %y",   # format as "Jan 19"
        )
        st.plotly_chart(fig_yields, width='stretch')

        st.subheader("Daily Yield Changes")
        fig_changes = go.Figure()
        for col in changes_bps.columns:
            fig_changes.add_trace(go.Scatter(
                x=changes_bps.index, y=changes_bps[col],
                name=col, mode="lines", line=dict(width=0.6),
            ))
        fig_changes.update_layout(
            yaxis_title="ΔYield (bps)", height=350,
            legend=dict(orientation="h", y=-0.15),
            margin=dict(t=30),
        )
        fig_changes.update_xaxes(
            dtick="M1",
            tickformat="%b %y",
        )
        st.plotly_chart(fig_changes, width='stretch')

        # ── Scree plot (smaller) + variance table side by side ───────────────
        st.subheader("PCA — Variance Explained (Full Sample)")
        shares = pca_full.explained_variance_ratio_ * 100
        cum    = shares.cumsum()
        pcs    = [f"PC{i+1}" for i in range(len(shares))]

        col_scree, col_var = st.columns([3, 2])
        with col_scree:
            fig_scree = make_subplots(specs=[[{"secondary_y": True}]])
            fig_scree.add_trace(
                go.Bar(x=pcs, y=shares, name="Individual %", opacity=0.7),
                secondary_y=False,
            )
            fig_scree.add_trace(
                go.Scatter(x=pcs, y=cum, name="Cumulative %",
                        mode="lines+markers", marker=dict(color="firebrick")),
                secondary_y=True,
            )
            fig_scree.update_layout(
                height=280, margin=dict(t=10, b=10),
                yaxis_title="Variance explained (%)",
                yaxis2_title="Cumulative (%)",
                yaxis2_range=[0, 105],
                legend=dict(orientation="h", y=-0.25),
            )
            st.plotly_chart(fig_scree, width='stretch')

        with col_var:
            # Variance table for full sample
            var_table = pd.DataFrame({
                "Variance %":    shares.round(2),
                "Cumulative %":  cum.round(2),
                "Eigenvalue":    pca_full.eigenvalues_.round(3),
            }, index=pcs)
            st.dataframe(var_table, width='stretch', height=280)    

        # ── PC Loadings — full sample + two regimes ───────────────────────────
        st.subheader("PC Loadings")

        # Full sample
        st.caption("Full Sample (2019–2025)")
        st.dataframe(
            pca_full.loadings_[["PC1","PC2","PC3", 'PC4', 'PC5', 'PC6']].round(4),
            width='stretch'
        )   

        # Fit sub-period PCAs aligned to full sample signs
        pre_changes  = changes_bps.loc[changes_bps.index <  REGIME_SPLIT]
        post_changes = changes_bps.loc[changes_bps.index >= REGIME_SPLIT]
        pca_pre  = TreasuryPCA(n_components=3).fit(pre_changes ).align_signs(pca_full)
        pca_post = TreasuryPCA(n_components=3).fit(post_changes).align_signs(pca_full)

        # Pre/post side by side
        col_pre, col_post = st.columns(2)
        with col_pre:
            pre_end = pd.Timestamp(REGIME_SPLIT) - pd.Timedelta(days=1)
            st.caption(f"Pre-hike regime (2019-01-01 – {pre_end.strftime('%Y-%m-%d')})")
            st.dataframe(
                pca_pre.loadings_[["PC1","PC2","PC3"]].round(4),
                width='stretch'
            )

        with col_post:
            st.caption(f"Post-hike regime ({REGIME_SPLIT} – 2025-12-31)")
            st.dataframe(
                pca_post.loadings_[["PC1","PC2","PC3"]].round(4),
                width='stretch',
            )

        st.info(
            "**PC1** = level · **PC2** = slope · **PC3** = curvature · "
            f"Full sample explains {cum[2]:.1f}% variance with 3 PCs. "
        )


    # ── TAB 2: Butterfly ─────────────────────────────────────────────────
    with tab2:
        st.subheader("DV01 per $100mm Notional")
        st.caption("Assumes par pricing: DV01 = Modified Duration × $100mm × 0.0001")
        col_dv01, _ = st.columns([2, 1])
        with col_dv01:
            st.dataframe(
                fly_full.dv01_table(notional=100.0).round(1),
                width='stretch', height=143,
            )

        st.divider()

        # ── Weights ───────────────────────────────────────────────────────
        st.subheader("Butterfly Leg Weights")
        st.caption(
            "Weights estimated on **full sample (2019–2025)** PCA loadings. "
            "Belly weight = +1 (long belly as base). Wings are negative (short)."
        )

        col_fn, col_dv, _ = st.columns([2, 2, 1])
        with col_fn:
            st.markdown("**Factor-Neutral**")
            st.caption("PC1 & PC2 dollar exposure = 0")
            w_fn = fly_full.solve_factor_neutral()
            st.dataframe(
                w_fn.round(4).to_frame("weight"),
                width='stretch', height=143
            )

        with col_dv:
            st.markdown("**DV01-Neutral (Naive)**")
            st.caption("Equal-DV01 wings, no factor info")
            w_dv = fly_full.solve_dv01_neutral()
            st.dataframe(
                w_dv.round(4).to_frame("weight"),
                width='stretch', height=143,
            )

        st.divider()

        # ── Residual exposure ─────────────────────────────────────────────
        st.subheader("Residual Factor Exposure")
        st.caption(
            f"Dollar P&L per 1-unit factor move at ${belly_mm:.0f}mm belly notional. "
            "PC1/PC2 as % of PC3 quantifies unintended factor bets."
        )

        exp_table = fly_full.exposure_table(belly_mm=belly_mm)
        col_exp, _ = st.columns([3, 1])
        with col_exp:
            st.dataframe(
                exp_table.round(2),
                width='stretch', height=108,
            )

        dv_pc1_pct = exp_table.loc["DV01-neutral", "PC1_as_%_of_PC3"]
        dv_pc2_pct = exp_table.loc["DV01-neutral", "PC2_as_%_of_PC3"]
        fn_pc1_pct = exp_table.loc["Factor-neutral", "PC1_as_%_of_PC3"]
        fn_pc2_pct = exp_table.loc["Factor-neutral", "PC2_as_%_of_PC3"]

        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.success(
                f"**Factor-neutral fly:** PC1 = {fn_pc1_pct:.1f}% · "
                f"PC2 = {fn_pc2_pct:.1f}% of curvature exposure — "
                "effectively zero unintended factor bets."
            )
        with col_info2:
            st.warning(
                f"**DV01-neutral fly:** PC1 = {dv_pc1_pct:.1f}% · "
                f"PC2 = {dv_pc2_pct:.1f}% of curvature exposure — "
                "material unintended level and slope bets."
            )

    # ── TAB 3: Strategy ──────────────────────────────────────────────────
    with tab3:

        # ── Pre-compute ───────────────────────────────────────────────────
        sig     = strat.build_signal()
        is_pnl  = is_result["pnl"]
        oos_pnl = oos_result["pnl"]
        all_pos = pd.concat([
            is_result["positions"],
            oos_result["positions"],
        ]).astype(int)

        # ── Signal plot ───────────────────────────────────────────────────
        st.subheader("Signal — Cumulative PC3 Score & Z-score")

        S          = sig["S"]
        S_lag      = S.shift(1)
        roll_mean  = S_lag.rolling(lookback).mean()
        z          = sig["z"]

        fig_signal = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.5, 0.5],
            subplot_titles=[
                "Cumulative PC3 Score (S) with Rolling Mean in bps",
                "Z-score with Entry/Exit Bands",
            ],
            vertical_spacing=0.08,
        )

        # Top: S with rolling mean
        fig_signal.add_trace(
            go.Scatter(x=S.index, y=S*100,
                       name="S", line=dict(width=1, color="steelblue")),
            row=1, col=1,
        )
        fig_signal.add_trace(
            go.Scatter(x=roll_mean.index, y=roll_mean*100,
                       name=f"{lookback}d mean",
                       line=dict(width=1, dash="dash", color="orange")),
            row=1, col=1,
        )
        fig_signal.add_vline(
            x=OOS_START, line_dash="dot", line_color="gray",
            annotation_text="IS | OOS Cutoff", annotation_position="top right",
        )

        # Bottom: z-score
        fig_signal.add_trace(
            go.Scatter(x=z.index, y=z,
                       name="z-score",
                       line=dict(width=1, color="steelblue")),
            row=2, col=1,
        )

        # Entry/exit bands
        for val, color, dash, label in [
            ( entry_z, "red",   "dash", f"+{entry_z}σ entry"),
            (-entry_z, "red",   "dash", f"-{entry_z}σ entry"),
            ( exit_z,  "green", "dot",  f"+{exit_z}σ exit"),
            (-exit_z,  "green", "dot",  f"-{exit_z}σ exit"),
        ]:
            fig_signal.add_hline(
                y=val, line_dash=dash, line_color=color,
                row=2, col=1,
                annotation_text=label,
                annotation_position="right",
            )

        # Position shading — fast: one vrect per contiguous trade period
        def get_shading_ranges(positions, value):
            mask    = (positions == value)
            ranges  = []
            in_band = False
            start   = None
            for date, val in mask.items():
                if val and not in_band:
                    start   = date
                    in_band = True
                elif not val and in_band:
                    ranges.append((start, date))
                    in_band = False
            if in_band:
                ranges.append((start, mask.index[-1]))
            return ranges

        for x0, x1 in get_shading_ranges(all_pos, 1):
            fig_signal.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(0,180,0,0.12)",
                line_width=0, layer="below",
                row=2, col=1,
            )
        for x0, x1 in get_shading_ranges(all_pos, -1):
            fig_signal.add_vrect(
                x0=x0, x1=x1,
                fillcolor="rgba(180,0,0,0.12)",
                line_width=0, layer="below",
                row=2, col=1,
            )

       # Dummy traces for legend
        fig_signal.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color="rgba(0,180,0,0.4)", size=10, symbol="square"),
                name="Long fly (long belly)",
                showlegend=True,
            ),
            row=2, col=1,
        )
        fig_signal.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(color="rgba(180,0,0,0.4)", size=10, symbol="square"),
                name="Short fly (short belly)",
                showlegend=True,
            ),
            row=2, col=1,
        ) 

        fig_signal.add_vline(
            x=OOS_START, line_dash="dot", line_color="gray",
            row=2, col=1,
        )
        fig_signal.update_yaxes(range=[-4, 4], row=2, col=1)
        fig_signal.update_xaxes(dtick="M1", tickformat="%b %y")
        fig_signal.update_layout(height=600, margin=dict(t=40))
        st.plotly_chart(fig_signal, width="stretch")

        st.divider()

        # ── Cumulative P&L ────────────────────────────────────────────────
        st.subheader("Cumulative P&L")

        all_pnl   = pd.concat([is_pnl, oos_pnl])
        cum_gross = all_pnl["gross_pnl"].cumsum() / 1e6
        cum_net   = all_pnl["cum_pnl"] / 1e6
        cum_cost  = -all_pnl["cost"].cumsum() / 1e6

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=all_pnl.index, y=cum_gross,
            name="Gross P&L ($mm)", line=dict(width=1),
        ))
        fig_pnl.add_trace(go.Scatter(
            x=all_pnl.index, y=cum_net,
            name="Net P&L ($mm)", line=dict(width=1),
        ))
        fig_pnl.add_trace(go.Scatter(
            x=all_pnl.index, y=cum_cost,
            name="Cost drag ($mm)",
            line=dict(width=1, dash="dash", color="red"),
        ))
        fig_pnl.add_vline(
            x=OOS_START, line_dash="dot", line_color="gray",
            annotation_text="IS | OOS Cutoff", annotation_position="top right",
        )
        fig_pnl.add_hline(y=0, line_color="black", line_width=0.5)
        fig_pnl.update_layout(
            yaxis_title="P&L ($mm)", height=400, margin=dict(t=30),
            xaxis=dict(dtick="M1", tickformat="%b %y"),
        )
        st.plotly_chart(fig_pnl, width="stretch")

        st.divider()

        # ── Performance metrics (includes cost decomposition) ─────────────
        st.subheader("Performance Metrics — In-sample vs Out-of-sample")
        renames = { "annual_return_%":  'Annualized Net Return (%)',
                    "annual_gross_return_%":  'Annualized Gross Return (%)',
                    "annual_cost_return_%":   'Annualized Cost (%)',
                    "annual_vol_%":     'Annualized Volatility (%, based on net returns)',
                    "sharpe":            'Sharpe Ratio',
                    "max_drawdown_mm":    'Max Drawdown ($)',
                    "max_drawdown_%":    'Max Drawdown (% of Gross Notional)',
                    "num_trades":        'Total Number of Trades',
                    "turnover_per_year": 'Turnover per Year',
                    "avg_holding_days":  'Average Holding Days',
                    "total_gross_pnl_mm":'Total Gross P&L ($mm)',
                    "total_cost_mm":     'Total Cost ($mm)',
                    "total_net_pnl_mm":  'Total Net P&L ($mm)',
                    }
        metrics_df = pd.DataFrame({
            f"In-Sample (through {IS_END})":  is_result["metrics"].rename(renames),
            f"Out-of-Sample (from {OOS_START})": oos_result["metrics"].rename(renames),
        })
        metrics_df.columns = metrics_df.columns.astype(str)
        st.dataframe(metrics_df, width="stretch", height = 528)

        st.divider()

        # ── Mean-reversion diagnostics ────────────────────────────────────
        st.subheader("Mean-Reversion Diagnostics")
        
        cols = ['ar1_phi', "half_life_days", "mr_test"]
        rename_cols = {
            'ar1_phi': 'AR(1) φ',
            "half_life_days": "Half-Life (Days)",
            "mr_test": "Mean Reversion Test"
        }

        mr_is  = strat.test_mean_reversion(slice(None, IS_END))
        mr_oos = strat.test_mean_reversion(slice(OOS_START, None))
        lr_is  = strat.test_mean_reversion_local(slice(None, IS_END))
        lr_oos = strat.test_mean_reversion_local(slice(OOS_START, None))

        col_global, col_local = st.columns(2)
        with col_global:
            st.caption("Global ADF (raw cumulative PC3 score S)")
            st.dataframe(
                pd.DataFrame(
                    [mr_is.loc[cols].rename(rename_cols), mr_oos.loc[cols].rename(rename_cols)],
                    index=["In-Sample", "Out-of-Sample"],
                ).T.astype(str),
                width="stretch",
            )
        with col_local:
            st.caption("Local ADF (demeaned signal S − rolling mean)")
            st.dataframe(
                pd.DataFrame(
                    [lr_is.loc[cols].rename(rename_cols), lr_oos.loc[cols].rename(rename_cols)],
                    index=["In-Sample", "Out-of-Sample"],
                ).T.astype(str),
                width="stretch",
            )

        st.info(
            "**Global ADF fails on IS** — curvature trended through the "
            "2022 hiking cycle rather than reverting. "
            "**Local ADF passes** — the demeaned signal shows mean reversion "
            f"with IS half-life ≈ {lr_is.get('half_life_days', 'N/A')} days "
            f"and OOS half-life ≈ {lr_oos.get('half_life_days', 'N/A')} days. "
            "However gross P&L is near zero — transaction costs dominate "
            "even at tight bid-ask assumptions."
        )

else:
    st.info("Configure parameters in the sidebar and click **Run Analysis**.")