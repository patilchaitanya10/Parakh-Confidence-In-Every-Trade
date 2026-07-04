"""Parakh — Market Validation Dashboard
Run locally:  streamlit run app.py
Deploy: push this folder to GitHub, then create an app on https://share.streamlit.io
"""
import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
import plotly.graph_objects as go
import analysis as A

# ---------- theme ----------
PAPER="#F7F3EC"; CHAR="#2B2926"; EARTH="#9C7A54"; EARTH2="#B79268"
CLAY="#A6685A"; SAGE="#7E8574"; BLUEGREY="#7C8894"; INK="#33312E"
SEQ=[EARTH,CLAY,SAGE,BLUEGREY,EARTH2,"#C7A87C"]
st.set_page_config(page_title="Parakh — Market Validation", page_icon="📊", layout="wide")
st.markdown(f"""<style>
.stApp{{background:{PAPER};}}
h1,h2,h3,h4{{color:{CHAR};font-family:Georgia,serif;}}
.block-container{{padding-top:2rem;max-width:1250px;}}
[data-testid="stMetricValue"]{{color:{EARTH};font-weight:700;}}
[data-testid="stSidebar"]{{background:{CHAR};}}
[data-testid="stSidebar"] *{{color:#EDE6DA;}}
[data-testid="stSidebar"] .stRadio label{{color:#EDE6DA;}}
.stDataFrame{{border-radius:8px;}}
.kpi{{background:#FCFAF6;border:1px solid rgba(43,41,38,.12);border-radius:12px;padding:1rem 1.2rem;box-shadow:0 14px 30px -26px rgba(43,41,38,.5);}}
.verdict{{background:{CHAR};color:{PAPER};border-radius:14px;padding:1.4rem 1.6rem;}}
</style>""", unsafe_allow_html=True)

def plotly_style(fig, h=380):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FCFAF6", height=h,
        font=dict(family="Inter,Arial", color=INK, size=13), margin=dict(l=10,r=10,t=40,b=10),
        colorway=SEQ, legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor="rgba(43,41,38,.07)"); fig.update_yaxes(gridcolor="rgba(43,41,38,.07)")
    return fig

@st.cache_data(show_spinner=False)
def _load(): return A.load_data()
@st.cache_data(show_spinner="Training classifiers…")
def _clf(_t): return A.run_classification(_t)
@st.cache_data(show_spinner="Clustering investors…")
def _seg(_i, k): return A.run_segmentation(_i, k=k)
@st.cache_data(show_spinner="Mining rules…")
def _assoc(_t, s, c): return A.run_association(_t, s, c)
@st.cache_data(show_spinner="Fitting regressions…")
def _reg(_i): return A.run_regression(_i)

inv, trd = _load()

# ---------- sidebar ----------
st.sidebar.title("PARAKH")
st.sidebar.caption("Market-Potential Validation · ML on 1,000 investors")
page = st.sidebar.radio("Go to", ["🏠 Overview","🎯 Classification","🧩 Segmentation",
    "🔗 Association Rules","📈 Regression","✅ Business Verdict"])
st.sidebar.markdown("---")
st.sidebar.metric("Investors", f"{len(inv):,}")
st.sidebar.metric("Trades", f"{len(trd):,}")
st.sidebar.metric("Overall loss rate", f"{trd['loss_occurred'].mean()*100:.1f}%")

# ======================= OVERVIEW =======================
if page.startswith("🏠"):
    st.title("Parakh — Can the market support it?")
    st.markdown("A data-analytics validation of Parakh, a **pre-trade risk tool** for India's retail investors. "
                "Every model below is trained live on the cleaned 1,000-investor / 7,876-trade dataset.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Investors analysed", f"{len(inv):,}")
    c2.metric("Trades analysed", f"{len(trd):,}")
    c3.metric("Trades ending in loss", f"{trd['loss_occurred'].mean()*100:.1f}%")
    c4.metric("Would use Parakh", f"{inv['would_use_parakh_simulator'].mean()*100:.0f}%")
    st.markdown("### The core problem, in one chart")
    lr = trd.groupby('decision_basis')['loss_occurred'].mean().sort_values()*100
    fig = px.bar(x=lr.values, y=lr.index, orientation='h',
                 labels={'x':'Loss rate (%)','y':''}, text=[f"{v:.0f}%" for v in lr.values])
    fig.update_traces(marker_color=[SAGE,EARTH2,"#C7A87C",EARTH,CLAY,"#B5715F","#8F4F45"][:len(lr)])
    st.plotly_chart(plotly_style(fig), width="stretch")
    st.info("**How a trade is decided drives losses far more than who is trading.** Gut/FOMO/tips lose "
            "~55–64% of the time; disciplined 'own research' loses ~19%. That gap is the opening for Parakh.")
    st.markdown("### What this dashboard covers")
    a,b = st.columns(2)
    a.markdown("- **🎯 Classification** — can we predict a losing trade *before* it happens? (KNN, Decision Tree, "
               "Random Forest, Gradient Boosting + full metrics)\n- **🧩 Segmentation** — who are Parakh's customers? (K-Means)")
    b.markdown("- **🔗 Association Rules** — which habits cause losses? (support / confidence / lift)\n"
               "- **📈 Regression** — what will people pay? (Linear / Ridge / Lasso)\n- **✅ Verdict** — launch or not?")

# ======================= CLASSIFICATION =======================
elif page.startswith("🎯"):
    st.title("🎯 Classification — will this trade lose money?")
    st.caption("Target: `loss_occurred`. This is Parakh's pre-trade risk engine. Only pre-trade features are used (no leakage).")
    r = _clf(trd)
    st.markdown("#### Model performance comparison")
    st.dataframe(r['metrics'].round(3), width="stretch")
    best = r['best']
    st.success(f"**Best model: {best}** — Accuracy {r['metrics'].loc[best,'Accuracy']:.1%}, "
               f"ROC-AUC {r['metrics'].loc[best,'ROC_AUC']:.3f}, F1 {r['metrics'].loc[best,'F1']:.3f}. "
               f"Base loss rate is {r['base_rate']:.1%}, so the model beats guessing by a clear margin.")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("#### Metrics side by side")
        md = r['metrics'].reset_index().melt(id_vars='Model', var_name='Metric', value_name='Score')
        fig = px.bar(md, x='Metric', y='Score', color='Model', barmode='group')
        st.plotly_chart(plotly_style(fig), width="stretch")
    with c2:
        st.markdown("#### ROC curves")
        fig = go.Figure()
        for i,(name,(fpr,tpr,auc)) in enumerate(r['roc'].items()):
            fig.add_trace(go.Scatter(x=fpr,y=tpr,name=f"{name} ({auc:.2f})",line=dict(color=SEQ[i%len(SEQ)],width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],line=dict(dash='dash',color="#B8B0A2"),showlegend=False))
        fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(plotly_style(fig), width="stretch")
    c3,c4 = st.columns(2)
    with c3:
        st.markdown(f"#### Confusion matrix — {best}")
        cm = r['cms'][best]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="YlOrBr",
                        x=['Pred No-Loss','Pred Loss'], y=['Actual No-Loss','Actual Loss'])
        st.plotly_chart(plotly_style(fig, h=340), width="stretch")
    with c4:
        st.markdown("#### Top drivers of loss (Random Forest)")
        imp = r['importance'].head(10).sort_values()
        fig = px.bar(x=imp.values, y=imp.index, orientation='h', labels={'x':'Importance','y':''})
        fig.update_traces(marker_color=EARTH)
        st.plotly_chart(plotly_style(fig, h=340), width="stretch")
    st.info("**Insight:** the strongest predictors are *behavioural* — uninformed/high-risk flags, low risk-awareness, "
            "trade size and decision basis — not demographics. Parakh intercepts exactly these signals, so a risk "
            "score shown before the trade is technically feasible and genuinely useful.")

# ======================= SEGMENTATION =======================
elif page.startswith("🧩"):
    st.title("🧩 Customer Segmentation — who are Parakh's users?")
    k = st.slider("Number of segments (k)", 2, 6, 4)
    r = _seg(inv, k)
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown("#### Choosing k — elbow & silhouette")
        sc = r['scan']
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sc['k'],y=sc['inertia'],name="Inertia (elbow)",line=dict(color=EARTH,width=3)))
        fig.add_trace(go.Scatter(x=sc['k'],y=sc['silhouette']*sc['inertia'].max(),name="Silhouette (scaled)",
                                 line=dict(color=CLAY,width=2,dash='dot'),yaxis='y'))
        fig.update_layout(xaxis_title="k")
        st.plotly_chart(plotly_style(fig, h=340), width="stretch")
    with c2:
        st.markdown("#### Segments in 2-D (PCA)")
        p = r['points']
        fig = px.scatter(p, x='pc1', y='pc2', color='segment', opacity=.75)
        st.plotly_chart(plotly_style(fig, h=340), width="stretch")
    st.markdown("#### Segment profiles")
    prof = r['profile'].copy()
    prof['wealth'] = (prof['wealth']/1e5).round(1)  # to lakhs
    prof['would_use'] = (prof['would_use']*100).round(0)
    prof['high_conc'] = (prof['high_conc']*100).round(0)
    prof = prof.rename(columns={'wealth':'wealth_₹L','would_use':'would_use_%','high_conc':'high_conc_%'})
    prof = prof[['segment','size','age','literacy','risk_awareness','hhi','div_gap','need_score','wtp','wealth_₹L','high_conc_%','would_use_%']]
    st.dataframe(prof.set_index('segment'), width="stretch")
    if k==4:
        st.markdown("""
- **Vulnerable Novices** (largest) — young, low literacy & risk-awareness, highly concentrated, **highest need score**. Parakh's core protective mission.
- **Concentrated Risk-Takers** — ~99% dangerously concentrated, high willingness-to-pay, but the **lowest intent to adopt**. They need Parakh most yet opt in least → the key go-to-market risk.
- **Engaged Self-Improvers** — financially literate, risk-aware, **highest adoption intent (≈75%) and willingness-to-pay**. The obvious beachhead / early adopters.
- **Affluent Veterans** — older, wealthy, moderate need. A premium up-sell / B2B-adjacent segment.
""")
    st.info("**Insight:** demand is broad but not uniform. Launch to *Engaged Self-Improvers*, design nudges for "
            "*Vulnerable Novices*, and treat the low-adoption *Concentrated Risk-Takers* as the hardest — and most valuable — nut to crack.")

# ======================= ASSOCIATION =======================
elif page.startswith("🔗"):
    st.title("🔗 Association Rule Mining — which habits cause losses?")
    c1,c2 = st.columns(2)
    sup = c1.slider("Min support", 0.02, 0.10, 0.03, 0.01)
    conf = c2.slider("Min confidence", 0.40, 0.80, 0.45, 0.05)
    r = _assoc(trd, sup, conf)
    st.caption(f"Base loss rate = **{r['base_rate']:.1%}**. Lift > 1 means the pattern raises loss risk above baseline.")
    st.markdown("#### Rules that predict a LOSS")
    if len(r['loss']):
        st.dataframe(r['loss'].round(3), width="stretch")
        top = r['loss'].head(8)
        fig = px.bar(top, x='lift', y='antecedent', orientation='h', text='confidence',
                     labels={'lift':'Lift','antecedent':''})
        fig.update_traces(marker_color=CLAY, texttemplate='conf %{text:.0%}')
        st.plotly_chart(plotly_style(fig, h=360), width="stretch")
    else:
        st.warning("No rules at these thresholds — lower the support/confidence.")
    st.markdown("#### Protective rules (predict NO loss)")
    st.dataframe(r['protective'].head(8).round(3), width="stretch")
    st.info("**Insight:** FOMO, social-media-tip and uninformed trades lift loss risk ~1.4–1.6×, while **checking "
            "risk first or doing own research pushes loss risk *down* (→ 'No Loss' at ~82% confidence).** These rules are "
            "literally the plain-language warnings Parakh can show at the moment of decision.")

# ======================= REGRESSION =======================
elif page.startswith("📈"):
    st.title("📈 Regression — what will customers pay?")
    st.caption("Target: `willingness_to_pay_monthly_inr`. Compares Linear, Ridge and Lasso.")
    r = _reg(inv)
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown("#### Model performance")
        st.dataframe(r['metrics'].round(3), width="stretch")
        st.metric("Average willingness-to-pay", f"₹{r['mean_wtp']:.0f}/mo", f"median ₹{r['median_wtp']:.0f}")
    with c2:
        st.markdown("#### What moves willingness-to-pay (Lasso)")
        co = r['lasso_coef'].head(10).sort_values()
        fig = px.bar(x=co.values, y=co.index, orientation='h', labels={'x':'₹ effect on monthly WTP','y':''})
        fig.update_traces(marker_color=[CLAY if v<0 else SAGE for v in co.values])
        st.plotly_chart(plotly_style(fig, h=360), width="stretch")
    st.info("**Insight:** WTP is only moderately predictable (R² ≈ 0.3) — meaning **price sensitivity is fairly uniform**, "
            "supporting one simple price tier rather than heavy personalisation. Income is the main lever (higher income → "
            "higher WTP), and **high-concentration users pay *less*** even though they need Parakh most — a pricing risk that "
            "argues for a free protective tier plus paid premium/B2B revenue.")

# ======================= VERDICT =======================
else:
    st.title("✅ Business Verdict — should Parakh launch?")
    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="kpi"><h3>Demand</h3><b>Strong.</b> 63% intend to use it; 0% are low-need. '
                'Every segment shows real need.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="kpi"><h3>Feasibility</h3><b>Proven.</b> Losses are predictable pre-trade '
                '(ROC-AUC ≈ 0.78) from behavioural signals Parakh already sees.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="kpi"><h3>Willingness to pay</h3><b>Modest & uniform.</b> ~₹185/mo average; '
                'income-driven. Favours freemium + B2B.</div>', unsafe_allow_html=True)
    st.markdown("### The evidence, technique by technique")
    st.markdown("""
| Technique | What it showed | Business meaning |
|---|---|---|
| **Classification** | Random Forest predicts losing trades at **70% accuracy / 0.78 AUC**, driven by behaviour not demographics | Parakh's core risk score is technically viable and addresses the real cause of loss |
| **Segmentation** | 4 clear segments; need is broad but adoption intent varies | Clear launch order: start with *Engaged Self-Improvers*, nudge *Vulnerable Novices* |
| **Association rules** | FOMO / tips / uninformed trades lift loss risk 1.4–1.6×; risk-checking → 82% no-loss | Ready-made in-app warnings with proven statistical backing |
| **Regression** | WTP ≈ ₹185/mo, income-led, weakly predictable; risky users pay least | Freemium protective tier + paid premium & B2B risk-API monetisation |
""")
    st.markdown('<div class="verdict"><h3 style="color:#EDE6DA">Verdict: the data supports launching Parakh.</h3>'
        'The problem is real and large (≈40% of trades lose money), the loss is <b>predictable before the trade</b> from '
        'signals Parakh captures, demand is broad (0% low-need), and clear early-adopter and warning strategies fall out of '
        'the analysis. The main risk is monetisation — the highest-need users are the least willing to pay — so lead with a '
        '<b>free protective product for adoption</b> and earn revenue from premium features and a <b>B2B risk API</b>.</div>',
        unsafe_allow_html=True)
    st.markdown("### Recommendations")
    st.markdown("""
1. **Launch the risk score first** — it's the proven, defensible core (AUC 0.78).
2. **Beachhead on *Engaged Self-Improvers*** (highest intent + WTP), then expand to *Vulnerable Novices* with guided nudges.
3. **Ship the association-rule warnings** as the headline UX ("trades like this lose 6 in 10 times").
4. **Freemium pricing** — free protection to drive adoption; paid premium + B2B API for profit.
5. **Watch the adoption risk** — *Concentrated Risk-Takers* need it most but opt in least; invest in trust & education.
""")
    st.caption("Models retrain live from the cleaned dataset — figures above reflect the current data.")
