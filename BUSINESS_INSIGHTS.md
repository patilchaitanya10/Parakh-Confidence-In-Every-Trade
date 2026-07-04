# Parakh — Market Validation: Findings, Risks & Verdict

*Machine-learning analysis of 1,000 investors and 7,876 trades. All figures are produced live by the dashboard.*

---

## The question
Does the data support launching **Parakh**, a pre-trade risk tool that shows retail investors how much a trade could lose them — in plain rupees — **before** they act?

## Headline answer
**Yes.** The problem is real and large, the loss is **predictable before the trade happens**, demand is broad, and clear go-to-market and warning strategies fall directly out of the analysis. The one genuine risk is monetisation.

---

## 1. Classification — can we predict a losing trade? (the product's engine)

Target: `loss_occurred` (does a trade lose money?). Only pre-trade features are used, so there is no leakage.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| KNN | 0.649 | 0.573 | 0.496 | 0.532 | 0.707 |
| Decision Tree | 0.654 | 0.584 | 0.487 | 0.531 | 0.697 |
| **Random Forest** | **0.702** | **0.659** | **0.538** | **0.592** | **0.781** |
| Gradient Boosted Trees | 0.670 | 0.608 | 0.506 | 0.552 | 0.718 |

**Random Forest wins** (70% accuracy, ROC-AUC 0.78) against a 40.2% base loss rate — a clear, honest lift over guessing. The top drivers are **behavioural, not demographic**: trade size, risk-awareness, the uninformed/high-risk flags, and decision basis (own research is protective). 

> **Business meaning:** Parakh's core promise — a risk score shown before you trade — is technically feasible and targets the true cause of loss. This is the strongest single piece of evidence for the product.

## 2. Customer Segmentation — who are the customers? (K-Means, k = 4)

| Segment | Size | Profile | Adoption intent | Role for Parakh |
|---|---|---|---|---|
| **Vulnerable Novices** | 329 | Young, low literacy & awareness, concentrated, **highest need** | ~60% | Core protective mission; guide with nudges |
| **Concentrated Risk-Takers** | 185 | ~99% dangerously concentrated, high WTP, **lowest intent** | ~59% | Need it most, opt in least → hardest + most valuable |
| **Engaged Self-Improvers** | 220 | Literate, risk-aware, **highest intent (~75%) & WTP (~₹222)** | ~75% | **Beachhead / early adopters** |
| **Affluent Veterans** | 266 | Older, wealthy (~₹82L), moderate need | ~61% | Premium up-sell / B2B-adjacent |

> **Business meaning:** demand is broad but not uniform. There is an obvious launch order — win the *Engaged Self-Improvers* first, then expand.

## 3. Association Rule Mining — which habits cause losses?

Base loss rate = 40.2%. Lift > 1 means the pattern raises loss risk.

**Loss-causing patterns**
- `FOMO` → Loss — confidence **61%**, lift **1.51**
- `market event + uninformed` → Loss — 61%, lift 1.52
- `uninformed trade` → Loss — 58%, lift 1.44
- `Social Media Tip` → Loss — 56%, lift 1.38

**Protective patterns**
- `checked risk first` → No Loss — **83%**, lift 1.38
- `Own Research` → No Loss — **81%**, lift 1.36

> **Business meaning:** these rules are literally the **plain-language warnings** Parakh can show at the moment of decision ("trades like this lose ~6 in 10 times"). The protective rules prove that Parakh's intended behaviour change actually works.

## 4. Regression — what will customers pay? (Linear / Ridge / Lasso)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear | 0.326 | 234 | 154 |
| Ridge | 0.325 | 235 | 154 |
| Lasso | 0.306 | 238 | 154 |

Average willingness-to-pay ≈ **₹185/month** (median ₹125). Main drivers (Lasso): **income** (higher income → higher WTP), conservative risk appetite and risk-awareness push WTP up; **high-concentration users pay less**.

> **Business meaning:** price sensitivity is fairly **uniform** (R² ≈ 0.3), so one simple price tier beats heavy personalisation. Crucially, the users who need Parakh most (concentrated, risky) are willing to pay the least — a monetisation risk.

---

## Market demand, drivers, and risks

- **Demand:** strong and broad — 63% intend to use Parakh and **0% of investors are low-need**.
- **Key drivers of the problem:** behaviour at the point of decision (FOMO, tips, not checking risk), not income, age or experience.
- **Risks:**
  1. **Monetisation** — highest-need users have the lowest willingness-to-pay.
  2. **Adoption** — *Concentrated Risk-Takers* need it most but opt in least.
  3. **Model ceiling** — behaviour is noisy; AUC 0.78 is useful, not perfect, so frame Parakh as guidance, not a guarantee.

## Verdict: **the data supports launching Parakh.**

The problem is real (≈40% of trades lose money), predictable in advance, and broadly felt. Lead with a **free protective product** to drive adoption, and monetise through **premium features and a B2B risk-API** rather than charging the most vulnerable users.

## Recommendations
1. **Launch the risk score first** — the proven, defensible core (AUC 0.78).
2. **Beachhead on Engaged Self-Improvers**, then expand to Vulnerable Novices with guided nudges.
3. **Ship the association-rule warnings** as the headline UX.
4. **Freemium pricing** — free protection for reach; paid premium + B2B API for profit.
5. **Invest in trust/education** to convert the low-adoption, high-need Concentrated Risk-Takers.

## Which techniques we used, and what we'd add
Used all four requested: **Classification, Segmentation, Association Rules, Regression**.
From the course guide we also recommend keeping two supporting pieces (already in the dashboard): **ROC-AUC model scoring** (grades classifiers at every threshold) and the **elbow + silhouette** method (justifies the number of segments), plus **PCA** to visualise clusters in 2-D. These strengthen the rigor without adding a new technique.
