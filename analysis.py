"""Parakh — shared analysis functions (used by the Streamlit app and standalone runs).
Every model is trained live from the cleaned CSVs so results are fully reproducible."""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------- data ----------------
def load_data(path="data"):
    inv = pd.read_csv(f"{path}/investors.csv")
    trd = pd.read_csv(f"{path}/trades.csv")
    return inv, trd

# ---------------- 1. CLASSIFICATION ----------------
CLF_CAT = ['trade_type','asset_class','decision_basis','self_reported_risk_appetite','city_tier','income_bracket']
CLF_NUM = ['trade_amount_inr','market_event_flag','checked_risk_before_trade','uninformed_trade_flag',
           'high_risk_trade_flag','age','years_investing','financial_literacy_score',
           'risk_awareness_score_out_of_10','uses_stop_loss','simulates_before_trading',
           'knows_what_var_is','checks_sector_concentration','checks_portfolio_correlation']

def run_classification(trd, test_size=0.25, seed=42):
    X = pd.get_dummies(trd[CLF_CAT+CLF_NUM], columns=CLF_CAT, drop_first=True)
    y = trd['loss_occurred']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=test_size,random_state=seed,stratify=y)
    sc = StandardScaler().fit(Xtr); Xtr_s=sc.transform(Xtr); Xte_s=sc.transform(Xte)
    models = {
        'KNN': (KNeighborsClassifier(n_neighbors=15), True),
        'Decision Tree': (DecisionTreeClassifier(max_depth=6, random_state=seed), False),
        'Random Forest': (RandomForestClassifier(n_estimators=250, max_depth=10, random_state=seed, n_jobs=-1), False),
        'Gradient Boosted Trees': (GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=seed), False),
    }
    rows, roc, cms, fitted = [], {}, {}, {}
    for name,(mdl,scale) in models.items():
        a,b = (Xtr_s,Xte_s) if scale else (Xtr.values,Xte.values)
        mdl.fit(a,ytr); p=mdl.predict(b); pr=mdl.predict_proba(b)[:,1]
        rows.append(dict(Model=name, Accuracy=accuracy_score(yte,p), Precision=precision_score(yte,p),
                         Recall=recall_score(yte,p), F1=f1_score(yte,p), ROC_AUC=roc_auc_score(yte,pr)))
        fpr,tpr,_ = roc_curve(yte,pr); roc[name]=(fpr,tpr,roc_auc_score(yte,pr))
        cms[name]=confusion_matrix(yte,p); fitted[name]=mdl
    metrics = pd.DataFrame(rows).set_index('Model').round(4)
    rf = fitted['Random Forest']
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return dict(metrics=metrics, roc=roc, cms=cms, importance=imp, base_rate=float(y.mean()),
                best=metrics['F1'].idxmax(), n=len(X))

# ---------------- 2. SEGMENTATION ----------------
SEG_FEAT = ['age','financial_literacy_score','risk_awareness_score_out_of_10','herfindahl_index',
            'diversification_gap','num_total_instruments','equity_to_total_wealth_pct',
            'willingness_to_pay_monthly_inr','parakh_need_score','years_investing']
SEG_NAMES = {0:"Affluent Veterans", 1:"Vulnerable Novices",
             2:"Concentrated Risk-Takers", 3:"Engaged Self-Improvers"}

def run_segmentation(inv, k=4, seed=42):
    Xs = StandardScaler().fit_transform(inv[SEG_FEAT])
    scan = []
    for kk in range(2,7):
        km = KMeans(n_clusters=kk, random_state=seed, n_init=10).fit(Xs)
        scan.append(dict(k=kk, inertia=km.inertia_, silhouette=silhouette_score(Xs, km.labels_)))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(Xs)
    d = inv.copy(); d['cluster']=km.labels_
    prof = d.groupby('cluster').agg(
        size=('investor_id','size'), age=('age','mean'), literacy=('financial_literacy_score','mean'),
        risk_awareness=('risk_awareness_score_out_of_10','mean'), hhi=('herfindahl_index','mean'),
        div_gap=('diversification_gap','mean'), need_score=('parakh_need_score','mean'),
        wtp=('willingness_to_pay_monthly_inr','mean'), wealth=('total_estimated_wealth_inr','mean'),
        high_conc=('high_concentration_flag','mean'), would_use=('would_use_parakh_simulator','mean')).round(2)
    prof['segment'] = [SEG_NAMES.get(i, f"Cluster {i}") for i in prof.index]
    pca = PCA(n_components=2).fit_transform(Xs)
    d['pc1'], d['pc2'] = pca[:,0], pca[:,1]
    d['segment'] = d['cluster'].map(lambda i: SEG_NAMES.get(i, f'Cluster {i}'))
    return dict(scan=pd.DataFrame(scan), profile=prof, points=d[['pc1','pc2','segment','cluster']], k=k)

# ---------------- 3. ASSOCIATION RULES ----------------
def run_association(trd, min_support=0.03, min_conf=0.45):
    from mlxtend.frequent_patterns import apriori, association_rules
    it = pd.DataFrame(index=trd.index)
    for v in trd['decision_basis'].unique(): it[f'basis={v}'] = (trd['decision_basis']==v)
    for v in trd['asset_class'].unique():    it[f'asset={v}'] = (trd['asset_class']==v)
    it['risk_checked=No']  = (trd['checked_risk_before_trade']==0)
    it['risk_checked=Yes'] = (trd['checked_risk_before_trade']==1)
    it['uninformed=Yes']   = (trd['uninformed_trade_flag']==1)
    it['market_event=Yes'] = (trd['market_event_flag']==1)
    it['OUTCOME=Loss']     = (trd['loss_occurred']==1)
    it['OUTCOME=NoLoss']   = (trd['loss_occurred']==0)
    it = it.astype(bool)
    freq = apriori(it, min_support=min_support, use_colnames=True, max_len=3)
    rules = association_rules(freq, metric='confidence', min_threshold=min_conf)
    has_out = lambda s: any('OUTCOME' in x for x in s)
    fmt = lambda s: ", ".join(sorted(s))
    loss = rules[(rules['consequents']==frozenset({'OUTCOME=Loss'})) & (~rules['antecedents'].apply(has_out))].copy()
    prot = rules[(rules['consequents']==frozenset({'OUTCOME=NoLoss'})) & (~rules['antecedents'].apply(has_out))].copy()
    for df in (loss, prot):
        df['antecedent'] = df['antecedents'].apply(fmt)
    loss = loss.sort_values('lift', ascending=False)
    prot = prot.sort_values('lift', ascending=False)
    cols = ['antecedent','support','confidence','lift']
    return dict(loss=loss[cols].round(3).reset_index(drop=True),
                protective=prot[cols].round(3).reset_index(drop=True),
                base_rate=float(trd['loss_occurred'].mean()))

# ---------------- 4. REGRESSION ----------------
REG_CAT = ['city_tier','income_bracket','self_reported_risk_appetite']
REG_NUM = ['age','financial_literacy_score','risk_awareness_score_out_of_10','herfindahl_index',
           'diversification_gap','parakh_need_score','equity_to_total_wealth_pct','years_investing',
           'high_concentration_flag','num_total_instruments']

def run_regression(inv, seed=42):
    d = inv.copy(); d['log_wealth']=np.log1p(d['total_estimated_wealth_inr'])
    num = REG_NUM + ['log_wealth']
    X = pd.get_dummies(d[REG_CAT+num], columns=REG_CAT, drop_first=True)
    y = d['willingness_to_pay_monthly_inr']
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.25,random_state=seed)
    sc = StandardScaler().fit(Xtr); a=sc.transform(Xtr); b=sc.transform(Xte)
    rows, coefs = [], {}
    for name,mdl in [('Linear',LinearRegression()),('Ridge',Ridge(alpha=10.0)),('Lasso',Lasso(alpha=5.0))]:
        mdl.fit(a,ytr); p=mdl.predict(b)
        rows.append(dict(Model=name, R2=r2_score(yte,p),
                         RMSE=mean_squared_error(yte,p)**0.5, MAE=mean_absolute_error(yte,p)))
        coefs[name]=pd.Series(mdl.coef_, index=X.columns)
    metrics = pd.DataFrame(rows).set_index('Model').round(3)
    lasso_coef = coefs['Lasso']; lasso_coef = lasso_coef[lasso_coef.abs()>0.5].sort_values(key=abs, ascending=False)
    return dict(metrics=metrics, lasso_coef=lasso_coef.round(2), mean_wtp=float(y.mean()), median_wtp=float(y.median()))
