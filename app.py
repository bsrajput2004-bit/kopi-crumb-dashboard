"""
Kopi & Crumb - consumer survey analytics dashboard (Streamlit)

Run locally:   streamlit run app.py
The app cleans the survey data, runs classification, clustering,
association rule mining and regression, and judges whether the
business is sustainable. Upload a real survey CSV with the same
columns in the sidebar to rerun everything on real data.
"""
import itertools

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.stats import chi2_contingency
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor, VotingClassifier)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, calinski_harabasz_score, confusion_matrix, davies_bouldin_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(page_title="Kopi & Crumb - Survey Analytics", page_icon="☕", layout="wide")

DATA_FILE = "kopi_crumb_survey_5000.csv"
COFFEE = ["#6F4E37", "#C8A27C", "#2E7D6B", "#D9822B", "#8C6BB1", "#4C78A8", "#B5B5B5", "#A0522D"]
GOOD, WARN, BAD = "#2E7D32", "#E09A1A", "#C62828"
px.defaults.template = "none"  # Streamlit styles the charts itself; this also avoids a Plotly template error on some servers
px.defaults.color_discrete_sequence = COFFEE

# --------------------------------------------------------------------------------------
# Questionnaire (22 questions, each mapped to the algorithm it feeds)
# --------------------------------------------------------------------------------------
QUESTIONNAIRE = pd.DataFrame([
    ("A. About you", "Q1", "What is your age group?", "18-24 / 25-34 / 35-44 / 45-54 / 55+", "Clustering, classification"),
    ("A. About you", "Q2", "What is your gender?", "Female / Male / Prefer not to say", "Clustering"),
    ("A. About you", "Q3", "What is your occupation?", "Student / Working professional / Self-employed or business owner / Homemaker / Retired", "Clustering, classification"),
    ("A. About you", "Q4", "What is your monthly household income?", "Below AED 10,000 / 10,000-19,999 / 20,000-34,999 / 35,000-49,999 / 50,000+", "Clustering, regression"),
    ("A. About you", "Q5", "Which area of Dubai do you live or work in?", "13 areas (Dubai Marina/JBR, JLT, Downtown/Business Bay, ...)", "Clustering, store location"),
    ("A. About you", "Q6", "How many people live in your household?", "1 / 2 / 3-4 / 5+", "Regression, bulk orders"),
    ("B. Current habits", "Q7", "How often do you visit a cafe or bakery?", "Never / Less than once a month / 1-3 times a month / Weekly / Several times a week", "Classification, clustering"),
    ("B. Current habits", "Q8", "How much do you usually spend per visit? (AED)", "Open number", "Regression, clustering"),
    ("B. Current habits", "Q9", "How do you usually buy baked goods?", "Walk-in / Delivery apps / A bakery's own app or website / WhatsApp or Instagram / I don't buy them", "Classification"),
    ("B. Current habits", "Q10", "Do you have any dietary needs when buying bakes? (pick all)", "None / Eggless / Vegan / Low sugar / Gluten-free", "Classification, menu planning"),
    ("B. Current habits", "Q11", "Have you tried Nanyang kopi or coffee with condensed milk?", "Yes, I love it / Yes, it's okay / Yes, I didn't like it / Never tried it", "Classification, clustering"),
    ("C. What matters", "Q12", "Rate how important each is (1-5): freshness, price, authentic taste, customising my order, ambience, speed and convenience, coffee quality", "1 = not important ... 5 = very important", "Clustering, classification"),
    ("D. The concept", "Q13", "How appealing is the Kopi & Crumb concept to you?", "1-5", "Regression"),
    ("D. The concept", "Q14", "Given the opportunity, would you visit or order from Kopi & Crumb?", "Definitely yes / Probably yes / Not sure / Probably not / Definitely not", "CLASSIFICATION TARGET"),
    ("D. The concept", "Q15", "Which of these would you buy or use? (pick all)", "Butter croissant / Kaya toast set / Kaya butter croissant / Pandan cake with custom icing / Signature kopi (hot) / Kopi peng (iced) / Regular coffee / Kopi tiramisu slice / Whole tiramisu or bulk cake / App with full customisation / Home delivery / Weekly pre-order box", "ASSOCIATION RULES"),
    ("D. The concept", "Q16", "How much icing do you usually prefer?", "Less than standard / Standard / Extra / On the side / No preference", "Product design"),
    ("D. The concept", "Q17", "How likely are you to download an app to customise and order?", "1-5", "Classification, clustering"),
    ("D. The concept", "Q18", "How much would you expect to spend at Kopi & Crumb in a month? (AED)", "Open number", "REGRESSION TARGET"),
    ("D. The concept", "Q19", "What is the most you'd pay for a signature kopi + butter croissant combo? (AED)", "Open number", "Pricing"),
    ("D. The concept", "Q20", "Would you place a bulk order (birthday, office, event) in the next 12 months?", "Yes / Maybe / No", "Regression, bulk demand"),
    ("D. The concept", "Q21", "How many times a month would you visit or order?", "Open number", "Sustainability (demand)"),
    ("D. The concept", "Q22", "What might stop you from trying it? (pick all)", "Price / Location / I already have a favourite / I don't like condensed milk in coffee / I don't want another app / I prefer bakes I already know", "Segment profiles, strategy"),
], columns=["Section", "No.", "Question", "Answer options", "Feeds"])

AGE = {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4}
INC = {"Below AED 10,000": 0, "AED 10,000-19,999": 1, "AED 20,000-34,999": 2, "AED 35,000-49,999": 3, "AED 50,000+": 4}
HH = {"1": 0, "2": 1, "3-4": 2, "5+": 3}
FREQ = {"Never": 0, "Less than once a month": 1, "1-3 times a month": 2, "Weekly": 3, "Several times a week": 4}
KOPI = {"Yes, I didn't like it": -1, "Never tried it": 0, "Yes, it's okay": 1, "Yes, I love it": 2}
INTENT = {"Definitely not": 0, "Probably not": 1, "Not sure": 2, "Probably yes": 3, "Definitely yes": 4}
INTENT_ORDER = list(INTENT)[::-1]
PRODUCTS = {
    "Q15_butter_croissant": "Butter croissant", "Q15_kaya_toast_set": "Kaya toast set",
    "Q15_kaya_butter_croissant": "Kaya butter croissant", "Q15_pandan_cake_custom_icing": "Pandan cake, custom icing",
    "Q15_signature_kopi_hot": "Signature kopi (hot)", "Q15_kopi_peng_iced": "Kopi peng (iced)",
    "Q15_regular_coffee": "Regular coffee", "Q15_kopi_tiramisu_slice": "Kopi tiramisu slice",
    "Q15_whole_tiramisu_bulk_cake": "Whole tiramisu / bulk cake", "Q15_app_full_customisation": "App with full customisation",
    "Q15_home_delivery": "Home delivery", "Q15_weekly_preorder_box": "Weekly pre-order box",
}
SERVICES = ["App with full customisation", "Home delivery", "Weekly pre-order box"]
BARRIERS = {
    "Q22_barrier_price": "Price", "Q22_barrier_location": "Location", "Q22_barrier_have_favourite": "Already has a favourite",
    "Q22_barrier_dislike_condensed_milk": "Dislikes condensed milk", "Q22_barrier_no_new_app": "Doesn't want another app",
    "Q22_barrier_prefer_familiar_bakes": "Prefers bakes they know", "Q22_barrier_none": "Nothing would stop them",
}
RATING_COLS = ["Q12a_imp_freshness", "Q12b_imp_price", "Q12c_imp_authentic_taste", "Q12d_imp_customisation",
               "Q12e_imp_ambience", "Q12f_imp_speed_convenience", "Q12g_imp_coffee_quality"]
REQUIRED = ["Q1_age_group", "Q4_household_income_aed_month", "Q7_cafe_visit_frequency", "Q8_spend_per_visit_aed",
            "Q11_kopi_experience", "Q14_visit_intent", "Q18_expected_monthly_spend_aed", "Q21_expected_visits_per_month"] \
    + RATING_COLS + list(PRODUCTS)


# --------------------------------------------------------------------------------------
# Model optimisation results (tune_v2.py + tune_v2_finish.py, run offline).
# Protocol: stratified 75/25 split made before tuning; every configuration scored by 5-fold CV on the training
# data only, with scaling and feature selection inside the pipeline; finalists re-ranked by repeated CV (5 x 3);
# the 25% test set is used once. Too slow to repeat live, so the results are stored here.
# --------------------------------------------------------------------------------------
TUNING = pd.DataFrame([('Logistic Regression', 'Exhaustive grid', 840, 4200, 'standard scaling, best 30 features, C=0.01, L1 share 0.25', 0.7342, 0.7431, 0.7379, 0.7392), ('Decision Tree', 'Exhaustive grid', 288, 1440, 'entropy, depth 6, min leaf 30, balanced classes', 0.6235, 0.7048, 0.7005, 0.63), ('K-Nearest Neighbours', 'Exhaustive grid', 432, 2160, 'min-max scaling, best 10 features, k=71, Euclidean', 0.62, 0.7271, 0.7315, 0.6275), ('Support Vector Machine', 'Exhaustive grid', 46, 230, 'standard scaling, best 20 features, linear kernel, C=1', 0.7162, 0.7363, 0.7369, 0.7188), ('Hist Gradient Boosting', 'Exhaustive grid', 324, 1620, 'learning rate 0.02, 400 iterations, depth 3, min leaf 20, L2=1', 0.7073, 0.7347, 0.73, 0.7294), ('Random Forest', 'Randomised (50 of 2160)', 50, 250, '500 trees, depth 8, sqrt features, min leaf 10', 0.7179, 0.7287, 0.7254, 0.7498), ('Gradient Boosting', 'Randomised (40 of 864)', 40, 200, '200 trees, learning rate 0.05, depth 2, min leaf 10, sqrt features', 0.7282, 0.7334, 0.733, 0.7416), ('Soft-Voting Ensemble', 'Ensemble of top 3', 1, 5, 'average of tuned LR, HGB and GB', None, 0.7347, 0.7333, None), ('Stacking Ensemble', 'Ensemble of top 3', 1, 5, 'LR on top of tuned LR, HGB and GB', None, 0.7377, 0.7367, None)], columns=["Model", "Search", "Configurations", "Fits", "Selected settings", "CV accuracy (default)",
                                        "CV accuracy (tuned)", "Repeated CV accuracy", "Test accuracy (default)"])
TUNING_TOTAL = {"configurations": 2022, "fits": 10110}
SELECTED_MODEL = "Logistic Regression"   # highest repeated-CV accuracy, and the simplest model
ACCURACY_CEILING = 0.806   # test-set accuracy of an oracle that knows every hidden trait used to simulate the data
REG_TUNING = pd.DataFrame([('Linear Regression', 1, 5, '-', 0.4461, 0.4482, 0.4642), ('Ridge', 8, 40, 'alpha=300', 0.4461, 0.4491, 0.4631), ('Lasso', 8, 40, 'alpha=3', 0.4466, 0.455, 0.4591), ('Elastic Net', 18, 90, 'alpha=0.3, l1_ratio=0.8', 0.426, 0.4505, 0.4634), ('K-Nearest Neighbours', 12, 60, 'n_neighbors=20, weights=distance', 0.2952, 0.3782, 1.0), ('Random Forest', 60, 300, 'max_depth=12, max_features=0.5, min_samples_leaf=8', 0.5847, 0.5781, 0.7086), ('Hist Gradient Boosting', 81, 405, 'learning_rate=0.05, max_depth=3, max_iter=100, min_samples_leaf=100', 0.5686, 0.5713, 0.6212)], columns=["Model", "Configurations", "Fits", "Selected settings", "Test R² (default)",
                                            "CV R² (tuned)", "Training R² (tuned)"])
REG_TOTAL = {"configurations": 188, "fits": 940}
CLUSTER_SEARCH = 126          # 3 scalers x 6 algorithms x k = 2..8
RULE_THRESHOLDS_TESTED = 100  # support x confidence x lift grid, validated on unseen baskets
BEST_THRESHOLD = 0.50


def pct(x, d=1):
    return f"{x * 100:.{d}f}%"


def aed(x):
    return f"AED {x:,.0f}"


# --------------------------------------------------------------------------------------
# 1. Cleaning
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Cleaning the survey data...")
def clean(raw: pd.DataFrame):
    df = raw.copy()
    log = []
    id_cols = [c for c in ["respondent_id"] if c in df.columns]
    dup = df.drop(columns=id_cols).duplicated()
    log.append(("Duplicate submissions removed", int(dup.sum())))
    df = df[~dup].copy()

    area_map = {"dubai marina": "Dubai Marina / JBR", "Marina": "Dubai Marina / JBR", "JBR ": "Dubai Marina / JBR",
                "jlt": "JLT", "JLT ": "JLT", "Jumeirah Lake Towers": "JLT", "Business bay": "Downtown / Business Bay",
                "downtown": "Downtown / Business Bay", "Downtown Dubai": "Downtown / Business Bay",
                "al barsha": "Al Barsha", "Barsha": "Al Barsha", "JVC": "JVC / Arjan", "jvc ": "JVC / Arjan",
                "Deira": "Deira / Bur Dubai", "bur dubai": "Deira / Bur Dubai"}
    gender_map = {"M": "Male", "female": "Female", "prefer not to say": "Prefer not to say"}
    fixed = 0
    if "Q5_area" in df:
        fixed += int(df["Q5_area"].isin(list(area_map)).sum())
        df["Q5_area"] = df["Q5_area"].replace(area_map)
    if "Q2_gender" in df:
        fixed += int(df["Q2_gender"].isin(list(gender_map)).sum())
        df["Q2_gender"] = df["Q2_gender"].replace(gender_map)
    log.append(("Inconsistent text standardised (area spellings, gender codes)", fixed))

    rate_cols = RATING_COLS + [c for c in ["Q13_concept_appeal", "Q17_app_download_likelihood"] if c in df]
    for c in rate_cols + ["Q8_spend_per_visit_aed", "Q18_expected_monthly_spend_aed",
                          "Q19_max_price_kopi_croissant_combo_aed", "Q21_expected_visits_per_month"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    bad = 0
    for c in ["Q8_spend_per_visit_aed", "Q21_expected_visits_per_month"]:
        m = df[c] < 0
        bad += int(m.sum())
        df.loc[m, c] = np.nan
    for c in rate_cols:
        m = df[c].notna() & ~df[c].between(1, 5)
        bad += int(m.sum())
        df.loc[m, c] = np.nan
    log.append(("Impossible values blanked (negative spend, -1 visits, ratings outside 1-5)", bad))

    sl = df[rate_cols].eq(5).all(axis=1)
    if "completion_time_sec" in df:
        sl &= pd.to_numeric(df["completion_time_sec"], errors="coerce") < 120
    log.append(("Straight-liners removed (all 5s and finished in under 2 minutes)", int(sl.sum())))
    df = df[~sl].copy()

    caps = {"Q8_spend_per_visit_aed": 300, "Q18_expected_monthly_spend_aed": 2000,
            "Q19_max_price_kopi_croissant_combo_aed": 100, "Q21_expected_visits_per_month": 31}
    out = 0
    for c, cap in caps.items():
        if c in df:
            m = df[c] > cap
            out += int(m.sum())
            df.loc[m, c] = np.nan
    log.append(("Outliers beyond realistic limits blanked (>AED 300/visit, >AED 2,000/month, combo >AED 100, >31 visits)", out))

    c1 = (df["Q14_visit_intent"] == "Definitely not") & (df["Q18_expected_monthly_spend_aed"] > 300)
    c2 = (df["Q7_cafe_visit_frequency"] == "Never") & (df["Q8_spend_per_visit_aed"] > 85)
    df.loc[c1, "Q18_expected_monthly_spend_aed"] = np.nan
    df.loc[c2, "Q8_spend_per_visit_aed"] = np.nan
    log.append(("Contradictory answers neutralised (spend blanked)", int(c1.sum() + c2.sum())))

    blanks = int(df.isna().sum().sum())
    df["Q4_household_income_aed_month"] = df["Q4_household_income_aed_month"].fillna("Not disclosed")
    for c in ["Q2_gender", "Q5_area", "Q6_household_size", "Q11_kopi_experience", "Q16_icing_preference",
              "Q3_occupation", "Q9_usual_purchase_channel", "Q20_bulk_order_intent_12m"]:
        if c in df and df[c].notna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])
    for c in rate_cols:
        df[c] = df[c].fillna(df[c].median())
    df["Q8_spend_per_visit_aed"] = df["Q8_spend_per_visit_aed"].fillna(df["Q8_spend_per_visit_aed"].median())
    for c in ["Q18_expected_monthly_spend_aed", "Q19_max_price_kopi_croissant_combo_aed", "Q21_expected_visits_per_month"]:
        if c in df:
            df[c] = df[c].fillna(df.groupby("Q14_visit_intent")[c].transform("median")).fillna(df[c].median())
    log.append(("Blanks filled (median or most common answer; income kept as 'Not disclosed')", blanks))

    df = df.reset_index(drop=True)
    df["interested"] = df["Q14_visit_intent"].isin(["Probably yes", "Definitely yes"]).astype(int)
    return df, pd.DataFrame(log, columns=["Cleaning step", "Rows or cells affected"])


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the pre-concept answers (Q1-Q12) into model-ready numbers."""
    E = pd.DataFrame(index=df.index)
    E["age"] = df["Q1_age_group"].map(AGE)
    E["income"] = df["Q4_household_income_aed_month"].map(INC)
    E["income_not_disclosed"] = E["income"].isna().astype(int)
    E["income"] = E["income"].fillna(E["income"].median())
    if "Q6_household_size" in df:
        E["household_size"] = df["Q6_household_size"].astype(str).str.replace(".0", "", regex=False).map(HH)
    E["cafe_frequency"] = df["Q7_cafe_visit_frequency"].map(FREQ)
    E["spend_per_visit"] = df["Q8_spend_per_visit_aed"]
    E["kopi_experience"] = df["Q11_kopi_experience"].map(KOPI)
    for c in [c for c in df.columns if c.startswith("Q10_diet")]:
        E[c.replace("Q10_", "")] = df[c]
    for c in RATING_COLS:
        E[c.split("_imp_")[1] + "_importance"] = df[c]
    cats = [c for c in ["Q2_gender", "Q3_occupation", "Q5_area", "Q9_usual_purchase_channel"] if c in df]
    X = pd.concat([E, pd.get_dummies(df[cats], drop_first=True, dtype=int)], axis=1)
    X.columns = [str(c).replace(" ", "_").replace("/", "").replace(",", "").replace("(", "").replace(")", "")
                 .replace("'", "") for c in X.columns]
    return X.fillna(X.median(numeric_only=True)).fillna(0)


NICE = {"cafe_frequency": "Cafe visit frequency", "spend_per_visit": "Spend per visit", "kopi_experience": "Kopi experience",
        "customisation_importance": "Values customisation", "coffee_quality_importance": "Values coffee quality",
        "authentic_taste_importance": "Values authentic taste", "price_importance": "Price importance",
        "freshness_importance": "Values freshness", "ambience_importance": "Values ambience",
        "speed_convenience_importance": "Values convenience", "age": "Age", "income": "Income",
        "household_size": "Household size", "visit_intent": "Visit intent (Q14)", "concept_appeal": "Concept appeal (Q13)",
        "app_likelihood": "App likelihood (Q17)", "n_items_wanted": "Items wanted (Q15 count)", "bulk_intent": "Bulk-order intent (Q20)"}


# --------------------------------------------------------------------------------------
# 2. Classification
# --------------------------------------------------------------------------------------
def tuned_classifiers():
    """Each model with the settings selected by the optimisation search."""
    def P(scaler, k, model):
        return Pipeline([("scale", scaler), ("select", SelectKBest(f_classif, k=k)), ("m", model)])
    lr = lambda: P(StandardScaler(), 30, LogisticRegression(solver="saga", C=0.01, l1_ratio=0.25, max_iter=3000))
    hgb = lambda: HistGradientBoostingClassifier(learning_rate=0.02, max_iter=400, max_depth=3, min_samples_leaf=20,
                                                 l2_regularization=1.0, random_state=42)
    gb = lambda: GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=2, min_samples_leaf=10,
                                            max_features="sqrt", subsample=1.0, random_state=42)
    return {
        "Logistic Regression": lr,
        "Decision Tree": lambda: DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=30,
                                                        class_weight="balanced", random_state=42),
        "K-Nearest Neighbours": lambda: P(MinMaxScaler(), 10, KNeighborsClassifier(n_neighbors=71, p=2)),
        "Support Vector Machine": lambda: P(StandardScaler(), 20, SVC(kernel="linear", C=1)),
        "Hist Gradient Boosting": hgb,
        "Random Forest": lambda: RandomForestClassifier(n_estimators=500, max_depth=8, max_features="sqrt", min_samples_leaf=10,
                                                        random_state=42, n_jobs=-1),
        "Gradient Boosting": gb,
        "Soft-Voting Ensemble": lambda: VotingClassifier([("lr", lr()), ("hgb", hgb()), ("gb", gb())], voting="soft"),
    }


def _score(m, X):
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(X)


@st.cache_resource(show_spinner="Training the tuned classification models...")
def run_classification(df: pd.DataFrame):
    X = encode(df)
    y = df["interested"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    makers = tuned_classifiers()
    rows, rocs, fitted, preds = [], {}, {}, {}
    for name, make in makers.items():
        m = make().fit(Xtr, ytr)
        p = _score(m, Xte)
        yp = m.predict(Xte)
        rows.append({"Model": name, "Accuracy": accuracy_score(yte, yp), "Precision": precision_score(yte, yp, zero_division=0),
                     "Recall": recall_score(yte, yp), "F1": f1_score(yte, yp), "ROC AUC": roc_auc_score(yte, p)})
        fpr, tpr, _ = roc_curve(yte, p)
        rocs[name] = (fpr, tpr)
        fitted[name] = m
        preds[name] = yp
    metrics = pd.DataFrame(rows).sort_values(["Accuracy", "ROC AUC"], ascending=False).reset_index(drop=True)
    best = SELECTED_MODEL
    cm = confusion_matrix(yte, preds[best])
    fi = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).sort_values(ascending=False).head(12)
    lrp = fitted["Logistic Regression"]
    kept = X.columns[lrp.named_steps["select"].get_support()]
    coef = pd.Series(lrp.named_steps["m"].coef_[0], index=kept)
    coef = coef[coef != 0]
    coef = coef.reindex(coef.abs().sort_values(ascending=False).index).head(12)
    oof = cross_val_predict(makers[best](), X, y, cv=StratifiedKFold(5, shuffle=True, random_state=1),
                            method="predict_proba")[:, 1]
    # Model B: the same profile answers plus one reaction to the concept (Q13 concept appeal)
    XB = X.copy()
    XB["concept_appeal"] = df["Q13_concept_appeal"].values
    XBtr, XBte = XB.loc[Xtr.index], XB.loc[Xte.index]
    mb = Pipeline([("scale", MinMaxScaler()), ("select", SelectKBest(f_classif, k=20)),
                   ("m", LogisticRegression(solver="saga", l1_ratio=1.0, C=1, max_iter=3000))]).fit(XBtr, ytr)
    pb = mb.predict(XBte)
    model_b = {"Accuracy": accuracy_score(yte, pb), "ROC AUC": roc_auc_score(yte, mb.predict_proba(XBte)[:, 1]),
               "Precision": precision_score(yte, pb), "Recall": recall_score(yte, pb), "cm": confusion_matrix(yte, pb)}
    return {"metrics": metrics, "best": best, "rocs": rocs, "cm": cm, "fi": fi, "coef": coef, "oof": oof,
            "model_b": model_b, "n_test": len(yte), "base_rate": float(yte.mean())}


# --------------------------------------------------------------------------------------
# 3. Clustering
# --------------------------------------------------------------------------------------
CLU_FEATS = ["age", "income", "household_size", "cafe_frequency", "spend_per_visit", "kopi_experience",
             "freshness_importance", "price_importance", "authentic_taste_importance", "customisation_importance",
             "ambience_importance", "speed_convenience_importance", "coffee_quality_importance"]


@st.cache_data(show_spinner="Finding customer segments...")
def cluster_matrix(df: pd.DataFrame):
    E = encode(df)
    C = E[[c for c in CLU_FEATS if c in E]].copy()
    C["spend_per_visit"] = np.log1p(C["spend_per_visit"].clip(lower=0))
    if "Q17_app_download_likelihood" in df:
        C["app_likelihood"] = df["Q17_app_download_likelihood"].values
    return RobustScaler().fit_transform(C)   # robust scaling won the 126-solution validity search


@st.cache_data(show_spinner="Testing different numbers of segments...")
def choose_k(df: pd.DataFrame):
    Cs = cluster_matrix(df)
    rs = np.random.default_rng(0)
    samp = rs.choice(len(Cs), min(2500, len(Cs)), replace=False)
    out = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=42).fit(Cs)
        out.append({"k": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(Cs[samp], km.labels_[samp]),
                    "Davies-Bouldin": davies_bouldin_score(Cs, km.labels_), "Calinski-Harabasz": calinski_harabasz_score(Cs, km.labels_)})
    return pd.DataFrame(out)


@st.cache_data(show_spinner="Building segments...")
def run_clustering(df: pd.DataFrame, k: int):
    Cs = cluster_matrix(df)
    km = KMeans(k, n_init=20, random_state=42).fit(Cs)
    pca = PCA(2, random_state=0).fit_transform(Cs)
    return km.labels_, pca


def name_segments(prof: pd.DataFrame, overall: dict) -> dict:
    """Give each cluster a plain-English name from how it differs from the average respondent."""
    names = {}
    for _, r in prof.iterrows():
        tags = []
        if r["Tried kopi"] >= overall["kopi"] + 0.3:
            tags.append("kopi-familiar")
        if r["Weekly café-goers"] >= overall["weekly"] + 0.15:
            tags.append("frequent")
        elif r["Weekly café-goers"] <= overall["weekly"] - 0.2:
            tags.append("occasional")
        if r["Income index (0-4)"] >= overall["income"] + 0.5:
            tags.append("affluent")
        elif r["Price importance (1-5)"] >= overall["price"] + 0.35:
            tags.append("budget-minded")
        elif r["Price importance (1-5)"] <= overall["price"] - 0.5:
            tags.append("price-insensitive")
        if r["Share aged 45+"] >= 0.45:
            tags.append("older")
        elif r["Share aged 18-34"] >= 0.65:
            tags.append("young")
        if r["App likelihood (1-5)"] <= overall["app"] - 0.5:
            tags.append("offline")
        if not tags:
            tags = ["mainstream"]
        names[r["Cluster"]] = ", ".join(tags[:3]).capitalize()
    return names


# --------------------------------------------------------------------------------------
# 4. Association rules
# --------------------------------------------------------------------------------------
def _rules(fis, n, metric, thr):
    try:
        return association_rules(fis, metric=metric, min_threshold=thr)
    except TypeError:  # older mlxtend versions need num_itemsets
        return association_rules(fis, num_itemsets=n, metric=metric, min_threshold=thr)


@st.cache_data(show_spinner="Mining association rules...")
def run_association(df: pd.DataFrame, min_support: float, min_conf: float, strong_conf: float, min_lift: float = 1.0):
    basket = df.loc[df["interested"] == 1, list(PRODUCTS)].astype(bool)
    basket.columns = [PRODUCTS[c] for c in basket.columns]
    support = basket.mean().sort_values(ascending=False)
    fis = apriori(basket, min_support=min_support, use_colnames=True, max_len=3)
    if fis.empty:
        return support, pd.DataFrame(), pd.DataFrame()
    rules = _rules(fis, len(basket), "confidence", min(min_conf, strong_conf))
    rules = rules[(rules["lift"] > 1.0) & (rules["consequents"].apply(len) == 1)].copy()
    rules["If they want"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["They also want"] = rules["consequents"].apply(lambda s: next(iter(s)))
    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    menu = [i for i in basket.columns if i not in SERVICES]
    B = basket[menu].values.astype(int)
    strong = rules[(rules["confidence"] >= strong_conf) & (rules["antecedents"].apply(len) <= 2)
                   & rules["They also want"].isin(menu)
                   & rules["antecedents"].apply(lambda a: all(i in menu for i in a))]
    strong_list = [(list(a), next(iter(c))) for a, c in zip(strong["antecedents"], strong["consequents"])]
    chosen = B.sum(1)

    def score(S):
        idx = [menu.index(i) for i in S]
        served = B[:, idx].sum(1)
        cover = np.where(chosen > 0, served / np.maximum(chosen, 1), 0).mean()
        let_down = np.zeros(len(B), dtype=bool)
        for ante, cons in strong_list:
            if set(ante) <= set(S) and cons not in S:
                cols = [menu.index(i) for i in ante] + [menu.index(cons)]
                let_down |= B[:, cols].all(1).astype(bool)
        return cover, (served > 0).mean(), let_down.mean()

    sets = []
    for k in range(3, len(menu) + 1):
        best = None
        for S in itertools.combinations(menu, k):
            cv, anyc, ld = score(S)
            val = cv - 0.5 * ld
            if best is None or val > best[0]:
                best = (val, cv, anyc, ld, S)
        sets.append({"Menu size": k, "Wish-list served": best[1], "Get at least one item": best[2],
                     "Let down by a missing pair": best[3], "Items": ", ".join(best[4])})
    shown = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)]
    return support, shown, pd.DataFrame(sets)


# --------------------------------------------------------------------------------------
# 5. Regression
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Training spend models...")
def run_regression(df: pd.DataFrame):
    X = encode(df)
    X["visit_intent"] = df["Q14_visit_intent"].map(INTENT)
    if "Q13_concept_appeal" in df:
        X["concept_appeal"] = df["Q13_concept_appeal"]
    if "Q17_app_download_likelihood" in df:
        X["app_likelihood"] = df["Q17_app_download_likelihood"]
    X["n_items_wanted"] = df[list(PRODUCTS)].sum(axis=1)
    if "Q20_bulk_order_intent_12m" in df:
        X["bulk_intent"] = df["Q20_bulk_order_intent_12m"].map({"No": 0, "Maybe": 1, "Yes": 2}).fillna(0)
    y = df["Q18_expected_monthly_spend_aed"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    lin = LinearRegression().fit(Xtr, ytr)
    ridge = Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=300))]).fit(Xtr, ytr)
    rf = RandomForestRegressor(n_estimators=300, max_depth=12, max_features=0.5, min_samples_leaf=8, random_state=42,
                               n_jobs=-1).fit(Xtr, ytr)
    hgb = HistGradientBoostingRegressor(learning_rate=0.05, max_iter=100, max_depth=3, min_samples_leaf=100,
                                        random_state=42).fit(Xtr, ytr)
    lasso = Pipeline([("sc", StandardScaler()), ("m", Lasso(alpha=3, max_iter=20000))]).fit(Xtr, ytr)
    rows = []
    preds = {}
    for name, m in [("Linear Regression", lin), ("Ridge Regression (tuned)", ridge), ("Lasso (tuned)", lasso), ("Random Forest (tuned)", rf),
                    ("Hist Gradient Boosting (tuned)", hgb)]:
        p = m.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, "R²": r2_score(yte, p), "MAE (AED)": mean_absolute_error(yte, p),
                     "RMSE (AED)": float(np.sqrt(mean_squared_error(yte, p)))})
    keep = ["visit_intent", "bulk_intent", "cafe_frequency", "concept_appeal", "n_items_wanted", "kopi_experience",
            "app_likelihood", "spend_per_visit", "income", "price_importance", "age"]
    coef = pd.Series(lin.coef_, index=X.columns).reindex([k for k in keep if k in X.columns])
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
    return {"metrics": pd.DataFrame(rows), "coef": coef, "fi": fi, "yte": yte.values, "preds": preds}


# --------------------------------------------------------------------------------------
# Sidebar: data and assumptions
# --------------------------------------------------------------------------------------
st.sidebar.title("☕ Kopi & Crumb")
st.sidebar.caption("Consumer survey analytics · Dubai")
up = st.sidebar.file_uploader("Use your own survey CSV (same columns)", type="csv",
                              help="Leave empty to use the built-in synthetic survey of 5,000 respondents.")
try:
    raw = pd.read_csv(up) if up is not None else pd.read_csv(DATA_FILE)
except FileNotFoundError:
    st.error(f"Could not find {DATA_FILE}. Upload it to the same GitHub folder as app.py.")
    st.stop()
missing = [c for c in REQUIRED if c not in raw.columns]
if missing:
    st.error("This file is missing columns the analysis needs: " + ", ".join(missing))
    st.stop()

with st.sidebar.expander("Model settings", expanded=False):
    k_choice = st.slider("Number of customer segments (k)", 2, 8, 4)
    hi_thr = st.slider("High conversion tier starts at", 0.40, 0.90, 0.60, 0.05)
    lo_thr = st.slider("Low conversion tier ends at", 0.10, 0.50, 0.35, 0.05)
    min_sup = st.slider("Association: minimum support", 0.05, 0.30, 0.08, 0.01)
    min_conf = st.slider("Association: minimum confidence", 0.40, 0.90, 0.70, 0.05)
    min_lift = st.slider("Association: minimum lift", 1.0, 2.0, 1.3, 0.1)
    strong_conf = st.slider("A rule is 'must keep together' above confidence", 0.55, 0.95, 0.65, 0.05)
    target_cover = st.slider("Launch menu should serve at least this much of a wish-list", 60, 95, 80, 5, format="%d%%") / 100

with st.sidebar.expander("Business assumptions (one café in JLT)", expanded=True):
    ticket = st.number_input("Average ticket per order (AED)", 15.0, 120.0, 38.0, 1.0)
    cogs = st.slider("Cost of goods (% of sales)", 20, 50, 32, 1, format="%d%%") / 100
    rent = st.number_input("Rent per month (AED)", 0, 200_000, 25_000, 1_000)
    staff = st.number_input("Staff per month (AED)", 0, 200_000, 31_000, 1_000)
    other = st.number_input("Utilities, marketing, licences per month (AED)", 0, 100_000, 16_000, 1_000)
    setup = st.number_input("Setup cost (AED)", 100_000, 3_000_000, 750_000, 25_000)
    deliv_share = st.slider("Share of orders through delivery apps", 0, 60, 20, 5, format="%d%%") / 100
    commission = st.slider("Delivery app commission", 0, 40, 28, 1, format="%d%%") / 100
    reach = st.number_input("People you can get to try it (first year)", 500, 50_000, 4_500, 250,
                            help="How many different people in the area will visit at least once. "
                                 "The survey then tells us how many of them become regulars.")
    st.caption("Intent discount (people overstate in surveys)")
    w_dy = st.slider("'Definitely yes' who really become regulars", 30, 100, 70, 5, format="%d%%") / 100
    w_py = st.slider("'Probably yes' who really become regulars", 0, 70, 30, 5, format="%d%%") / 100
    w_ns = st.slider("'Not sure' who really become regulars", 0, 30, 5, 1, format="%d%%") / 100

# --------------------------------------------------------------------------------------
# Run everything
# --------------------------------------------------------------------------------------
df, clean_log = clean(raw)
cls = run_classification(df)
df["p_interest"] = cls["oof"]
df["Conversion tier"] = np.select([df["p_interest"] >= hi_thr, df["p_interest"] < lo_thr], ["High", "Low"], "Medium")
k_table = choose_k(df)
labels, pca_xy = run_clustering(df, k_choice)
df["cluster"] = labels
E_all = encode(df)

prof_rows = []
for c in sorted(df["cluster"].unique()):
    s = df[df["cluster"] == c]
    e = E_all.loc[s.index]
    prof_rows.append({
        "Cluster": c, "Share of sample": len(s) / len(df), "Interested": s["interested"].mean(),
        "High tier": (s["Conversion tier"] == "High").mean(),
        "Median spend / month (AED)": s["Q18_expected_monthly_spend_aed"].median(),
        "Visits / month": s["Q21_expected_visits_per_month"].mean(),
        "Weekly café-goers": s["Q7_cafe_visit_frequency"].isin(["Weekly", "Several times a week"]).mean(),
        "Income index (0-4)": e["income"].mean(), "Price importance (1-5)": e["price_importance"].mean(),
        "App likelihood (1-5)": s["Q17_app_download_likelihood"].mean() if "Q17_app_download_likelihood" in s else np.nan,
        "Share aged 18-34": s["Q1_age_group"].isin(["18-24", "25-34"]).mean(),
        "Share aged 45+": s["Q1_age_group"].isin(["45-54", "55+"]).mean(),
        "Tried kopi": (s["Q11_kopi_experience"] != "Never tried it").mean(),
        "Main barrier": (s[[b for b in BARRIERS if b in s and b != "Q22_barrier_none"]].mean().idxmax()
                         if any(b in s for b in BARRIERS) else ""),
    })
prof = pd.DataFrame(prof_rows)
overall = {"weekly": df["Q7_cafe_visit_frequency"].isin(["Weekly", "Several times a week"]).mean(),
           "income": E_all["income"].mean(), "price": E_all["price_importance"].mean(),
           "app": df["Q17_app_download_likelihood"].mean() if "Q17_app_download_likelihood" in df else 3,
           "kopi": (df["Q11_kopi_experience"] != "Never tried it").mean()}
seg_names = name_segments(prof, overall)
prof = prof.sort_values("Interested", ascending=False).reset_index(drop=True)
roles = [("Primary target" if i == 0 else "Secondary target" if v >= 0.5 else "Low priority" if i == len(prof) - 1 else "Bulk and walk-in")
         for i, v in enumerate(prof["Interested"])]
prof.insert(1, "Role", roles[:len(prof)])
prof.insert(1, "Segment", [f"S{i + 1}: {seg_names[c]}" for i, c in enumerate(prof["Cluster"])])
prof["Main barrier"] = prof["Main barrier"].map(BARRIERS).fillna(prof["Main barrier"])
seg_label = dict(zip(prof["Cluster"], prof["Segment"]))
df["Segment"] = df["cluster"].map(seg_label)

support, rules, launch = run_association(df, min_sup, min_conf, strong_conf, min_lift)
reg = run_regression(df)

# sustainability maths
weights = {"Definitely yes": w_dy, "Probably yes": w_py, "Not sure": w_ns}
df["conv_weight"] = df["Q14_visit_intent"].map(weights).fillna(0.0)
conv = df["conv_weight"].mean()
cw = df[df["conv_weight"] > 0]
visits = float(np.average(cw["Q21_expected_visits_per_month"], weights=cw["conv_weight"])) if len(cw) else 0.0
fixed = rent + staff + other
contrib = ticket * (1 - cogs) - ticket * deliv_share * commission
be_day = fixed / contrib / 30 if contrib > 0 else np.inf
regulars = reach * conv
orders_day = regulars * visits / 30
revenue = orders_day * 30 * ticket
profit = orders_day * 30 * contrib - fixed
payback = setup / profit if profit > 0 else np.inf
margin = profit / revenue if revenue > 0 else 0
reach_be = be_day * 30 / visits / conv if visits > 0 and conv > 0 else np.inf
reach_36 = ((setup / 36 + fixed) / contrib) / visits / conv if contrib > 0 and visits > 0 and conv > 0 else np.inf

_sel = cls["metrics"].set_index("Model").loc[cls["best"]]
best_auc = _sel["ROC AUC"]
best_acc = _sel["Accuracy"]
core = prof.iloc[0]
targets = prof[prof["Interested"] > df["interested"].mean()]
target_share = targets["Share of sample"].sum()
target_names = ", ".join(targets["Segment"]) if len(targets) else core["Segment"]
ok_launch = launch[(launch["Wish-list served"] >= target_cover) & (launch["Let down by a missing pair"] == 0)] if len(launch) else launch
rec_menu = ok_launch.iloc[0] if len(ok_launch) else (launch.iloc[-1] if len(launch) else None)
checks = [
    ("Enough demand", df["interested"].mean() >= 0.35, f"{pct(df['interested'].mean())} would probably or definitely come (need 35%+)"),
    ("A big enough target market", target_share >= 0.25,
     f"Segments more interested than average ({target_names}) cover {pct(target_share)} of people (need 25%+)"),
    ("A focused launch menu", rec_menu is not None and rec_menu["Menu size"] <= 7,
     (f"{int(rec_menu['Menu size'])} items serve {pct(rec_menu['Wish-list served'], 0)} of a wish-list (need 7 or fewer)" if rec_menu is not None else "No menu found")),
    ("Customers can be predicted", best_auc >= 0.75,
     f"Tuned {cls['best'].lower()}: {pct(best_acc)} accuracy, ROC AUC {best_auc:.2f} (need AUC 0.75+); "
     f"{pct(cls['model_b']['Accuracy'])} with the concept question"),
    ("The money works", payback <= 36, (f"Payback in {payback:.0f} months (need 36 or fewer)" if np.isfinite(payback) else "Loses money at these assumptions")),
]
passed = sum(c[1] for c in checks)
if profit <= 0:
    verdict, vcol = "Not sustainable at these assumptions", BAD
elif payback > 36 or passed < 4:
    verdict, vcol = "Borderline: sustainable only if conditions improve", WARN
else:
    verdict, vcol = "Sustainable", GOOD

# --------------------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------------------
st.title("Kopi & Crumb: is the business sustainable?")
src = "your uploaded survey" if up is not None else "a synthetic survey"
st.caption(f"Nanyang kopi bar with butter croissants and kaya toast, JLT Dubai · {src} of {len(raw):,} respondents "
           f"({len(df):,} after cleaning)")

tabs = st.tabs(["Verdict", "Questionnaire", "Data & cleaning", "EDA & correlation", "1 Classification", "2 Clustering",
                "3 Association rules", "4 Regression", "Sustainability"])

# ---- Verdict
with tabs[0]:
    st.markdown(f"<div style='padding:18px 22px;border-radius:10px;background:{vcol}18;border-left:6px solid {vcol}'>"
                f"<div style='font-size:0.9rem;color:#555'>Verdict</div>"
                f"<div style='font-size:1.9rem;font-weight:600;color:{vcol}'>{verdict}</div>"
                f"<div style='margin-top:6px'>{passed} of 5 checks passed. One café needs <b>{be_day:,.0f} orders a day</b> to break even; "
                f"reaching {reach:,} people gives about <b>{orders_day:,.0f}</b>.</div></div>", unsafe_allow_html=True)
    st.write("")
    c = st.columns(5)
    c[0].metric("Would probably or definitely visit", pct(df["interested"].mean()))
    c[1].metric("Likely to become regulars", pct(conv))
    c[2].metric("Selected model accuracy (tuned)", pct(best_acc), cls["best"], delta_color="off")
    c[3].metric("Orders a day vs break-even", f"{orders_day:,.0f}", f"{orders_day - be_day:+,.0f} vs {be_day:,.0f}")
    c[4].metric("Monthly profit", aed(profit), f"payback {payback:.0f} months" if np.isfinite(payback) else "no payback",
                delta_color="normal" if profit > 0 else "inverse")
    st.subheader("Sustainability checklist")
    for name, ok, why in checks:
        st.markdown(f"{'✅' if ok else '❌'} **{name}**: {why}")
    st.subheader("What this means")
    st.markdown(
        f"- **Who to target first:** {target_names} ({pct(target_share)} of people together). "
        f"The warmest is {core['Segment']}: {pct(core['Interested'])} interested, median spend "
        f"{aed(core['Median spend / month (AED)'])} a month.\n"
        + (f"- **What to launch with:** {rec_menu['Items']}.\n" if rec_menu is not None else "")
        + f"- **What decides it:** visits. Regulars expect about {visits:.1f} visits a month. "
          f"You need roughly {reach_be:,.0f} people to try it to break even, and {reach_36:,.0f} to pay back the setup within 3 years.\n"
          f"- **Biggest barrier:** {BARRIERS.get(df[[b for b in BARRIERS if b in df and b != 'Q22_barrier_none']].mean().idxmax(), '')}.")
    st.info("Change the business assumptions in the sidebar and the verdict updates. "
            "The built-in data is simulated; upload the real survey CSV in the sidebar to rerun everything.")

# ---- Questionnaire
with tabs[1]:
    st.subheader("The 22-question survey")
    st.write("Every question feeds at least one algorithm. Q14 is the classification target, Q15 feeds association rules, "
             "Q18 is the regression target, and Q1 to Q12 describe people for clustering.")
    st.dataframe(QUESTIONNAIRE, hide_index=True, width="stretch", height=560)
    st.download_button("Download the questionnaire (CSV)", QUESTIONNAIRE.to_csv(index=False).encode(),
                       "kopi_crumb_questionnaire.csv", "text/csv")

# ---- Data & cleaning
with tabs[2]:
    a, b, c3 = st.columns(3)
    a.metric("Raw responses", f"{len(raw):,}")
    b.metric("After cleaning", f"{len(df):,}")
    c3.metric("Rows removed", f"{len(raw) - len(df):,}")
    st.subheader("What was wrong and what we did")
    st.dataframe(clean_log, hide_index=True, width="stretch")
    col1, col2 = st.columns(2)
    miss = raw.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(12)
    if len(miss):
        f = px.bar(x=miss.values, y=miss.index, orientation="h", labels={"x": "Share blank", "y": ""},
                   title="Blank answers in the raw file")
        f.update_layout(yaxis={"autorange": "reversed"}, xaxis_tickformat=".0%", height=380)
        col1.plotly_chart(f, width="stretch")
    before = pd.to_numeric(raw["Q18_expected_monthly_spend_aed"], errors="coerce")
    box = pd.DataFrame({"AED per month": pd.concat([before, df["Q18_expected_monthly_spend_aed"]]),
                        "Data": ["Raw"] * len(before) + ["Cleaned"] * len(df)})
    f = px.box(box, x="Data", y="AED per month", points="outliers", title="Q18 monthly spend: outliers before and after")
    f.update_layout(height=380, yaxis_type="log")
    col2.plotly_chart(f, width="stretch")
    st.subheader("Preview of the raw data")
    st.dataframe(raw.head(200), width="stretch", height=300)
    st.download_button("Download cleaned data (CSV)", df.to_csv(index=False).encode(), "kopi_crumb_clean.csv", "text/csv")

# ---- EDA & correlation
with tabs[3]:
    st.subheader("Descriptive and diagnostic analytics")
    st.write("Descriptive statistics summarise what respondents said; the chi-square tests and correlations diagnose "
             "why some people are interested and others are not.")
    num = df[["Q8_spend_per_visit_aed", "Q18_expected_monthly_spend_aed", "Q19_max_price_kopi_croissant_combo_aed",
              "Q21_expected_visits_per_month"]].describe().T.round(1)
    num.index = ["Spend per visit (AED)", "Expected monthly spend (AED)", "Max combo price (AED)", "Visits per month"]
    st.dataframe(num, width="stretch")
    col1, col2 = st.columns(2)
    sh = df["Q14_visit_intent"].value_counts(normalize=True).reindex(INTENT_ORDER).reset_index()
    sh.columns = ["Answer", "Share"]
    f = px.bar(sh, x="Answer", y="Share", text=sh["Share"].map(lambda v: pct(v)), title="Q14: would you visit?",
               labels={"Answer": ""})
    f.update_layout(yaxis_tickformat=".0%", height=380)
    col1.plotly_chart(f, width="stretch")
    col1.caption("Distribution of the target variable. The top two boxes together give the interest rate used throughout.")
    inter = df[df["interested"] == 1]
    dem = inter[list(PRODUCTS)].mean().rename(index=PRODUCTS).sort_values()
    f = px.bar(x=dem.values, y=dem.index, orientation="h", text=[pct(v, 0) for v in dem.values],
               title="Q15: what interested people want", labels={"x": "Share", "y": ""})
    f.update_layout(xaxis_tickformat=".0%", height=380)
    col2.plotly_chart(f, width="stretch")
    col2.caption("Breakfast items and kopi lead demand; whole cakes and the pre-order box are niche add-ons.")
    st.subheader("Correlation matrix (Spearman)")
    C = pd.DataFrame({
        "Age": E_all["age"], "Income": E_all["income"], "Café frequency": E_all["cafe_frequency"],
        "Spend per visit": E_all["spend_per_visit"], "Kopi experience": E_all["kopi_experience"],
        "Price importance": E_all["price_importance"], "Customisation importance": E_all["customisation_importance"],
        "Coffee quality importance": E_all["coffee_quality_importance"],
        "App likelihood": df["Q17_app_download_likelihood"], "Concept appeal": df["Q13_concept_appeal"],
        "Visit intent": df["Q14_visit_intent"].map(INTENT), "Items wanted": df[list(PRODUCTS)].sum(axis=1),
        "Visits / month": df["Q21_expected_visits_per_month"], "Monthly spend": df["Q18_expected_monthly_spend_aed"]})
    corr = C.corr(method="spearman")
    f = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#FFFFFF"], [0.5, "#BDBDBD"], [1, "#000000"]], zmin=-1, zmax=1, aspect="auto")
    f.update_layout(height=620)
    st.plotly_chart(f, width="stretch")
    st.caption("Spearman correlation is used because most variables are ordered categories or skewed amounts. "
               "Visits per month and visit intent move most closely with monthly spend; price importance moves against it.")
    col1, col2 = st.columns(2)
    pre = C[["Age", "Income", "Café frequency", "Spend per visit", "Kopi experience", "Price importance",
             "Customisation importance", "Coffee quality importance", "App likelihood"]]
    ci = pre.corrwith(df["interested"], method="spearman").sort_values()
    f = px.bar(x=ci.values, y=ci.index, orientation="h", text=[f"{v:.2f}" for v in ci.values],
               title="Correlation of each trait with being interested", labels={"x": "Spearman r", "y": ""},
               color=ci.values > 0, color_discrete_map={True: COFFEE[2], False: BAD})
    f.update_layout(showlegend=False, height=400)
    col1.plotly_chart(f, width="stretch")
    rows = []
    for col in ["Q7_cafe_visit_frequency", "Q11_kopi_experience", "Q1_age_group", "Q4_household_income_aed_month",
                "Q3_occupation", "Q5_area"]:
        if col in df:
            t = pd.crosstab(df[col], df["interested"])
            chi2, p, dof, _ = chi2_contingency(t)
            rows.append({"Variable": col.split("_", 1)[1].replace("_", " ").capitalize(), "Chi-square": chi2, "df": dof,
                         "p-value": p, "Cramér's V": np.sqrt(chi2 / (t.values.sum() * (min(t.shape) - 1))),
                         "Significant (p < 0.05)": "Yes" if p < 0.05 else "No"})
    col2.write("**Chi-square tests: is interest linked to each characteristic?**")
    col2.dataframe(pd.DataFrame(rows).style.format({"Chi-square": "{:.1f}", "p-value": "{:.2g}", "Cramér's V": "{:.3f}"}),
                   hide_index=True, width="stretch")
    col2.caption("Café frequency has the strongest link with interest. Area is not significant, so location should be "
                 "chosen on rent and footfall rather than on where interest is highest.")

# ---- Classification
with tabs[4]:
    st.subheader("Will they come? Predicting Q14 interest")
    st.write("Target: 'Probably yes' or 'Definitely yes' on Q14. The models only use answers we would know before showing "
             "the concept (Q1 to Q12), so the same model can score new prospects. 25% of people are held back for testing. "
             "Every model below uses the best settings found by hyperparameter tuning.")
    m = cls["metrics"].copy()
    st.dataframe(m.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.3f}",
                                 "ROC AUC": "{:.3f}"}).highlight_max(subset=["Accuracy"], color="#E8D9C5"),
                 hide_index=True, width="stretch")

    st.subheader("Hyperparameter tuning")
    st.write(f"**{TUNING_TOTAL['configurations']:,} configurations and {TUNING_TOTAL['fits']:,} model fits.** Exhaustive grids "
             "(randomised search for random forest and gradient boosting) covered scaling method, feature selection, "
             "regularisation, class weighting and each model's own settings, all scored by 5-fold cross-validated accuracy on "
             "the training data only, with scaling and feature selection inside the pipeline. Finalists were re-ranked with "
             "repeated cross-validation (5 folds x 3 repeats); the test set was used once. The decision threshold was tuned "
             f"too and stayed at {BEST_THRESHOLD:.2f}.")
    col1, col2 = st.columns([3, 2])
    long = TUNING.melt(id_vars="Model", value_vars=["CV accuracy (default)", "CV accuracy (tuned)"], var_name="Stage",
                       value_name="Accuracy").dropna()
    long["Stage"] = long["Stage"].map({"CV accuracy (default)": "Default settings", "CV accuracy (tuned)": "Tuned"})
    f = px.bar(long, x="Accuracy", y="Model", color="Stage", barmode="group", orientation="h",
               text=long["Accuracy"].map(lambda v: pct(v)), color_discrete_map={"Default settings": "#BBBBBB", "Tuned": COFFEE[0]},
               title="Cross-validated accuracy before and after tuning (training data)")
    f.add_vline(x=ACCURACY_CEILING, line_dash="dash", line_color="#444444",
                annotation_text=f"Ceiling {pct(ACCURACY_CEILING)}", annotation_position="top")
    f.update_layout(xaxis_tickformat=".0%", xaxis_range=[0.55, 0.85], height=480, yaxis={"autorange": "reversed"},
                    legend_title_text="", legend={"orientation": "h", "y": -0.18})
    col1.plotly_chart(f, width="stretch")
    sel = m.set_index("Model").loc[SELECTED_MODEL]
    col2.write(f"**Selected model: tuned {SELECTED_MODEL.lower()}**")
    col2.write("Highest repeated cross-validation accuracy, no gap between training and validation accuracy, and the "
               "easiest model to explain. Every tuned model sits within about 1.5 points of it on the test set.")
    a1, a2 = col2.columns(2)
    a1.metric("Test accuracy", pct(sel["Accuracy"]))
    a2.metric("ROC AUC", f"{sel['ROC AUC']:.3f}")
    mb = cls["model_b"]
    col2.write("**Model B: add one concept question (Q13)**")
    b1, b2 = col2.columns(2)
    b1.metric("Test accuracy", pct(mb["Accuracy"]), f"{(mb['Accuracy'] - sel['Accuracy']) * 100:+.1f} pts")
    b2.metric("Precision", pct(mb["Precision"]))
    st.dataframe(TUNING.style.format({"CV accuracy (default)": "{:.1%}", "CV accuracy (tuned)": "{:.1%}",
                                      "Repeated CV accuracy": "{:.1%}", "Test accuracy (default)": "{:.1%}",
                                      "Configurations": "{:,.0f}", "Fits": "{:,.0f}"}, na_rep="-"),
                 hide_index=True, width="stretch")
    st.caption(f"The ceiling ({pct(ACCURACY_CEILING)}) is what a model would score if it knew every hidden trait used to "
               "simulate each respondent. Profile answers reveal only part of those traits, so profile-only models stop near "
               "74%. Model B adds a reaction to the concept and reaches the ceiling.")
    col1, col2 = st.columns(2)
    f = go.Figure()
    for i, (name, (fpr, tpr)) in enumerate(cls["rocs"].items()):
        f.add_trace(go.Scatter(x=fpr, y=tpr, name=name.replace(" (LR + HGB + GB)", ""), line={"color": COFFEE[i % len(COFFEE)], "width": 3 if name == cls["best"] else 1.2}))
    f.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random guess", line={"dash": "dash", "color": "#999"}))
    f.update_layout(title="ROC curves", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
    col1.plotly_chart(f, width="stretch")
    cm = cls["cm"]
    f = px.imshow(cm, text_auto=True, color_continuous_scale=["#FFF8F0", "#6F4E37"],
                  x=["Predicted not interested", "Predicted interested"], y=["Not interested", "Interested"],
                  title=f"Confusion matrix: {cls['best']} (selected model)")
    f.update_layout(height=420, coloraxis_showscale=False)
    col2.plotly_chart(f, width="stretch")
    col1, col2 = st.columns(2)
    fi = cls["fi"].rename(index=lambda x: NICE.get(x, x))
    f = px.bar(x=fi.values, y=fi.index, orientation="h", title="What predicts interest (random forest importance)",
               labels={"x": "Importance", "y": ""})
    f.update_layout(yaxis={"autorange": "reversed"}, height=440)
    col1.plotly_chart(f, width="stretch")
    co = cls["coef"].rename(index=lambda x: NICE.get(x, x))
    f = px.bar(x=co.values, y=co.index, orientation="h", title="Direction of effect (logistic regression coefficients)",
               labels={"x": "Coefficient (standardised)", "y": ""}, color=co.values > 0,
               color_discrete_map={True: COFFEE[2], False: BAD})
    f.update_layout(yaxis={"autorange": "reversed"}, height=440, showlegend=False)
    col2.plotly_chart(f, width="stretch")
    st.subheader("Interest by group")
    g = st.selectbox("Break interest down by", ["Q7_cafe_visit_frequency", "Q11_kopi_experience", "Q1_age_group",
                                                 "Q4_household_income_aed_month", "Q5_area", "Q3_occupation"],
                     format_func=lambda c: c.split("_", 1)[1].replace("_", " ").capitalize())
    by = df.groupby(g).agg(Interested=("interested", "mean"), People=("interested", "size")).reset_index()
    if g == "Q7_cafe_visit_frequency":
        by["o"] = by[g].map(FREQ); by = by.sort_values("o")
    elif g == "Q1_age_group":
        by["o"] = by[g].map(AGE); by = by.sort_values("o")
    else:
        by = by.sort_values("Interested", ascending=False)
    f = px.bar(by, x=g, y="Interested", text=by["Interested"].map(lambda v: pct(v, 0)), hover_data=["People"],
               labels={g: ""})
    f.update_layout(yaxis_tickformat=".0%", height=380)
    st.plotly_chart(f, width="stretch")

# ---- Clustering
with tabs[5]:
    st.subheader("Who should we target? K-means segments")
    st.write(f"K-means on age, income, household size, café habits, spend, kopi experience, the seven Q12 ratings and app "
             f"likelihood, with **robust scaling**. {CLUSTER_SEARCH} solutions (3 scalers x 6 algorithms x k = 2 to 8) were "
             f"compared on three validity measures; robust-scaled k-means led them. Showing **k = {k_choice}**; change it in the sidebar.")
    cols = st.columns(3)
    for col, (met, note) in zip(cols, [("Silhouette", "higher is better"), ("Davies-Bouldin", "lower is better"),
                                       ("Calinski-Harabasz", "higher is better")]):
        f = px.line(k_table, x="k", y=met, markers=True, title=f"{met} ({note})")
        f.add_vline(x=k_choice, line_dash="dash", line_color=COFFEE[1])
        f.update_layout(height=300)
        col.plotly_chart(f, width="stretch")
    st.caption("k = 3 scores best on all three measures. k = 4 is kept because it beats the original standard-scaled "
               "4-segment solution on all three and splits frequent café-goers by price sensitivity, which the pricing plan "
               "needs. Scores this low are normal for survey ratings.")
    show = prof.drop(columns=["Cluster"])
    st.dataframe(show.style.format({"Share of sample": "{:.1%}", "Interested": "{:.1%}", "High tier": "{:.1%}",
                                    "Median spend / month (AED)": "{:,.0f}", "Visits / month": "{:.1f}",
                                    "Weekly café-goers": "{:.0%}", "Income index (0-4)": "{:.2f}",
                                    "Price importance (1-5)": "{:.2f}", "App likelihood (1-5)": "{:.2f}",
                                    "Share aged 18-34": "{:.0%}", "Share aged 45+": "{:.0%}", "Tried kopi": "{:.0%}"}),
                 hide_index=True, width="stretch")
    col1, col2 = st.columns(2)
    pts = pd.DataFrame({"PC1": pca_xy[:, 0], "PC2": pca_xy[:, 1], "Segment": df["Segment"]}).sample(
        min(2000, len(df)), random_state=0)
    f = px.scatter(pts, x="PC1", y="PC2", color="Segment", opacity=0.6, title="Segments on a 2-D map (PCA)")
    f.update_layout(height=430)
    col1.plotly_chart(f, width="stretch")
    tier = (df.groupby(["Segment", "Conversion tier"]).size() / df.groupby("Segment").size()).rename("Share").reset_index()
    f = px.bar(tier, x="Share", y="Segment", color="Conversion tier", orientation="h",
               category_orders={"Conversion tier": ["High", "Medium", "Low"], "Segment": list(prof["Segment"])},
               color_discrete_map={"High": COFFEE[2], "Medium": COFFEE[1], "Low": "#BBBBBB"},
               title="High, medium and low conversion probability in each segment")
    f.update_layout(xaxis_tickformat=".0%", height=430)
    col2.plotly_chart(f, width="stretch")
    t = df.groupby("Conversion tier").agg(People=("interested", "size"), Actually_interested=("interested", "mean"),
                                          Median_spend=("Q18_expected_monthly_spend_aed", "median")).reindex(["High", "Medium", "Low"])
    st.write("**Do the conversion tiers hold up?** Share actually interested in each predicted tier:")
    st.dataframe(t.rename(columns={"Actually_interested": "Actually interested", "Median_spend": "Median spend / month (AED)"})
                 .style.format({"Actually interested": "{:.1%}", "Median spend / month (AED)": "{:,.0f}"}),
                 width="stretch")

# ---- Association
with tabs[6]:
    st.subheader("What should we launch first? Association rules on Q15")
    st.write("Apriori on the wish-lists of interested respondents. For every menu size, every combination is tested to find "
             "the one that serves the most of each person's wish-list without breaking a 'must keep together' rule "
             "(if they want A and B, they also expect C).")
    col1, col2 = st.columns(2)
    f = px.bar(x=support.values, y=support.index, orientation="h", title="How many interested people want each item (support)",
               labels={"x": "Support", "y": ""}, text=[pct(v, 0) for v in support.values])
    f.update_layout(yaxis={"autorange": "reversed"}, xaxis_tickformat=".0%", height=460)
    col1.plotly_chart(f, width="stretch")
    if len(launch):
        rec_size = int(rec_menu["Menu size"]) if rec_menu is not None else None
        colors = [COFFEE[0] if s == rec_size else ("#C62828" if l > 0 else "#BBBBBB")
                  for s, l in zip(launch["Menu size"], launch["Let down by a missing pair"])]
        f = go.Figure(go.Bar(x=launch["Menu size"], y=launch["Wish-list served"], marker_color=colors,
                             text=[pct(v, 0) + (f"<br>{pct(l, 0)} let down" if l > 0 else "")
                                   for v, l in zip(launch["Wish-list served"], launch["Let down by a missing pair"])],
                             textposition="outside", hovertext=launch["Items"]))
        f.add_hline(y=target_cover, line_dash="dash", line_color=COFFEE[1], annotation_text="target")
        f.update_layout(title="Best menu at each size (brown = recommended, red = breaks a rule)", yaxis_tickformat=".0%",
                        xaxis_title="Number of menu items", yaxis_title="Wish-list served", height=460, yaxis_range=[0, 1.15])
        col2.plotly_chart(f, width="stretch")
        if rec_menu is not None:
            st.success(f"**Recommended launch menu ({int(rec_menu['Menu size'])} items):** {rec_menu['Items']}. "
                       f"Serves {pct(rec_menu['Wish-list served'], 0)} of an average wish-list. "
                       "App, delivery and the weekly pre-order box are judged separately below.")
    st.subheader("The rules")
    if rules.empty:
        st.warning("No rules at these settings. Lower the minimum support or confidence in the sidebar.")
    else:
        only_menu = st.checkbox("Hide rules that only involve the app, delivery and pre-order box", value=False)
        r = rules
        if only_menu:
            r = r[~(r["antecedents"].apply(lambda a: all(i in SERVICES for i in a)) & r["They also want"].isin(SERVICES))]
        top = r[["If they want", "They also want", "support", "confidence", "lift"]].head(40)
        st.dataframe(top.rename(columns={"support": "Support", "confidence": "Confidence", "lift": "Lift"})
                     .style.format({"Support": "{:.1%}", "Confidence": "{:.1%}", "Lift": "{:.2f}"}),
                     hide_index=True, width="stretch", height=420)
        st.caption(f"{len(rules):,} rules at the validated thresholds (support 8%, confidence 70%, lift 1.3, chosen from "
                   f"{RULE_THRESHOLDS_TESTED} threshold combinations because every rule kept its lift above 1 on unseen baskets). "
                   "Support = share of interested people who want all the items. "
                   "Confidence = of those who want the left side, the share who also want the right side. "
                   "Lift above 1 = they go together more than by chance.")
        svc = rules[rules["They also want"].isin(SERVICES) & (rules["antecedents"].apply(len) == 1)].head(6)
        if len(svc):
            st.write("**App, delivery and pre-order box:**")
            st.dataframe(svc[["If they want", "They also want", "confidence", "lift"]]
                         .rename(columns={"confidence": "Confidence", "lift": "Lift"})
                         .style.format({"Confidence": "{:.1%}", "Lift": "{:.2f}"}), hide_index=True, width="stretch")

# ---- Regression
with tabs[7]:
    st.subheader("How much will they spend? Predicting Q18 monthly spend")
    st.write("Inputs: profile answers plus concept appeal (Q13), visit intent (Q14), app likelihood (Q17), "
             "number of items wanted (Q15) and bulk-order intent (Q20).")
    st.dataframe(reg["metrics"].style.format({"R²": "{:.3f}", "MAE (AED)": "{:,.1f}", "RMSE (AED)": "{:,.1f}"})
                 .highlight_max(subset=["R²"], color="#E8D9C5"), hide_index=True, width="stretch")
    with st.expander("Hyperparameter tuning for the spend models"):
        st.dataframe(REG_TUNING.style.format({"Test R² (default)": "{:.3f}", "CV R² (tuned)": "{:.3f}", "Training R² (tuned)": "{:.3f}"}),
                     hide_index=True, width="stretch")
        st.caption(f"{REG_TOTAL['configurations']} configurations and {REG_TOTAL['fits']} fits, scored by 5-fold cross-validated R² "
                   "on the training data. Gradient boosting is used for predictions: its CV R² is within 0.01 of the random "
                   "forest, but its training-validation gap is much smaller, so it is the more stable choice.")
    col1, col2 = st.columns(2)
    best_reg = reg["metrics"].sort_values("R²", ascending=False).iloc[0]["Model"]
    ap = pd.DataFrame({"Actual": reg["yte"], "Predicted": reg["preds"][best_reg]})
    f = px.scatter(ap, x="Actual", y="Predicted", opacity=0.4, title=f"Actual vs predicted ({best_reg})")
    mx = float(max(ap.max()))
    f.add_trace(go.Scatter(x=[0, mx], y=[0, mx], mode="lines", line={"dash": "dash", "color": "#999"}, name="Perfect"))
    f.update_layout(height=430, xaxis_title="Actual AED / month", yaxis_title="Predicted AED / month")
    col1.plotly_chart(f, width="stretch")
    co = reg["coef"].sort_values().rename(index=lambda x: NICE.get(x, x))
    f = px.bar(x=co.values, y=co.index, orientation="h", title="Linear regression: AED per month for each step up",
               labels={"x": "AED per month", "y": ""}, color=co.values > 0,
               color_discrete_map={True: COFFEE[2], False: BAD}, text=[f"{v:+.1f}" for v in co.values])
    f.update_layout(height=430, showlegend=False)
    col2.plotly_chart(f, width="stretch")
    sp = df.groupby("Q14_visit_intent").agg(Median=("Q18_expected_monthly_spend_aed", "median"),
                                             Visits=("Q21_expected_visits_per_month", "mean")).reindex(INTENT_ORDER).reset_index()
    f = px.bar(sp, x="Q14_visit_intent", y="Median", text=sp["Median"].map(aed), title="Median expected monthly spend by Q14 answer",
               labels={"Q14_visit_intent": "", "Median": "AED per month"})
    f.update_layout(height=360)
    st.plotly_chart(f, width="stretch")

# ---- Sustainability
with tabs[8]:
    st.subheader("Does the money work?")
    st.markdown(f"<div style='font-size:1.4rem;font-weight:600;color:{vcol}'>{verdict}</div>", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].metric("Break-even orders a day", f"{be_day:,.0f}")
    c[1].metric("Expected orders a day", f"{orders_day:,.0f}", f"{regulars:,.0f} regulars × {visits:.1f} visits/month")
    c[2].metric("Profit margin", pct(margin) if profit > 0 else "Loss")
    c[3].metric("Payback on setup", f"{payback:.0f} months" if np.isfinite(payback) else "Never")
    st.markdown(f"""
**How the numbers connect**
1. {reach:,} people try Kopi & Crumb in the first year (your input).
2. The survey says {pct(conv)} of them become regulars after discounting stated intent → **{regulars:,.0f} regulars**.
3. Regulars expect {visits:.1f} visits a month → **{orders_day:,.0f} orders a day**.
4. Each order brings in AED {ticket:.0f} and leaves **AED {contrib:.2f}** after cost of goods and delivery commission.
5. Fixed costs are **{aed(fixed)} a month**, so break-even is **{be_day:,.0f} orders a day**.
6. To break even you need about **{reach_be:,.0f}** people to try it; to pay back the {aed(setup)} setup within 3 years, about **{reach_36:,.0f}**.
""")
    sc_rows = []
    for o in [60, 80, 100, 120, 150, 180, 220]:
        p = o * 30 * contrib - fixed
        sc_rows.append({"Orders a day": o, "Revenue / month": o * 30 * ticket, "Profit / month": p})
    sc_df = pd.DataFrame(sc_rows)
    f = go.Figure(go.Bar(x=sc_df["Orders a day"], y=sc_df["Profit / month"],
                         marker_color=[COFFEE[2] if v > 0 else BAD for v in sc_df["Profit / month"]],
                         text=[aed(v) for v in sc_df["Profit / month"]], textposition="outside"))
    f.add_vline(x=orders_day, line_dash="dash", line_color=COFFEE[0], annotation_text="expected")
    f.update_layout(title="Monthly profit at different order levels", xaxis_title="Orders a day",
                    yaxis_title="AED per month", height=420)
    st.plotly_chart(f, width="stretch")
    st.subheader("What could stop people (Q22)")
    bars = df[[b for b in BARRIERS if b in df]].mean().rename(index=BARRIERS).sort_values()
    f = px.bar(x=bars.values, y=bars.index, orientation="h", labels={"x": "Share of respondents", "y": ""},
               text=[pct(v, 0) for v in bars.values])
    f.update_layout(xaxis_tickformat=".0%", height=360)
    st.plotly_chart(f, width="stretch")
    if "Q19_max_price_kopi_croissant_combo_aed" in df:
        q19 = df.loc[df["interested"] == 1, "Q19_max_price_kopi_croissant_combo_aed"]
        st.write(f"**Pricing check (Q19):** interested people would pay a median of **AED {q19.median():.0f}** for a kopi and "
                 f"croissant combo; {pct((q19 >= 30).mean(), 0)} would pay AED 30 or more.")
    st.caption("Rent, staff and setup defaults follow published 2026 Dubai café cost ranges; other inputs are assumptions. "
               "All survey numbers come from simulated respondents unless you upload real data.")
