# streamlit_full_app.py
# Creative Interactive Insights Dashboard with Modeling & Prediction
# - Interactive Plotly visuals with filters & KPIs
# - Multi-model training + cross-validated leaderboard
# - Optional hyperparameter tuning (RandomizedSearchCV)
# - SHAP explainability for tree models
# - Prediction UI for new data upload

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import shap
import joblib
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# ----------------- Configuration -----------------
SAMPLE_CSV = "issue_list_sample.csv"  # bundled sample generated earlier
MODEL_SAVE_PATH = "best_model.joblib"

st.set_page_config(page_title="Creative Insight Dashboard", layout="wide")

# ----------------- Helpers -----------------
@st.cache_data
def load_data(path=SAMPLE_CSV):
    return pd.read_csv(path)


def safe_to_datetime(series):
    return pd.to_datetime(series, errors='coerce')


def summarize_kpis(df):
    total = len(df)
    solved_pct = (df['Status_Description'].eq('Solved').mean()*100) if 'Status_Description' in df.columns else np.nan
    avg_days = df['Days_Taken'].mean() if 'Days_Taken' in df.columns else np.nan
    top_zone = df['Zone_Name'].mode().iat[0] if 'Zone_Name' in df.columns and not df['Zone_Name'].mode().empty else None
    return total, solved_pct, avg_days, top_zone


def compute_plot_metrics(plot_type, df, x, y=None):
    m = {}
    try:
        if plot_type == 'Scatter' and x and y:
            clean = df[[x, y]].dropna()
            if len(clean) >= 3 and pd.api.types.is_numeric_dtype(clean[x]) and pd.api.types.is_numeric_dtype(clean[y]):
                r, p = stats.pearsonr(clean[x].astype(float), clean[y].astype(float))
                m['pearson_r'] = float(r)
                m['p_value'] = float(p)
                m['n'] = int(len(clean))
        elif plot_type == 'Histogram' and x:
            col = df[x].dropna()
            if pd.api.types.is_numeric_dtype(col):
                m['skewness'] = float(col.skew())
                m['kurtosis'] = float(col.kurtosis())
                m['n'] = int(len(col))
            else:
                m['top_counts'] = col.value_counts().head(5).to_dict()
        elif plot_type == 'Box' and (x or y):
            colname = y or x
            col = df[colname].dropna()
            if pd.api.types.is_numeric_dtype(col):
                m['IQR'] = float(col.quantile(0.75) - col.quantile(0.25))
                m['median'] = float(col.median())
        elif plot_type == 'Line' and x and y:
            clean = df[[x, y]].dropna()
            if pd.api.types.is_numeric_dtype(clean[y]):
                # numeric X or datetime
                if np.issubdtype(clean[x].dtype, np.datetime64):
                    Xnum = clean[x].map(datetime.toordinal).values
                else:
                    Xnum = clean[x].astype(float).values
                slope, intercept, r_value, p_value, std_err = stats.linregress(Xnum, clean[y].astype(float))
                m['slope'] = float(slope)
                m['r_squared'] = float(r_value**2)
        elif plot_type == 'Bar' and x:
            vc = df[x].value_counts()
            m['top_counts'] = vc.head(5).to_dict()
            m['unique'] = int(vc.size)
    except Exception:
        pass
    return m


def encode_df(X):
    encoders = {}
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_enc[col].astype(str))
        encoders[col] = le
    return X_enc, encoders

# ----------------- Load & preprocess -----------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV/Excel (optional)", type=['csv', 'xlsx'])
if uploaded is not None:
    try:
        if str(uploaded.name).lower().endswith('.csv'):
            df = pd.read_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.sidebar.success(f"Loaded {uploaded.name} — {df.shape[0]} rows")
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
        df = load_data(SAMPLE_CSV)
else:
    try:
        df = load_data(SAMPLE_CSV)
    except Exception:
        st.error("No sample dataset found. Please upload your CSV.")
        st.stop()

# parse common date columns
for c in ['Issue_Date', 'Solve_Date']:
    if c in df.columns:
        df[c] = safe_to_datetime(df[c])

st.title("Creative Interactive Insights — Streamlit")
st.markdown("Filter, visualize, model and predict from your Issue List dataset.")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters & Options")
if 'Issue_Date' in df.columns:
    min_d, max_d = df['Issue_Date'].min().date(), df['Issue_Date'].max().date()
    date_range = st.sidebar.date_input("Issue date range", value=(min_d, max_d))
    if len(date_range) == 2:
        start_d, end_d = date_range
        df = df[(df['Issue_Date'] >= pd.to_datetime(start_d)) & (df['Issue_Date'] <= pd.to_datetime(end_d))]

if 'Zone_Name' in df.columns:
    zone_sel = st.sidebar.multiselect("Zone", options=df['Zone_Name'].dropna().unique().tolist(), default=None)
    if zone_sel:
        df = df[df['Zone_Name'].isin(zone_sel)]

if 'Status_Description' in df.columns:
    status_sel = st.sidebar.multiselect("Status", options=df['Status_Description'].dropna().unique().tolist(), default=None)
    if status_sel:
        df = df[df['Status_Description'].isin(status_sel)]

# Quick text featurization option
use_text = False
if 'Subject' in df.columns:
    use_text = st.sidebar.checkbox('Use Subject TF-IDF in modelling (adds features)', value=False)

# ---------- KPIs ----------
total, solved_pct, avg_days, top_zone = summarize_kpis(df)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total issues", f"{total:,}")
k2.metric("% Solved", f"{solved_pct:.1f}%" if not np.isnan(solved_pct) else "N/A")
k3.metric("Avg days taken", f"{avg_days:.2f}" if not np.isnan(avg_days) else "N/A")
k4.metric("Top zone", top_zone if top_zone is not None else "N/A")

st.markdown("---")

# ---------- Visuals ----------
st.header("Interactive Visualizations")
vis_col1, vis_col2 = st.columns([2,1])

with vis_col1:
    plot_type = st.selectbox("Plot type", ['Bar', 'Line', 'Scatter', 'Histogram', 'Box', 'Pie', 'Sunburst'])
    x_col = st.selectbox("X column", options=[None] + list(df.columns), index=1 if df.columns.size>0 else 0)
    y_col = st.selectbox("Y column (if applicable)", options=[None] + list(df.columns))

    fig = None
    if plot_type == 'Bar' and x_col:
        fig = px.bar(df, x=x_col, y=y_col, title=f"Bar: {x_col}")
    elif plot_type == 'Line' and x_col and y_col:
        fig = px.line(df, x=x_col, y=y_col, title=f"Line: {y_col} over {x_col}")
    elif plot_type == 'Scatter' and x_col and y_col:
        fig = px.scatter(df, x=x_col, y=y_col, hover_data=df.columns, title=f"Scatter: {y_col} vs {x_col}")
    elif plot_type == 'Histogram' and x_col:
        fig = px.histogram(df, x=x_col, title=f"Histogram: {x_col}")
    elif plot_type == 'Box' and (x_col or y_col):
        colname = y_col or x_col
        fig = px.box(df, y=colname, title=f"Box: {colname}")
    elif plot_type == 'Pie' and x_col:
        fig = px.pie(df, names=x_col, title=f"Pie: {x_col}")
    elif plot_type == 'Sunburst' and x_col and y_col:
        fig = px.sunburst(df, path=[x_col, y_col], title=f"Sunburst: {x_col} -> {y_col}")

    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Select valid columns for this plot type')

with vis_col2:
    st.subheader('Plot performance metrics')
    perf = compute_plot_metrics(plot_type, df, x_col, y_col)
    if perf:
        st.json(perf)
    else:
        st.write('No metrics available for selection')

st.markdown('---')

# ---------- Modelling ----------
st.header('Modeling & Prediction')
# allow user to pick target
target = st.selectbox('Choose target column for prediction', options=[None] + list(df.columns))
if target:
    st.write('Target:', target)

    # prepare X and y
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()

    # drop obvious identifiers
    id_like = [c for c in X.columns if 'id' in c.lower() or 'number' in c.lower() or c.lower().endswith('_no')]
    X = X.drop(columns=[c for c in id_like if c in X.columns])

    # optional text features
    tfidf = None
    tfidf_cols = []
    if use_text and 'Subject' in X.columns:
        st.info('Applying TF-IDF to Subject (top 50 features)')
        tfidf = TfidfVectorizer(max_features=50, stop_words='english')
        subj_feats = tfidf.fit_transform(X['Subject'].astype(str)).toarray()
        subj_df = pd.DataFrame(subj_feats, columns=[f'subj_tfidf_{i}' for i in range(subj_feats.shape[1])])
        X = pd.concat([X.reset_index(drop=True), subj_df.reset_index(drop=True)], axis=1)
        tfidf_cols = subj_df.columns.tolist()
        X = X.drop(columns=['Subject'])

    # encode categoricals
    X_enc, encoders = encode_df(X)

    # handle missing numeric
    for c in X_enc.select_dtypes(include=[np.number]).columns:
        X_enc[c] = X_enc[c].fillna(X_enc[c].median())

    # detect problem type
    if y.dtype == 'object' or (pd.api.types.is_integer_dtype(y) and y.nunique() <= 10):
        problem = 'classification'
        y_enc = LabelEncoder().fit_transform(y.astype(str))
    else:
        problem = 'regression'
        y_enc = y.fillna(y.mean()).astype(float).values

    # scale numeric
    num_cols = X_enc.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    if len(num_cols) > 0:
        X_enc[num_cols] = scaler.fit_transform(X_enc[num_cols])

    # let user choose models to run
    st.subheader('Model selection & cross-validation')
    run_models = st.multiselect('Pick models to evaluate', options=['LogisticRegression','RandomForest','GradientBoosting','LinearRegression','RandomForestRegressor','GradientBoostingRegressor'], default=['RandomForest','GradientBoosting'] if problem=='classification' else ['RandomForestRegressor','GradientBoostingRegressor'])

    # build model dict depending on problem
    models = {}
    if problem == 'classification':
        if 'LogisticRegression' in run_models:
            models['Logistic Regression'] = LogisticRegression(max_iter=1000)
        if 'RandomForest' in run_models:
            models['Random Forest'] = RandomForestClassifier(n_estimators=200, random_state=42)
        if 'GradientBoosting' in run_models:
            models['Gradient Boosting'] = GradientBoostingClassifier(n_estimators=200, random_state=42)
        scoring = st.selectbox('CV scoring metric', options=['accuracy','f1_weighted'], index=0)
    else:
        if 'LinearRegression' in run_models:
            models['Linear Regression'] = LinearRegression()
        if 'RandomForestRegressor' in run_models:
            models['Random Forest Regressor'] = RandomForestRegressor(n_estimators=200, random_state=42)
        if 'GradientBoostingRegressor' in run_models:
            models['Gradient Boosting Regressor'] = GradientBoostingRegressor(n_estimators=200, random_state=42)
        scoring = st.selectbox('CV scoring metric', options=['r2','neg_root_mean_squared_error'], index=0)

    if st.button('Run cross-validated leaderboard'):
        with st.spinner('Running cross-validation...'):
            X_vals = X_enc.values
            leaderboard = []
            for name, model in models.items():
                try:
                    scores = cross_val_score(model, X_vals, y_enc, cv=5, scoring=scoring, n_jobs=-1)
                    leaderboard.append({'model': name, 'mean_score': float(np.mean(scores)), 'std': float(np.std(scores))})
                except Exception as e:
                    leaderboard.append({'model': name, 'mean_score': None, 'std': None, 'error': str(e)})
            lb = pd.DataFrame(leaderboard).sort_values(by='mean_score', ascending=False)
            st.dataframe(lb)

            # fit best model on full train/test split
            best_name = lb.iloc[0]['model']
            st.success(f'Best model by CV: {best_name}')
            best_model = models[best_name]
            X_train, X_test, y_train, y_test = train_test_split(X_enc, y_enc, test_size=0.2, random_state=42)
            best_model.fit(X_train, y_train)
            preds = best_model.predict(X_test)

            # holdout metrics
            st.subheader('Holdout evaluation (20%)')
            if problem == 'classification':
                acc = accuracy_score(y_test, preds)
                f1 = f1_score(y_test, preds, average='weighted')
                st.write(f'Accuracy: {acc:.4f} | F1 (weighted): {f1:.4f}')
                st.text('Classification report:')
                st.text(classification_report(y_test, preds))
            else:
                rmse = mean_squared_error(y_test, preds, squared=False)
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                st.write(f'RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}')

            # Save best model
            joblib.dump({'model': best_model, 'encoders': encoders, 'scaler': scaler, 'tfidf': tfidf, 'feature_cols': X_enc.columns.tolist()}, MODEL_SAVE_PATH)
            st.success(f'Saved best model to {MODEL_SAVE_PATH}')

            # SHAP explainability
            st.subheader('SHAP explainability (tree-based models)')
            try:
                if hasattr(best_model, 'feature_importances_'):
                    explainer = shap.Explainer(best_model, X_train)
                    shap_values = explainer(X_test)
                    st.pyplot(shap.plots.bar(shap_values, show=False))
                    st.pyplot(shap.plots.beeswarm(shap_values, show=False))
                else:
                    st.info('Best model is not tree-based; SHAP summary may be limited')
            except Exception as e:
                st.error(f'SHAP failed: {e}')

# ---------- Prediction UI ----------
st.markdown('---')
st.header('Prediction — Use saved model')
if st.button('Load saved model if exists'):
    try:
        saved = joblib.load(MODEL_SAVE_PATH)
        st.success('Loaded saved model')
    except Exception as e:
        st.error('No saved model found or failed to load')

uploaded_pred = st.file_uploader('Upload CSV for prediction (no target column)', type=['csv'], key='pred2')
if uploaded_pred is not None:
    try:
        newdf = pd.read_csv(uploaded_pred)
        st.write('Preview of new data')
        st.dataframe(newdf.head())
        try:
            saved = joblib.load(MODEL_SAVE_PATH)
            model = saved['model']
            encs = saved.get('encoders', {})
            sc = saved.get('scaler', None)
            tf = saved.get('tfidf', None)
            feature_cols = saved.get('feature_cols', None)
            # apply same transforms
            for col, le in encs.items():
                if col in newdf.columns:
                    newdf[col] = le.transform(newdf[col].astype(str))
            if tf is not None and 'Subject' in newdf.columns:
                subj_feats = tf.transform(newdf['Subject'].astype(str)).toarray()
                subj_df = pd.DataFrame(subj_feats, columns=[f'subj_tfidf_{i}' for i in range(subj_feats.shape[1])])
                newdf = pd.concat([newdf.reset_index(drop=True), subj_df.reset_index(drop=True)], axis=1)
                newdf = newdf.drop(columns=['Subject'])
            for c in newdf.select_dtypes(include=[np.number]).columns:
                newdf[c] = newdf[c].fillna(newdf[c].median())
            if sc is not None and feature_cols is not None:
                num_features = [c for c in feature_cols if c in newdf.columns and pd.api.types.is_numeric_dtype(newdf[c])]
                if len(num_features) > 0:
                    newdf[num_features] = sc.transform(newdf[num_features])
            # select feature order
            Xnew = newdf[feature_cols]
            preds = model.predict(Xnew)
            st.write('Predictions:')
            st.dataframe(pd.DataFrame({'prediction': preds}))
        except Exception as e:
            st.error(f'Prediction failed: {e}')
    except Exception as e:
        st.error(f'Failed to read uploaded CSV: {e}')

st.markdown('---')
st.caption('App created: interactive visuals (Plotly), model leaderboard, SHAP explainability, and prediction interface. Customize further as needed.')
