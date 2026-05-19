"""
Forest Fire Prediction Web Application (With Authentication)
==============================================================
Streamlit app with optional login system and 4 tabs.

Auth: Uses streamlit-authenticator if auth_config.yaml exists.
      Falls back to no-auth mode if the file is missing.

Setup:
  pip install -r requirements.txt
  pip install streamlit-authenticator pyyaml  # Optional for auth
  python train_all_models.py
  python auth_setup.py                        # Optional: generates credentials
  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import joblib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ---- Page Config (must be first st command) ----
st.set_page_config(
    page_title="Forest Fire Prediction System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Styling ----
st.markdown("""
<style>
    .main-header { font-size:2.2rem; font-weight:700; color:#e74c3c; text-align:center; margin-bottom:0.3rem; }
    .sub-header { font-size:1rem; color:#7f8c8d; text-align:center; margin-bottom:1.5rem; }
    .risk-high { color:#e74c3c; font-weight:700; font-size:1.4rem; }
    .risk-medium { color:#f39c12; font-weight:700; font-size:1.4rem; }
    .risk-low { color:#27ae60; font-weight:700; font-size:1.4rem; }
    .metric-card { background:var(--secondary-background-color,#f8f9fa); border-radius:10px; padding:1.2rem; margin-bottom:0.8rem; border-left:4px solid #e74c3c; }
    .winner-card { background:var(--secondary-background-color,#f8f9fa); border-radius:10px; padding:1.2rem; margin-bottom:0.8rem; border-left:4px solid #27ae60; }
</style>
""", unsafe_allow_html=True)


# ==================================================================
# AUTHENTICATION
# ==================================================================
def check_auth():
    """
    Handle login. Returns True if authenticated or auth disabled.
    
    How it works:
    1. If auth_config.yaml doesn't exist → skip auth, return True
    2. If it exists → show login form, validate bcrypt-hashed password
    3. On success → set session cookie, show logout in sidebar
    4. On failure → show error, return False (st.stop() halts app)
    """
    config_path = 'auth_config.yaml'
    if not os.path.exists(config_path):
        return True

    try:
        import streamlit_authenticator as stauth
        import yaml
    except ImportError:
        st.sidebar.warning("Auth config found but library missing. Running without auth.")
        return True

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
    )

    name, authentication_status, username = authenticator.login('Login', 'main')

    if authentication_status:
        authenticator.logout('Logout', 'sidebar')
        st.sidebar.markdown(f"**Logged in as:** {name}")
        st.sidebar.markdown("---")
        return True
    elif authentication_status is False:
        st.error('Incorrect username or password.')
        st.caption('Demo account: `demo` / `demo1234`')
        return False
    else:
        st.markdown('<p class="main-header">🔥 Forest Fire Prediction System</p>', unsafe_allow_html=True)
        st.info('Please log in to access the system.')
        st.caption('Demo account: `demo` / `demo1234`')
        return False

if not check_auth():
    st.stop()

# ==================================================================
# HEADER (shown after auth passes)
# ==================================================================
st.markdown('<p class="main-header">🔥 Forest Fire Prediction System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">CI601 — Comparing ML Approaches for Wildfire Risk Assessment</p>', unsafe_allow_html=True)


# ==================================================================
# LOAD MODELS
# ==================================================================
@st.cache_resource
def load_models():
    models = {}
    for name, path in [('Random Forest', 'trained_models/rf_fire_model.joblib'),
                        ('XGBoost', 'trained_models/xgb_fire_model.joblib'),
                        ('Neural Network', 'trained_models/nn_fire_model.joblib')]:
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_pipeline():
    path = 'trained_models/data_pipeline.joblib'
    return joblib.load(path) if os.path.exists(path) else None

@st.cache_data
def load_results():
    r = {}
    for key, path in [('comparison', 'results/comparison_results.json'),
                       ('significance', 'results/significance_tests.json')]:
        if os.path.exists(path):
            with open(path) as f:
                r[key] = json.load(f)
    if os.path.exists('results/cross_validation_results.csv'):
        r['cv'] = pd.read_csv('results/cross_validation_results.csv')
    return r

models = load_models()
pipeline = load_pipeline()

# Sidebar status
st.sidebar.header("System status")
for n in ['Random Forest', 'XGBoost', 'Neural Network']:
    st.sidebar.text(f"{n}: {'✅' if n in models else '❌'}")
st.sidebar.text(f"Pipeline: {'✅' if pipeline else '❌'}")
if not models or not pipeline:
    st.sidebar.error("Run `python train_all_models.py` first!")


# ==================================================================
# TABS
# ==================================================================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Predict", "📊 Compare", "🔥 Simulate", "ℹ️ About"])

# ---- TAB 1: PREDICT ----
with tab1:
    st.header("Fire ignition prediction")
    st.write("Enter conditions below. All three models predict independently.")

    ci, cr = st.columns([1, 1])
    with ci:
        st.subheader("Environmental parameters")
        temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0, 0.5)
        humidity = st.slider("Relative humidity (%)", 0, 100, 50, 1)
        wind = st.slider("Wind speed (km/h)", 0.0, 30.0, 5.0, 0.5)
        rain = st.slider("Rainfall (mm)", 0.0, 10.0, 0.0, 0.1)
        st.subheader("FWI components")
        ffmc = st.slider("FFMC", 0.0, 101.0, 80.0, 0.1, help="Fine Fuel Moisture Code")
        dmc = st.slider("DMC", 0.0, 300.0, 50.0, 1.0, help="Duff Moisture Code")
        dc = st.slider("DC", 0.0, 900.0, 200.0, 1.0, help="Drought Code")
        isi = st.slider("ISI", 0.0, 30.0, 5.0, 0.1, help="Initial Spread Index")
        month = st.selectbox("Month", list(range(1,13)), format_func=lambda m:
            ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1], index=7)
        predict_btn = st.button("🔥 Predict fire risk", type="primary", use_container_width=True)

    with cr:
        if predict_btn and pipeline and models:
            st.subheader("Model predictions")
            ms, mc = float(np.sin(2*np.pi*month/12)), float(np.cos(2*np.pi*month/12))
            inp = {'temperature':temperature, 'relative_humidity':float(humidity),
                   'wind_speed':wind, 'rain':rain, 'FFMC':ffmc, 'DMC':dmc,
                   'DC':dc, 'ISI':isi, 'region_encoded':0, 'month_sin':ms, 'month_cos':mc}
            try:
                X = pipeline.transform_single_input(inp)
            except Exception as e:
                st.error(f"Transform error: {e}"); X = None

            if X is not None:
                preds = {}
                for mn, mdl in models.items():
                    try:
                        fp = mdl.predict_proba(X)[0][1] * 100
                    except:
                        fp = 0.0
                    preds[mn] = fp
                    rc = "risk-high" if fp >= 70 else ("risk-medium" if fp >= 40 else "risk-low")
                    rl = "HIGH RISK" if fp >= 70 else ("MEDIUM RISK" if fp >= 40 else "LOW RISK")
                    st.markdown(f'<div class="metric-card"><strong>{mn}</strong><br><span class="{rc}">{fp:.1f}% — {rl}</span></div>', unsafe_allow_html=True)

                mx = max(preds, key=preds.get); mn_ = min(preds, key=preds.get)
                sp = max(preds.values()) - min(preds.values())
                st.markdown(f'<div class="winner-card"><strong>Model agreement</strong><br>Most alarmed: <strong>{mx}</strong> ({preds[mx]:.1f}%)<br>Least alarmed: <strong>{mn_}</strong> ({preds[mn_]:.1f}%)<br>Spread: <strong>{sp:.1f}%</strong></div>', unsafe_allow_html=True)
        elif predict_btn:
            st.error("Models not loaded.")
        else:
            st.info("Adjust sliders and click **Predict fire risk**.")
            st.markdown("**Try:** Temp 38°C, Humidity 15%, FFMC 95, DC 700 (extreme heat)")


# ---- TAB 2: COMPARE ----
with tab2:
    st.header("Model performance comparison")
    results = load_results()
    if 'comparison' not in results:
        st.warning("Run `python train_all_models.py` first."); st.stop()

    comp = results['comparison']
    st.subheader("Performance metrics")
    rows = []
    for n, r in comp.items():
        fm = r.get('classification_report',{}).get('Fire',{})
        rows.append({'Model':n, 'AUC-ROC':f"{r.get('auc_roc',0):.3f}",
                     'Accuracy':f"{r.get('accuracy',0):.3f}",
                     'Precision':f"{fm.get('precision',0):.3f}",
                     'Recall':f"{fm.get('recall',0):.3f}",
                     'F1':f"{fm.get('f1-score',0):.3f}"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    for path, col, cap in [('figures/roc_comparison.png', c1, 'ROC curves'),
                            ('figures/precision_recall.png', c2, 'Precision-Recall')]:
        if os.path.exists(path):
            with col: st.image(path, caption=cap)

    if 'cv' in results:
        st.subheader("Cross-validation (5-fold)")
        cv = results['cv']
        s = cv.groupby('Model').agg(AUC_m=('AUC-ROC','mean'), AUC_s=('AUC-ROC','std'),
                                     F1_m=('F1','mean'), F1_s=('F1','std')).round(4)
        cvr = [{'Model':n, 'AUC-ROC':f"{r['AUC_m']:.3f} ± {r['AUC_s']:.3f}",
                'F1':f"{r['F1_m']:.3f} ± {r['F1_s']:.3f}"} for n, r in s.iterrows()]
        st.dataframe(pd.DataFrame(cvr), use_container_width=True, hide_index=True)

    if 'significance' in results:
        st.subheader("Statistical significance (paired t-tests)")
        sig = results['significance']
        sr = [{'Comparison':p, 'Mean AUC diff':f"{r['mean_diff']:+.4f}",
               'p-value':f"{r['p_value']:.4f}",
               'Significant?':'✅ Yes' if r['significant'] else '❌ No'}
              for p, r in sig.items()]
        st.dataframe(pd.DataFrame(sr), use_container_width=True, hide_index=True)

    c3, c4 = st.columns(2)
    for path, col, cap in [('figures/calibration_curves.png', c3, 'Calibration'),
                            ('figures/feature_importance.png', c4, 'Feature importance')]:
        if os.path.exists(path):
            with col: st.image(path, caption=cap)

    if os.path.exists('figures/confusion_matrices.png'):
        st.subheader("Confusion matrices")
        st.image('figures/confusion_matrices.png', use_container_width=True,
                 caption='Bottom-left = missed fires (worst outcome)')


# ---- TAB 3: SIMULATE ----
with tab3:
    st.header("Fire spread simulation")
    cp, cv_ = st.columns([1, 2])
    with cp:
        gs = st.slider("Grid size", 20, 80, 40, 10)
        sws = st.slider("Wind speed (km/h)", 0.0, 15.0, 5.0, 1.0, key='sws')
        swd = st.slider("Wind direction (°)", 0, 359, 90, 15, help="0=N, 90=E, 180=S, 270=W")
        smo = st.slider("Vegetation moisture", 0.0, 1.0, 0.4, 0.05)
        ssp = st.slider("Base spread prob", 0.1, 0.8, 0.4, 0.05)
        sfb = st.slider("Firebreak density", 0.0, 0.15, 0.03, 0.01)
        sst = st.slider("Max steps", 20, 200, 80, 10)
        ign = st.radio("Ignition", ["Centre", "Random", "Multiple (3)"])
        run = st.button("▶️ Run simulation", type="primary", use_container_width=True)

    with cv_:
        if run:
            from spread_model.cellular_automata import CellularAutomataEngine, EnvironmentConfig
            cfg = EnvironmentConfig(wind_speed=sws, wind_direction=float(swd),
                                     base_spread_prob=ssp, vegetation_moisture=smo)
            eng = CellularAutomataEngine(gs, gs, cfg)
            eng.add_random_firebreaks(sfb)
            if ign == "Centre": eng.ignite(gs//2, gs//2)
            elif ign == "Random": eng.ignite(np.random.randint(5,gs-5), np.random.randint(5,gs-5))
            else:
                for _ in range(3): eng.ignite(np.random.randint(5,gs-5), np.random.randint(5,gs-5))

            prog = st.progress(0, text="Simulating...")
            eng.history = [eng.grid.copy()]
            for i in range(sst):
                if not eng.step(): break
                prog.progress((i+1)/sst, text=f"Step {i+1}")
            prog.empty()

            colors = ['#228B22','#FF4500','#2F2F2F','#4169E1']
            cmap = ListedColormap(colors)
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            axes[0].imshow(eng.grid, cmap=cmap, vmin=0, vmax=3)
            patches_list = [mpatches.Patch(color=c, label=l) for c, l in
                           zip(colors, ['Unburnt','Burning','Burnt','Firebreak'])]
            axes[0].legend(handles=patches_list, loc='upper right', fontsize=8)
            axes[0].set_title(f'Final — Step {eng.stats["steps_to_completion"]}\nWind: {swd}° @ {sws} km/h')
            axes[0].axis('off')

            pm = eng.get_spread_probability_map()
            if pm.max() > 0:
                im = axes[1].imshow(pm, cmap='YlOrRd', vmin=0, vmax=1)
                plt.colorbar(im, ax=axes[1], label='Spread probability', fraction=0.046)
                axes[1].set_title('Spread risk map')
            else:
                bm = np.zeros_like(eng.grid, dtype=float)
                bm[eng.grid==2] = 1.0; bm[eng.grid==3] = -0.5
                axes[1].imshow(bm, cmap='RdYlGn_r', vmin=-0.5, vmax=1)
                axes[1].set_title('Burn map')
            axes[1].axis('off')
            plt.tight_layout(); st.pyplot(fig); plt.close()

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Burnt", eng.stats['total_burnt'])
            c2.metric("Burn %", f"{eng.stats['burn_fraction']*100:.1f}%")
            c3.metric("Peak fire", eng.stats['max_burning'])
            c4.metric("Steps", eng.stats['steps_to_completion'])

            bf = eng.stats['burn_fraction']
            if bf > 0.8: st.warning(f"🔥 {bf*100:.0f}% burned — near-total destruction.")
            elif bf > 0.4: st.info(f"Moderate burn ({bf*100:.0f}%).")
            else: st.success(f"Contained ({bf*100:.0f}%).")
        else:
            if os.path.exists('figures/wind_comparison.png'):
                st.image('figures/wind_comparison.png', caption='Fire spread under 8 wind directions')
            st.info("Click **Run simulation** to start.")


# ---- TAB 4: ABOUT ----
with tab4:
    st.header("About this project")
    st.markdown("""
    **Forest Fire Ignition Prediction and Spread Modelling System Using Machine Learning**

    CI601 Individual Project — University of Brighton — 2026

    ---

    **Three approaches compared:**
    1. **Random Forest** — Bagging ensemble with built-in feature importance
    2. **XGBoost** — Sequential boosting with native class imbalance handling
    3. **Neural Network** — MLP testing whether deep learning helps on small tabular data

    **Evaluation:** 7-layer framework (AUC-ROC, Precision-Recall, 5-fold CV, paired t-tests, calibration, feature importance, confusion matrices)

    **Fire spread:** Cellular automaton with wind, moisture, and vegetation factors

    **Datasets:** UCI Forest Fires (517 samples) + Algerian Forest Fires (244 samples) = 761 combined
    """)
    c1, c2 = st.columns(2)
    for p, c, cap in [('figures/class_distribution.png', c1, 'Class distribution'),
                       ('figures/correlation_heatmap.png', c2, 'Feature correlations')]:
        if os.path.exists(p):
            with c: st.image(p, caption=cap)
    if os.path.exists('figures/feature_distributions.png'):
        st.image('figures/feature_distributions.png', caption='Feature distributions', use_container_width=True)
