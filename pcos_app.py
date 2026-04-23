import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PCOS Diagnostic Tool",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# LOAD EXTERNAL CSS
# ─────────────────────────────────────────────────────────
def load_css(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load CSS: {e}")

load_css("style.css")

# ─────────────────────────────────────────────────────────
# PHENOTYPE INFO
# ─────────────────────────────────────────────────────────
PHENOTYPE_INFO = {
    "A": {
        "label": "Phenotype A", "sublabel": "Full Classic PCOS",
        "color": "#7c52cc",
        "description": "Anovulation + Hyperandrogenism + Polycystic Ovaries. The most common and most severe phenotype.",
        "features": ["Irregular menstrual cycles (anovulation)", "Elevated androgens (hirsutism, acne)", "Polycystic ovaries on ultrasound"],
    },
    "B": {
        "label": "Phenotype B", "sublabel": "Classic without Polycystic Ovaries",
        "color": "#b83232",
        "description": "Anovulation + Hyperandrogenism, but with normal ovarian morphology.",
        "features": ["Irregular menstrual cycles", "Elevated androgens", "Normal ovarian morphology"],
    },
    "C": {
        "label": "Phenotype C", "sublabel": "Ovulatory PCOS",
        "color": "#1a6e3c",
        "description": "Hyperandrogenism + Polycystic Ovaries, but with regular menstrual cycles.",
        "features": ["Regular menstrual cycles", "Elevated androgens", "Polycystic ovaries on ultrasound"],
    },
    "D": {
        "label": "Phenotype D", "sublabel": "Non-Androgenic PCOS",
        "color": "#9a7010",
        "description": "Anovulation + Polycystic Ovaries, but without hyperandrogenism. The mildest phenotype.",
        "features": ["Irregular menstrual cycles (anovulation)", "No hyperandrogenism signs", "Polycystic ovaries on ultrasound"],
    },
}

# Normal reference ranges for lab markers
NORMAL_RANGES = {
    "BMI":          (18.5, 24.9, "kg/m²"),
    "AMH":          (1.0,  3.5,  "ng/mL"),
    "FSH":          (3.0,  10.0, "mIU/mL"),
    "LH":           (2.0,  15.0, "mIU/mL"),
    "FSH/LH":       (1.0,  3.0,  "ratio"),
    "TSH":          (0.4,  4.0,  "mIU/L"),
    "PRL":          (2.0,  29.0, "ng/mL"),
    "Hemoglobin":   (12.0, 16.0, "g/dl"),
    "Vitamin D3":   (20.0, 50.0, "ng/mL"),
    "Progesterone": (1.0,  25.0, "ng/mL"),
    "RBS":          (70.0, 140.0,"mg/dl"),
    "Waist:Hip":    (0.0,  0.85, "ratio"),
}

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
SECTIONS = ["anthropometric", "vitals", "menstrual", "labs", "ultrasound", "symptoms"]
SECTION_LABELS = {
    "anthropometric": ("01", "Anthropometric Measurements", "Height, weight, BMI, body ratios"),
    "vitals":         ("02", "Vitals",                      "Pulse and blood pressure"),
    "menstrual":      ("03", "Menstrual & Reproductive History", "Cycle regularity, pregnancy history"),
    "labs":           ("04", "Laboratory Values",           "Hormones, blood markers"),
    "ultrasound":     ("05", "Ultrasound Findings",         "Follicle count and size, endometrium"),
    "symptoms":       ("06", "Clinical Symptoms",           "Self-reported signs and lifestyle"),
}

DEFAULTS = {
    "active_section": 0,
    "section_data": {},
    "app_step": "form",
    "pcos_result": None,
    "phenotype_result": None,
    "inputs": {},
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
# CUSTOM CLASSES
# ─────────────────────────────────────────────────────────
class TOMIMSelector(BaseEstimator, TransformerMixin):
    def __init__(self, random_state=42, percentile=0):
        self.random_state  = random_state
        self.percentile    = percentile
        self.selected_idx_ = None
        self.threshold_    = None

    def fit(self, X, y):
        mi = mutual_info_classif(X, y, random_state=self.random_state)
        self.threshold_    = np.percentile(mi, self.percentile)
        self.selected_idx_ = np.where(mi >= self.threshold_)[0]
        if len(self.selected_idx_) == 0:
            self.selected_idx_ = np.argsort(mi)[-5:]
        return self

    def transform(self, X):
        return X[:, self.selected_idx_]

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return [input_features[i] for i in self.selected_idx_]
        return self.selected_idx_

# ─────────────────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    p1 = pickle.load(open("p1_model.pkl", "rb"))
    p2 = pickle.load(open("p2_model.pkl", "rb"))
    return p1, p2

p1_model, p2_model = load_models()

# ─────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────
P1_FEATURES = [
    "Hb(g/dl)", "I   beta-HCG(mIU/mL)", "II    beta-HCG(mIU/mL)",
    "FSH(mIU/mL)", "LH(mIU/mL)", "FSH/LH", "Waist(inch)",
    "TSH (mIU/L)", "AMH(ng/mL)", "PRL(ng/mL)", "Vit D3 (ng/mL)",
    "PRG(ng/mL)", "RBS(mg/dl)", "Follicle No. (L)", "Follicle No. (R)",
    "Avg. F size (L) (mm)",
]

P2_FEATURES = [
    "age", "weight", "height", "bmi", "blood group",
    "pulse rate (bpm)", "cycle (2/4)", "marraige status (yrs)",
    "pregnant (1/0)", "no. of abortions", "i   beta-hcg(miu/ml)",
    "hip (inch)", "waist (inch)", "waist:hip ratio",
    "amh (ng/ml)", "rbs (mg/dl)", "weight gain (1/0)",
    "hair growth (1/0)", "skin darkening (1/0)", "pimples (1/0)",
    "fast food (1/0)", "reg.exercise (1/0)",
    "bp _systolic (mmhg)", "bp _diastolic (mmhg)",
    "follicle no. (l)", "follicle no. (r)",
]

def predict_pcos(inp):
    X = np.array([inp.get(c, np.nan) for c in P1_FEATURES]).reshape(1, -1)
    prob = p1_model.predict_proba(X)[0]
    positive_class_index = list(p1_model.classes_).index(1)
    return prob[positive_class_index] >= 0.5

def predict_phenotype(inp):
    X = pd.DataFrame([[inp.get(c, np.nan) for c in P2_FEATURES]], columns=P2_FEATURES)
    probs_arr = p2_model.predict_proba(X)[0]
    probs = {c: round(float(p), 3) for c, p in zip(["A","B","C","D"], probs_arr)}
    return max(probs, key=probs.get), probs

def compute_shap_values(inp):
    """Compute SHAP values for P1 model."""
    try:
        import shap
        X = np.array([inp.get(c, np.nan) for c in P1_FEATURES]).reshape(1, -1)
        explainer = shap.TreeExplainer(p1_model)
        shap_vals = explainer.shap_values(X)
        # For binary classification, take positive class
        if isinstance(shap_vals, list):
            sv = shap_vals[1][0]
        else:
            sv = shap_vals[0]
        return dict(zip(P1_FEATURES, sv)), float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
    except Exception:
        return None, None

def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

# ─────────────────────────────────────────────────────────
# SIDEBAR HELPERS
# ─────────────────────────────────────────────────────────
CHECK = '<span class="nav-check">&#10003;</span>'

def _nav_row(num, label, state):
    css = f"nav-{state}"
    check = CHECK if state == "done" else ""
    badge_extra = ' style="background:rgba(124,82,204,0.2);color:#b08af5;"' if state == "done" else ""
    return (
        f'<div class="nav-section {css}">'
        f'<span class="step-badge"{badge_extra}>{num}</span>'
        f'<span style="flex:1;">{label}</span>'
        f'{check}'
        f'</div>'
    )

def _pipeline_stages(p1_state, p2_state, dash_state="locked"):
    def stage(num, title, desc, state):
        css = f"pipeline-stage stage-{state}"
        check = CHECK if state == "done" else ""
        return (
            f'<div class="{css}">'
            f'<span class="stage-badge">{num}</span>'
            f'<span style="flex:1;">'
            f'<span style="display:block;line-height:1.3;">{title}</span>'
            f'<span class="stage-desc">{desc}</span>'
            f'</span>'
            f'{check}'
            f'</div>'
        )
    html = '<div class="pipeline-label">ML Pipelines</div>'
    html += stage("P1", "PCOS Detection",         "Binary positive / negative", p1_state)
    html += stage("P2", "Phenotype Classification","Types A / B / C / D",       p2_state)
    html += stage("DB", "Clinical Dashboard",      "Charts, SHAP & summary",     dash_state)
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">PCOS Diagnostic Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">ML-Powered Clinical Assistant</div>', unsafe_allow_html=True)

    app_step   = st.session_state.app_step
    active_sec = st.session_state.active_section

    if app_step == "form":
        pct = int((active_sec / len(SECTIONS)) * 100)
        st.markdown(
            f'<div class="sidebar-progress-label">Progress &mdash; {pct}%</div>'
            f'<div class="sidebar-progress-bar-bg">'
            f'<div class="sidebar-progress-bar-fill" style="width:{pct}%;"></div>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="pipeline-label">Data Collection</div>', unsafe_allow_html=True)
    rows_html = ""
    for i, sec in enumerate(SECTIONS):
        num, label, _ = SECTION_LABELS[sec]
        if app_step == "form":
            if i == active_sec:   state = "active"
            elif i < active_sec:  state = "done"
            else:                 state = "locked"
        else:
            state = "done"
        rows_html += _nav_row(num, label, state)
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if app_step == "form":
        _pipeline_stages(p1_state="locked", p2_state="locked", dash_state="locked")
    elif app_step == "pcos_result":
        _pipeline_stages(p1_state="active", p2_state="locked", dash_state="locked")
    elif app_step == "phenotype_result":
        _pipeline_stages(p1_state="done",   p2_state="active", dash_state="locked")
    elif app_step == "dashboard":
        _pipeline_stages(p1_state="done",   p2_state="done",   dash_state="active")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if app_step != "form" or active_sec > 0:
        if st.button("Start Over", use_container_width=True):
            reset(); st.rerun()

# ─────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────
app_step   = st.session_state.app_step
active_sec = st.session_state.active_section

if app_step == "form":
    _, label, desc = SECTION_LABELS[SECTIONS[active_sec]]
    st.markdown(f"""
    <div class="main-header">
        <h1>Step {active_sec + 1} of {len(SECTIONS)} — {label}</h1>
        <p>{desc} &nbsp;&middot;&nbsp; Fill in the fields below and click <strong>Next</strong> to continue.</p>
    </div>
    """, unsafe_allow_html=True)
elif app_step == "pcos_result":
    st.markdown('<div class="main-header"><h1>PCOS Detection Result</h1><p>Based on all entered clinical data.</p></div>', unsafe_allow_html=True)
elif app_step == "phenotype_result":
    st.markdown('<div class="main-header"><h1>Phenotype Classification Result</h1><p>PCOS confirmed — classifying phenotype.</p></div>', unsafe_allow_html=True)
elif app_step == "dashboard":
    st.markdown('<div class="main-header"><h1>Clinical Dashboard</h1><p>Full diagnostic summary, biomarker analysis, and model explainability.</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────
if app_step == "form":
    sd = st.session_state.section_data

    def nav_buttons(form_key, back_idx, next_label):
        col1, col2 = st.columns([1, 4])
        with col1:
            back = st.form_submit_button("Back") if back_idx is not None else False
        with col2:
            nxt = st.form_submit_button(next_label, use_container_width=True)
        return back, nxt

    if active_sec == 0:
        with st.form("sec_0"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">01</div>
                <div><div class="section-title">Anthropometric Measurements</div>
                <div class="section-desc">Height, weight, BMI, body ratios</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            age    = c1.number_input("Age (years)",  min_value=15,   max_value=55,   value=sd.get("age",    None), placeholder="e.g. 25")
            weight = c2.number_input("Weight (kg)",  min_value=30.0, max_value=150.0,value=sd.get("weight", None), step=0.5, placeholder="e.g. 60.0")
            height = c3.number_input("Height (cm)",  min_value=130.0,max_value=200.0,value=sd.get("height", None), step=0.5, placeholder="e.g. 160.0")
            bmi = weight / ((height / 100) ** 2) if (weight and height) else None
            bmi_display = f"{bmi:.1f} kg/m²" if bmi else "—"
            c1, c2, c3, c4, c5 = st.columns(5)
            hip   = c1.number_input("Hip (inch)",   min_value=20.0, max_value=60.0, value=sd.get("hip",   None), step=0.5, placeholder="e.g. 38.0")
            waist = c2.number_input("Waist (inch)", min_value=20.0, max_value=60.0, value=sd.get("waist", None), step=0.5, placeholder="e.g. 30.0")
            whr = round(waist / hip, 3) if (waist and hip) else None
            whr_display = f"{whr:.3f}" if whr else "—"
            bg_opts = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
            blood_group = c3.selectbox("Blood Group", bg_opts, index=bg_opts.index(sd.get("blood_group","A+")))
            c4.markdown(f'<div class="auto-pill">BMI (auto-calculated)<span>{bmi_display}</span></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="auto-pill">Waist:Hip Ratio (auto)<span>{whr_display}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            _, nxt = nav_buttons("sec_0", None, "Next — Vitals")
        if nxt:
            missing = [f for f, v in [("Age",age),("Weight",weight),("Height",height),("Hip",hip),("Waist",waist)] if v is None]
            if missing: st.error(f"Please fill in: {', '.join(missing)}")
            else:
                bg_map = {"A+":11,"A-":12,"B+":13,"B-":14,"O+":15,"O-":16,"AB+":17,"AB-":18}
                sd.update({"age":age,"weight":weight,"height":height,"bmi":round(bmi,2),
                           "hip":hip,"waist":waist,"whr":whr,"blood_group":blood_group,"bg_code":bg_map[blood_group]})
                st.session_state.active_section = 1; st.rerun()

    elif active_sec == 1:
        with st.form("sec_1"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">02</div>
                <div><div class="section-title">Vitals</div>
                <div class="section-desc">Pulse and blood pressure</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            pulse  = c1.number_input("Pulse Rate (bpm)",    min_value=40, max_value=130, value=sd.get("pulse",  None), placeholder="e.g. 78")
            bp_sys = c2.number_input("BP Systolic (mmHg)",  min_value=70, max_value=200, value=sd.get("bp_sys", None), placeholder="e.g. 120")
            bp_dia = c3.number_input("BP Diastolic (mmHg)", min_value=40, max_value=130, value=sd.get("bp_dia", None), placeholder="e.g. 80")
            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_1", 0, "Next — Menstrual History")
        if nxt:
            missing = [f for f, v in [("Pulse Rate",pulse),("BP Systolic",bp_sys),("BP Diastolic",bp_dia)] if v is None]
            if missing: st.error(f"Please fill in: {', '.join(missing)}")
            else:
                sd.update({"pulse":pulse,"bp_sys":bp_sys,"bp_dia":bp_dia})
                st.session_state.active_section = 2; st.rerun()
        if back: st.session_state.active_section = 0; st.rerun()

    elif active_sec == 2:
        with st.form("sec_2"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">03</div>
                <div><div class="section-title">Menstrual & Reproductive History</div>
                <div class="section-desc">Cycle regularity, pregnancy history</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            cy_opts   = ["Regular","Irregular"]
            cycle_ri  = c1.selectbox("Menstrual Cycle", cy_opts, index=cy_opts.index(sd.get("cycle_ri_label","Regular")))
            marriage  = c2.number_input("Marriage Duration (years)", min_value=0, max_value=40, value=sd.get("marriage_yr", None), placeholder="e.g. 2")
            pr_opts   = ["No","Yes"]
            pregnant  = c3.selectbox("Currently Pregnant?", pr_opts, index=pr_opts.index(sd.get("pregnant_label","No")))
            abortions = c4.number_input("No. of Abortions", min_value=0, max_value=10, value=sd.get("abortions", None), placeholder="e.g. 0")
            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_2", 1, "Next — Laboratory Values")
        if nxt:
            missing = [f for f, v in [("Marriage Duration",marriage),("No. of Abortions",abortions)] if v is None]
            if missing: st.error(f"Please fill in: {', '.join(missing)}")
            else:
                sd.update({"cycle_ri_label":cycle_ri,"cycle_ri":1 if cycle_ri=="Irregular" else 0,
                           "cycle_24":4 if cycle_ri=="Irregular" else 2,"marriage_yr":marriage,
                           "pregnant_label":pregnant,"pregnant":1 if pregnant=="Yes" else 0,"abortions":abortions})
                st.session_state.active_section = 3; st.rerun()
        if back: st.session_state.active_section = 1; st.rerun()

    elif active_sec == 3:
        with st.form("sec_3"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">04</div>
                <div><div class="section-title">Laboratory Values</div>
                <div class="section-desc">Hormones and blood markers</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            hb          = c1.number_input("Hemoglobin (g/dl)",     min_value=5.0,  max_value=20.0,  value=sd.get("hb",          None), step=0.1, placeholder="e.g. 13.0")
            beta_hcg_i  = c2.number_input("Beta-HCG I (mIU/mL)",  min_value=0.0,  max_value=500.0, value=sd.get("beta_hcg_i",  None), step=0.1, placeholder="e.g. 1.5")
            beta_hcg_ii = c3.number_input("Beta-HCG II (mIU/mL)", min_value=0.0,  max_value=500.0, value=sd.get("beta_hcg_ii", None), step=0.1, placeholder="e.g. 1.5")
            fsh         = c4.number_input("FSH (mIU/mL)",          min_value=0.0,  max_value=30.0,  value=sd.get("fsh",         None), step=0.1, placeholder="e.g. 5.0")
            c1, c2, c3, c4 = st.columns(4)
            lh    = c1.number_input("LH (mIU/mL)",  min_value=0.0, max_value=50.0,  value=sd.get("lh",  None), step=0.1, placeholder="e.g. 6.0")
            tsh   = c2.number_input("TSH (mIU/L)",  min_value=0.0, max_value=10.0,  value=sd.get("tsh", None), step=0.1, placeholder="e.g. 2.5")
            prl   = c3.number_input("PRL (ng/mL)",  min_value=0.0, max_value=100.0, value=sd.get("prl", None), step=0.1, placeholder="e.g. 15.0")
            amh   = c4.number_input("AMH (ng/mL)",  min_value=0.0, max_value=15.0,  value=sd.get("amh", None), step=0.1, placeholder="e.g. 2.0")
            c1, c2, c3, c4 = st.columns(4)
            vit_d = c1.number_input("Vitamin D3 (ng/mL)",         min_value=0.0, max_value=100.0, value=sd.get("vit_d", None), step=0.1, placeholder="e.g. 25.0")
            prg   = c2.number_input("Progesterone (ng/mL)",       min_value=0.0, max_value=30.0,  value=sd.get("prg",   None), step=0.1, placeholder="e.g. 1.5")
            rbs   = c3.number_input("Random Blood Sugar (mg/dl)", min_value=50.0,max_value=400.0, value=sd.get("rbs",   None), step=1.0, placeholder="e.g. 90")
            fsh_lh = round(fsh / lh, 3) if (fsh and lh and lh > 0) else None
            fsh_lh_display = f"{fsh_lh:.3f}" if fsh_lh else "—"
            c4.markdown(f'<div class="auto-pill">FSH/LH Ratio (auto)<span>{fsh_lh_display}</span></div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_3", 2, "Next — Ultrasound Findings")
        if nxt:
            lab_fields = [("Hemoglobin",hb),("Beta-HCG I",beta_hcg_i),("Beta-HCG II",beta_hcg_ii),
                          ("FSH",fsh),("LH",lh),("TSH",tsh),("PRL",prl),("AMH",amh),
                          ("Vitamin D3",vit_d),("Progesterone",prg),("Random Blood Sugar",rbs)]
            missing = [f for f, v in lab_fields if v is None]
            if missing: st.error(f"Please fill in: {', '.join(missing)}")
            else:
                sd.update({"hb":hb,"beta_hcg_i":beta_hcg_i,"beta_hcg_ii":beta_hcg_ii,
                           "fsh":fsh,"lh":lh,"fsh_lh":fsh_lh,"tsh":tsh,"prl":prl,
                           "amh":amh,"vit_d":vit_d,"prg":prg,"rbs":rbs})
                st.session_state.active_section = 4; st.rerun()
        if back: st.session_state.active_section = 2; st.rerun()

    elif active_sec == 4:
        with st.form("sec_4"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">05</div>
                <div><div class="section-title">Ultrasound Findings</div>
                <div class="section-desc">Follicle count and size, endometrium thickness</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            follicle_l  = c1.number_input("Follicle No. (Left)",       min_value=0,   max_value=30,   value=sd.get("follicle_l", None), placeholder="e.g. 5")
            follicle_r  = c2.number_input("Follicle No. (Right)",      min_value=0,   max_value=30,   value=sd.get("follicle_r", None), placeholder="e.g. 5")
            avg_f_l     = c3.number_input("Avg. Follicle Size L (mm)", min_value=0.0, max_value=30.0, value=sd.get("avg_f_l",   None), step=0.5, placeholder="e.g. 8.0")
            avg_f_r     = c4.number_input("Avg. Follicle Size R (mm)", min_value=0.0, max_value=30.0, value=sd.get("avg_f_r",   None), step=0.5, placeholder="e.g. 8.0")
            endometrium = st.number_input("Endometrium Thickness (mm)",min_value=0.0, max_value=20.0, value=sd.get("endometrium", None), step=0.1, placeholder="e.g. 7.0")
            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_4", 3, "Next — Clinical Symptoms")
        if nxt:
            missing = [f for f, v in [("Follicle No. Left",follicle_l),("Follicle No. Right",follicle_r),
                                       ("Avg. Follicle Size L",avg_f_l),("Avg. Follicle Size R",avg_f_r),
                                       ("Endometrium Thickness",endometrium)] if v is None]
            if missing: st.error(f"Please fill in: {', '.join(missing)}")
            else:
                sd.update({"follicle_l":follicle_l,"follicle_r":follicle_r,
                           "avg_f_l":avg_f_l,"avg_f_r":avg_f_r,"endometrium":endometrium})
                st.session_state.active_section = 5; st.rerun()
        if back: st.session_state.active_section = 3; st.rerun()

    elif active_sec == 5:
        with st.form("sec_5"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">06</div>
                <div><div class="section-title">Clinical Symptoms</div>
                <div class="section-desc">Self-reported signs and lifestyle factors</div></div>
            </div>""", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            yn = ["No","Yes"]
            weight_gain = c1.radio("Weight Gain?",         yn, index=yn.index(sd.get("weight_gain_label","No")), horizontal=True)
            hair_growth = c2.radio("Excess Hair Growth?",  yn, index=yn.index(sd.get("hair_growth_label","No")), horizontal=True)
            skin_dark   = c3.radio("Skin Darkening?",      yn, index=yn.index(sd.get("skin_dark_label","No")),   horizontal=True)
            c1, c2, c3 = st.columns(3)
            pimples   = c1.radio("Pimples / Acne?",        yn, index=yn.index(sd.get("pimples_label","No")),   horizontal=True)
            fast_food = c2.radio("Fast Food (regularly)?", yn, index=yn.index(sd.get("fast_food_label","No")), horizontal=True)
            exercise  = c3.radio("Regular Exercise?",      yn, index=yn.index(sd.get("exercise_label","No")),  horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_5", 4, "Run PCOS Detection")
        if nxt:
            sd.update({
                "weight_gain_label":weight_gain, "weight_gain":1 if weight_gain=="Yes" else 0,
                "hair_growth_label":hair_growth, "hair_growth":1 if hair_growth=="Yes" else 0,
                "skin_dark_label":skin_dark,     "skin_darkening":1 if skin_dark=="Yes" else 0,
                "pimples_label":pimples,         "pimples":1 if pimples=="Yes" else 0,
                "fast_food_label":fast_food,     "fast_food":1 if fast_food=="Yes" else 0,
                "exercise_label":exercise,       "exercise":1 if exercise=="Yes" else 0,
            })
            s = sd
            inp = {
                "Hb(g/dl)":s["hb"], "I   beta-HCG(mIU/mL)":s["beta_hcg_i"],
                "II    beta-HCG(mIU/mL)":s["beta_hcg_ii"], "FSH(mIU/mL)":s["fsh"],
                "LH(mIU/mL)":s["lh"], "FSH/LH":s["fsh_lh"], "Waist(inch)":s["waist"],
                "TSH (mIU/L)":s["tsh"], "AMH(ng/mL)":s["amh"], "PRL(ng/mL)":s["prl"],
                "Vit D3 (ng/mL)":s["vit_d"], "PRG(ng/mL)":s["prg"], "RBS(mg/dl)":s["rbs"],
                "Follicle No. (L)":s["follicle_l"], "Follicle No. (R)":s["follicle_r"],
                "Avg. F size (L) (mm)":s["avg_f_l"],
                "age":s["age"], "weight":s["weight"], "height":s["height"], "bmi":s["bmi"],
                "blood group":s["bg_code"], "pulse rate (bpm)":s["pulse"],
                "cycle (2/4)":s["cycle_24"], "marraige status (yrs)":s["marriage_yr"],
                "pregnant (1/0)":s["pregnant"], "no. of abortions":s["abortions"],
                "i   beta-hcg(miu/ml)":s["beta_hcg_i"], "hip (inch)":s["hip"],
                "waist (inch)":s["waist"], "waist:hip ratio":s["whr"],
                "amh (ng/ml)":s["amh"], "rbs (mg/dl)":s["rbs"],
                "weight gain (1/0)":s["weight_gain"], "hair growth (1/0)":s["hair_growth"],
                "skin darkening (1/0)":s["skin_darkening"], "pimples (1/0)":s["pimples"],
                "fast food (1/0)":s["fast_food"], "reg.exercise (1/0)":s["exercise"],
                "bp _systolic (mmhg)":s["bp_sys"], "bp _diastolic (mmhg)":s["bp_dia"],
                "follicle no. (l)":s["follicle_l"], "follicle no. (r)":s["follicle_r"],
                "cycle_ri":s["cycle_ri"],
            }
            st.session_state.inputs = inp
            pcos_pos = predict_pcos(inp)
            st.session_state.pcos_result = pcos_pos
            st.session_state.app_step = "pcos_result"
            st.rerun()
        if back: st.session_state.active_section = 4; st.rerun()

# ─────────────────────────────────────────────────────────
# PCOS RESULT
# ─────────────────────────────────────────────────────────
elif app_step == "pcos_result":
    pcos_pos = st.session_state.pcos_result
    if pcos_pos:
        st.markdown("""
        <div class="result-positive">
            <div class="result-positive-accent">!</div>
            <div class="result-title">PCOS Detected</div>
            <div class="result-subtitle">The model predicts a <strong>positive</strong> result for Polycystic Ovary Syndrome.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Proceed to Phenotype Classification", use_container_width=True):
                st.session_state.app_step = "phenotype_result"; st.rerun()
        with col2:
            if st.button("Start Over", use_container_width=True):
                reset(); st.rerun()
    else:
        st.markdown("""
        <div class="result-negative">
            <div class="result-negative-accent">&#10003;</div>
            <div class="result-title" style="color:#154030;">No PCOS Detected</div>
            <div class="result-subtitle">The model predicts a <strong>negative</strong> result. Phenotype classification will not proceed.</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Over with New Patient", use_container_width=True):
            reset(); st.rerun()
    st.markdown("""<div class="disclaimer"><strong>Clinical Disclaimer:</strong>
        This tool is for research and educational purposes only. Results do not constitute a medical diagnosis
        and must be confirmed by a licensed healthcare professional.</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# PHENOTYPE RESULT
# ─────────────────────────────────────────────────────────
elif app_step == "phenotype_result":
    inp = st.session_state.inputs
    ph, probs = predict_phenotype(inp)
    st.session_state.phenotype_result = (ph, probs)
    info = PHENOTYPE_INFO[ph]

    st.markdown(f"""
    <div class="phenotype-card">
        <div style="width:52px;height:52px;background:{info['color']};border-radius:50%;
                    display:flex;align-items:center;justify-content:center;margin:0 auto 0.9rem;
                    color:white;font-family:'Cormorant Garamond',serif;font-size:1.2rem;font-weight:600;">
            {ph}
        </div>
        <div class="result-title">{info['label']} — {info['sublabel']}</div>
        <div class="result-subtitle">{info['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Phenotype Probabilities")
        ph_colors = {"A":"#7c52cc","B":"#b83232","C":"#1a6e3c","D":"#9a7010"}
        for pk in ["A","B","C","D"]:
            pv  = probs[pk]; bw = int(pv * 100)
            sty = "font-weight:700;" if pk == ph else "opacity:0.55;"
            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-label" style="{sty}">
                    <span>Phenotype {pk} — {PHENOTYPE_INFO[pk]['sublabel']}</span>
                    <span>{bw}%</span>
                </div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{bw}%; background:{ph_colors[pk]};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown("#### Clinical Features of this Phenotype")
        for feat in info["features"]:
            st.markdown(f"&bull; {feat}")
        st.markdown("#### Key Indicators Found")
        findings = []
        if inp.get("cycle_ri") == 1:             findings.append(("Irregular menstrual cycle","tag-red"))
        if inp.get("hair growth (1/0)") == 1:    findings.append(("Excess hair growth","tag-red"))
        if inp.get("pimples (1/0)") == 1:        findings.append(("Acne / pimples","tag-orange"))
        if inp.get("weight gain (1/0)") == 1:    findings.append(("Weight gain","tag-orange"))
        if inp.get("skin darkening (1/0)") == 1: findings.append(("Skin darkening","tag-yellow"))
        amh_v = inp.get("amh (ng/ml)", 0) or 0
        if amh_v > 3.5:                          findings.append((f"Elevated AMH ({amh_v:.1f} ng/mL)","tag-red"))
        fl = inp.get("follicle no. (l)", 0) or 0
        fr = inp.get("follicle no. (r)", 0) or 0
        if fl > 10 or fr > 10:                   findings.append((f"Polycystic ovaries (L:{fl} R:{fr})","tag-red"))
        if not findings:                          findings.append(("No major hyperandrogenism markers","tag-green"))
        tags = "".join(f'<span class="finding-tag {cls}">{txt}</span>' for txt, cls in findings)
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Clinical Dashboard", use_container_width=True):
            st.session_state.app_step = "dashboard"; st.rerun()
    with col2:
        if st.button("Start Over with New Patient", use_container_width=True):
            reset(); st.rerun()

    st.markdown("""<div class="disclaimer"><strong>Clinical Disclaimer:</strong>
        This tool is for research and educational purposes only. Phenotype classification must be confirmed
        by a licensed OB-GYN or reproductive endocrinologist using the Rotterdam criteria.</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────
elif app_step == "dashboard":
    inp = st.session_state.inputs
    sd  = st.session_state.section_data
    ph, probs = st.session_state.phenotype_result
    info = PHENOTYPE_INFO[ph]

    PURPLE = "#7c52cc"
    RED    = "#c0392b"
    GREEN  = "#27ae60"

    # ─────────────────────────────────────────
    # 🧾 1. CLINICAL SUMMARY
    # ─────────────────────────────────────────
    st.markdown("## 🧾 Clinical Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Diagnosis", "PCOS Positive" if st.session_state.pcos_result else "Negative")
    col2.metric("Phenotype", f"{info['label']} ({ph})")
    col3.metric("BMI", f"{sd.get('bmi','—')}")

    st.markdown("---")

    # ─────────────────────────────────────────
    # 🚨 2. KEY RISK FLAGS (ONLY ABNORMAL)
    # ─────────────────────────────────────────
    st.markdown("## 🚨 Key Risk Indicators")

    risk_items = [
        ("AMH", sd.get("amh", 0), 1.0, 3.5, "ng/mL"),
        ("FSH/LH", sd.get("fsh_lh", 0), 1.0, 3.0, ""),
        ("BMI", sd.get("bmi", 0), 18.5, 24.9, ""),
        ("Waist:Hip", sd.get("whr", 0), 0.0, 0.85, ""),
        ("Vitamin D", sd.get("vit_d", 0), 20.0, 50.0, "ng/mL"),
        ("RBS", sd.get("rbs", 0), 70.0, 140.0, "mg/dl"),
    ]

    flags = []
    for name, val, lo, hi, unit in risk_items:
        if val < lo:
            flags.append((name, val, "Low"))
        elif val > hi:
            flags.append((name, val, "High"))

    if flags:
        for name, val, level in flags:
            color = RED if level == "High" else "#d68910"
            st.markdown(
                f"<span style='background:{color}22;color:{color};padding:6px 12px;"
                f"border-radius:20px;margin-right:6px;font-size:0.8rem;'>"
                f"{name}: {val} ({level})</span>",
                unsafe_allow_html=True
            )
    else:
        st.success("All key markers within normal range")

    st.markdown("---")

    # ─────────────────────────────────────────
    # 📊 3. PHENOTYPE PROBABILITIES
    # ─────────────────────────────────────────
    st.markdown("## 📊 Phenotype Probabilities")

    fig_prob = go.Figure(go.Bar(
        x=list(probs.values()),
        y=[f"Phenotype {k}" for k in probs.keys()],
        orientation="h",
        text=[f"{int(v*100)}%" for v in probs.values()],
        textposition="auto"
    ))

    fig_prob.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20)
    )

    st.plotly_chart(fig_prob, use_container_width=True)

    st.markdown("---")

    # ─────────────────────────────────────────
    # 📉 4. ABNORMAL LAB VISUALIZATION
    # ─────────────────────────────────────────
    st.markdown("## 📉 Abnormal Lab Markers")

    markers = [
        ("AMH", sd.get("amh", 0), 1.0, 3.5),
        ("FSH", sd.get("fsh", 0), 3.0, 10.0),
        ("LH", sd.get("lh", 0), 2.0, 15.0),
        ("TSH", sd.get("tsh", 0), 0.4, 4.0),
        ("Vitamin D", sd.get("vit_d", 0), 20.0, 50.0),
        ("RBS", sd.get("rbs", 0), 70.0, 140.0),
    ]

    abnormal = [(n, v) for n, v, lo, hi in markers if v < lo or v > hi]

    if abnormal:
        fig_abn = go.Figure(go.Bar(
            x=[v for _, v in abnormal],
            y=[n for n, _ in abnormal],
            orientation="h",
        ))
        fig_abn.update_layout(height=250)
        st.plotly_chart(fig_abn, use_container_width=True)
    else:
        st.info("No abnormal lab values detected")

    st.markdown("---")

    # ─────────────────────────────────────────
    # 🧠 5. SHAP (COLLAPSIBLE)
    # ─────────────────────────────────────────
    with st.expander("🧠 Model Explainability (SHAP)"):
        with st.spinner("Computing SHAP values..."):
            shap_vals, _ = compute_shap_values(inp)

        if shap_vals:
            sorted_shap = sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

            labels = [k for k, _ in sorted_shap]
            values = [v for _, v in sorted_shap]

            fig_shap = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h"
            ))

            fig_shap.update_layout(height=300)
            st.plotly_chart(fig_shap, use_container_width=True)

            st.caption("Positive values increase PCOS likelihood; negative values decrease it.")
        else:
            st.warning("SHAP not available. Install with: pip install shap")

    # ─────────────────────────────────────────
    # 📋 6. FULL DATA (COLLAPSIBLE)
    # ─────────────────────────────────────────
    with st.expander("📋 View Full Patient Data"):
        df = pd.DataFrame(list(sd.items()), columns=["Feature", "Value"])
        st.dataframe(df, use_container_width=True)

    # ─────────────────────────────────────────
    # 🔁 ACTIONS
    # ─────────────────────────────────────────
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Back to Results", use_container_width=True):
            st.session_state.app_step = "phenotype_result"
            st.rerun()

    with col2:
        if st.button("Start New Patient", use_container_width=True):
            reset()
            st.rerun()

    st.markdown("""<div class="disclaimer"><strong>Clinical Disclaimer:</strong>
        This dashboard is for research and educational purposes only. All findings must be interpreted
        by a licensed OB-GYN or reproductive endocrinologist. SHAP values reflect model behaviour,
        not clinical causation.</div>""", unsafe_allow_html=True)