import streamlit as st
import numpy as np
import pandas as pd
import pickle
import traceback
import base64
import json
import plotly.graph_objects as go
import streamlit.components.v1 as components
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

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

OVARY_IMG     = img_b64("ovarian_morphology.jpg")
PHENOTYPES_IMG = img_b64("phenotypes.jpg")

# ─────────────────────────────────────────────────────────
# INJECT FIXES
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none !important; }
input::placeholder { color: #b0a8c8 !important; font-style: italic; font-size: 0.83rem; }
input[type="number"] { -moz-appearance: textfield; }
input[type="number"]::-webkit-inner-spin-button,
input[type="number"]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
</style>
<script>
document.addEventListener('keydown', function(e) {
    if (e.target.type === 'number') {
        if (['e','E','-','+'].includes(e.key)) e.preventDefault();
    }
}, true);
</script>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# ROTTERDAM RULE-BASED PCOS DETECTION
# ─────────────────────────────────────────────────────────
def evaluate_rotterdam(inp):
    oa  = bool(inp.get("cycle_ri", 0))
    ha  = bool(
        inp.get("hair growth (1/0)", 0)
        or inp.get("pimples (1/0)", 0)
        or inp.get("skin darkening (1/0)", 0)
    )
    fl  = inp.get("follicle no. (l)", 0) or 0
    fr  = inp.get("follicle no. (r)", 0) or 0
    pcom = (fl >= 12) or (fr >= 12)
    return oa, ha, pcom


def classify_phenotype_rule(oa, ha, pcom):
    if   oa and ha and pcom: return "A"
    elif oa and ha:          return "B"
    elif ha and pcom:        return "C"
    elif oa and pcom:        return "D"
    else:                    return None


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

# ─────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────
SECTIONS = ["anthropometric", "vitals", "menstrual", "labs", "ultrasound", "symptoms"]
SECTION_LABELS = {
    "anthropometric": ("01", "Anthropometric Measurements",      "Height, weight, BMI, body ratios"),
    "vitals":         ("02", "Vitals",                           "Pulse and blood pressure"),
    "menstrual":      ("03", "Menstrual & Reproductive History", "Cycle regularity, pregnancy history"),
    "labs":           ("04", "Laboratory Values",                "Beta-HCG I, AMH, Random Blood Sugar"),
    "ultrasound":     ("05", "Ultrasound Findings",              "Follicle count (left & right ovary)"),
    "symptoms":       ("06", "Clinical Symptoms",                "Self-reported signs and lifestyle"),
}

DEFAULTS = {
    "active_section":  0,
    "section_data":    {},
    "app_step":        "overview",
    "pcos_result":     None,
    "phenotype_result":None,
    "rotterdam_flags": None,
    "inputs":          {},
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
    p2 = pickle.load(open("p2_model.pkl", "rb"))
    return p2

p2_model = load_models()

# ─────────────────────────────────────────────────────────
# 26 TOMIM FEATURES (P2)
# ─────────────────────────────────────────────────────────
P2_FEATURES = [
    "age",
    "weight",
    "height",
    "bmi",
    "blood group",
    "pulse rate (bpm)",
    "cycle (2/4)",
    "marraige status (yrs)",
    "pregnant (1/0)",
    "no. of abortions",
    "i   beta-hcg(miu/ml)",
    "hip (inch)",
    "waist (inch)",
    "waist:hip ratio",
    "amh (ng/ml)",
    "rbs (mg/dl)",
    "skin darkening (1/0)",
    "weight gain (1/0)",
    "hair growth (1/0)",
    "pimples (1/0)",
    "fast food (1/0)",
    "reg.exercise (1/0)",
    "bp _systolic (mmhg)",
    "bp _diastolic (mmhg)",
    "follicle no. (l)",
    "follicle no. (r)",
]

# ─────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────
def predict_phenotype(inp):
    X = np.array([[inp.get(c, np.nan) for c in P2_FEATURES]])
    probs_arr = p2_model.predict_proba(X)[0]
    probs = {c: round(float(p), 3) for c, p in zip(["A", "B", "C", "D"], probs_arr)}
    return max(probs, key=probs.get), probs


def compute_shap_values(inp):
    try:
        import shap

        model    = p2_model.named_steps["clf"]
        selector = p2_model.named_steps["tomim"]

        selected_names = [P2_FEATURES[i] for i in selector.selected_idx_]

        X = np.array([[inp.get(c, np.nan) for c in P2_FEATURES]])

        for step_name, step_obj in p2_model.steps:
            if step_name == "clf":
                break
            if not hasattr(step_obj, "transform"):
                continue
            X = step_obj.transform(X)
            if step_name == "tomim":
                break

        explainer   = shap.TreeExplainer(model)
        shap_values = explainer(X)

        return shap_values, selected_names, None

    except Exception:
        return None, None, traceback.format_exc()


def get_shap_driver_features(inp, ph, top_n=6):
    """
    Extract top-N features driving the predicted phenotype from SHAP values.
    Returns list of (label, patient_value, shap_value, clinical_ref) tuples.
    Falls back to hardcoded list if SHAP fails.
    """
    LABEL_MAP = {
        "i   beta-hcg(miu/ml)": "β-HCG I",
        "amh (ng/ml)":           "AMH",
        "rbs (mg/dl)":           "RBS",
        "follicle no. (l)":      "Follicles L",
        "follicle no. (r)":      "Follicles R",
        "waist:hip ratio":       "Waist:Hip",
        "bmi":                   "BMI",
        "cycle (2/4)":           "Cycle",
        "weight gain (1/0)":     "Weight Gain",
        "hair growth (1/0)":     "Hair Growth",
        "pimples (1/0)":         "Pimples",
        "skin darkening (1/0)":  "Skin Darkening",
        "fast food (1/0)":       "Fast Food",
        "reg.exercise (1/0)":    "Exercise",
        "bp _systolic (mmhg)":   "BP Systolic",
        "bp _diastolic (mmhg)":  "BP Diastolic",
        "pulse rate (bpm)":      "Pulse Rate",
        "waist (inch)":          "Waist",
        "hip (inch)":            "Hip",
        "blood group":           "Blood Group",
        "marraige status (yrs)": "Marriage (yrs)",
        "pregnant (1/0)":        "Pregnant",
        "no. of abortions":      "Abortions",
        "age":                   "Age",
        "weight":                "Weight",
        "height":                "Height",
    }
    # Clinical reference thresholds for % display
    REF_MAP = {
        "amh (ng/ml)":           3.5,
        "rbs (mg/dl)":           140.0,
        "follicle no. (l)":      12.0,
        "follicle no. (r)":      12.0,
        "waist:hip ratio":       0.85,
        "bmi":                   24.9,
        "i   beta-hcg(miu/ml)":  5.0,
        "bp _systolic (mmhg)":   120.0,
        "bp _diastolic (mmhg)":  80.0,
        "pulse rate (bpm)":      100.0,
        "waist (inch)":          35.0,
        "hip (inch)":            45.0,
        "age":                   40.0,
        "weight":                80.0,
        "height":                170.0,
    }

    shap_values, selected_names, err = compute_shap_values(inp)

    if shap_values is not None:
        class_order = ["A", "B", "C", "D"]
        class_idx   = class_order.index(ph)
        sv_class    = shap_values[:, :, class_idx]
        raw_shap    = sv_class[0].values.tolist()

        paired = sorted(
            zip(raw_shap, selected_names),
            key=lambda x: abs(x[0]),
            reverse=True
        )[:top_n]

        result = []
        for shap_val, feat_key in paired:
            label   = LABEL_MAP.get(feat_key, feat_key)
            val     = inp.get(feat_key, 0) or 0
            ref     = REF_MAP.get(feat_key, None)
            result.append((label, val, shap_val, ref))
        return result, True
    else:
        # Fallback: hardcoded clinical drivers
        fallback = [
            ("AMH",         inp.get("amh (ng/ml)", 0) or 0,         None, 3.5),
            ("β-HCG I",     inp.get("i   beta-hcg(miu/ml)", 0) or 0, None, 5.0),
            ("Follicles L", inp.get("follicle no. (l)", 0) or 0,     None, 12.0),
            ("Follicles R", inp.get("follicle no. (r)", 0) or 0,     None, 12.0),
            ("Waist:Hip",   inp.get("waist:hip ratio", 0) or 0,      None, 0.85),
            ("RBS",         inp.get("rbs (mg/dl)", 0) or 0,          None, 140.0),
        ]
        return [(l, v, None, r) for l, v, _, r in fallback], False


def reset():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if not isinstance(v, dict) else {}

# ─────────────────────────────────────────────────────────
# KPI NUMBER FORMATTER — FIX Q4
# ─────────────────────────────────────────────────────────
def fmt_kpi(val):
    """Smart formatter: never use scientific notation, show enough precision."""
    try:
        v = float(val)
        if v == int(v) and abs(v) < 10000:
            return str(int(v))
        elif abs(v) >= 100:
            return f"{v:.1f}"
        elif abs(v) >= 10:
            return f"{v:.2f}"
        else:
            return f"{v:.3f}".rstrip('0').rstrip('.')
    except:
        return str(val)

# ─────────────────────────────────────────────────────────
# SIDEBAR HELPERS
# ─────────────────────────────────────────────────────────
CHECK = '<span class="nav-check">&#10003;</span>'

def _nav_row(num, label, state):
    css         = f"nav-{state}"
    check       = CHECK if state == "done" else ""
    badge_extra = ' style="background:rgba(124,82,204,0.2);color:#b08af5;"' if state == "done" else ""
    return (
        f'<div class="nav-section {css}">'
        f'<span class="step-badge"{badge_extra}>{num}</span>'
        f'<span style="flex:1;">{label}</span>'
        f'{check}'
        f'</div>'
    )

def _pipeline_stages(rule_state, p2_state, dash_state="locked"):
    def stage(num, title, desc, state):
        css   = f"pipeline-stage stage-{state}"
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
    html  = '<div class="pipeline-label">Diagnostic Pipeline</div>'
    html += stage("01", "Rotterdam Rules",          "Criteria-based PCOS detection", rule_state)
    html += stage("02", "Phenotype Classification", "Types A / B / C / D",           p2_state)
    html += stage("03", "Clinical Dashboard",       "Charts, importance & summary",  dash_state)
    st.markdown(html, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">PCOS Diagnostic Tool</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">Rule-Based + ML Clinical Assistant</div>', unsafe_allow_html=True)

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
            if i == active_sec:  state = "active"
            elif i < active_sec: state = "done"
            else:                state = "locked"
        elif app_step == "overview":
            state = "locked"
        else:
            state = "done"
        rows_html += _nav_row(num, label, state)
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if app_step == "overview":
        _pipeline_stages(rule_state="locked", p2_state="locked", dash_state="locked")
    elif app_step == "form":
        _pipeline_stages(rule_state="locked", p2_state="locked", dash_state="locked")
    elif app_step == "result":
        pcos_pos = st.session_state.pcos_result
        if pcos_pos:
            _pipeline_stages(rule_state="done", p2_state="done", dash_state="locked")
        else:
            _pipeline_stages(rule_state="active", p2_state="locked", dash_state="locked")
    elif app_step == "dashboard":
        _pipeline_stages(rule_state="done", p2_state="done", dash_state="active")

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if app_step not in ("overview", "form") or active_sec > 0:
        if st.button("Start Over", use_container_width=True):
            reset(); st.rerun()

# ─────────────────────────────────────────────────────────
# MAIN — HEADER
# ─────────────────────────────────────────────────────────
app_step   = st.session_state.app_step
active_sec = st.session_state.active_section

if app_step == "overview":
    st.markdown('<div class="main-header"><h1>PCOS Diagnostic Tool</h1><p>An overview of PCOS and the classification system used in this tool. Read before proceeding to the clinical assessment.</p></div>', unsafe_allow_html=True)
elif app_step == "form":
    _, label, desc = SECTION_LABELS[SECTIONS[active_sec]]
    st.markdown(f"""
    <div class="main-header">
        <h1>Step {active_sec + 1} of {len(SECTIONS)} — {label}</h1>
        <p>{desc} &nbsp;&middot;&nbsp; Fill in the fields below and click <strong>Next</strong> to continue.</p>
    </div>
    """, unsafe_allow_html=True)
elif app_step == "result":
    pcos_pos = st.session_state.pcos_result
    if pcos_pos:
        st.markdown('<div class="main-header"><h1>PCOS Detected — Phenotype Classification</h1><p>Rotterdam criteria confirmed PCOS. The ML model has classified the phenotype below.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header"><h1>PCOS Detection Result</h1><p>Based on Rotterdam criteria applied to all entered clinical data.</p></div>', unsafe_allow_html=True)
elif app_step == "dashboard":
    st.markdown('<div class="main-header"><h1>Clinical Dashboard</h1><p>Full diagnostic summary, biomarker analysis, and model explainability.</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────────────────────
if app_step == "overview":

    rotterdam_criteria = [
        ("1", "Oligo / Anovulation",  "Irregular or absent menstrual cycles due to infrequent or absent ovulation"),
        ("2", "Hyperandrogenism",      "Elevated androgens — assessed clinically (acne, hirsutism, skin darkening) or biochemically"),
        ("3", "Polycystic Ovaries",    "&ge;12 follicles per ovary or ovarian volume &gt;10 mL on ultrasound"),
    ]
    rotterdam_html = ""
    for n, title, desc in rotterdam_criteria:
        rotterdam_html += (
            '<div class="ov-rotterdam-item">'
            '<div class="ov-rotterdam-num">Criterion ' + n + '</div>'
            '<div class="ov-rotterdam-title">' + title + '</div>'
            '<div class="ov-rotterdam-desc">' + desc + '</div>'
            '</div>'
        )

    hero_html = (
        '<div class="ov-hero">'
        '<div class="ov-hero-bar"></div>'
        '<div class="ov-hero-inner">'
        '<div class="ov-hero-icon-col"><div class="ov-hero-icon">&#x1F52C;</div></div>'
        '<div style="flex:1;min-width:0;">'
        '<div class="ov-eyebrow">Background</div>'
        '<div class="ov-hero-title">What is PCOS?</div>'
        '<p class="ov-hero-lead">'
        'Polycystic Ovary Syndrome (PCOS) is a hormonal disorder affecting women of reproductive age. '
        'It is characterised by a combination of reproductive, metabolic, and endocrine features. '
        'Diagnosis follows the <strong>Rotterdam 2003 criteria</strong>, requiring at least two of three '
        'defined features to be present, with other causes of hyperandrogenism excluded first.'
        '</p>'
        '<div class="ov-stat-row">'
        '<div class="ov-stat-chip"><span class="ov-stat-val">6&ndash;12%</span><span class="ov-stat-label">estimated prevalence (WHO)</span></div>'
        '<div class="ov-stat-chip"><span class="ov-stat-val">Rotterdam 2003</span><span class="ov-stat-label">diagnostic standard</span></div>'
        '<div class="ov-stat-chip"><span class="ov-stat-val">4 phenotypes</span><span class="ov-stat-label">clinical subtypes</span></div>'
        '</div>'
        f'<div style="margin:1.2rem 0 1rem;border-radius:12px;overflow:hidden;border:1px solid #ded5f0;'
        f'box-shadow:0 4px 24px rgba(108,63,197,0.10);position:relative;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#7c52cc,#b08af5,#7c52cc);"></div>'
        f'<img src="data:image/jpeg;base64,{OVARY_IMG}" style="width:100%;display:block;" />'
        f'<div style="padding:0.55rem 0.9rem;background:#faf8fe;border-top:1px solid #ede6f5;">'
        f'<span style="font-size:0.65rem;color:#9580b8;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;">Fig. 1 — Normal vs. Polycystic Ovarian Morphology</span>'
        f'</div>'
        f'</div>'
        '<div class="ov-rotterdam-grid">' + rotterdam_html + '</div>'
        '<div class="ov-rotterdam-rule">'
        '<strong>Rotterdam criteria</strong> &mdash; at least <strong>2 of the 3</strong> features above '
        'must be present for diagnosis. Other causes of hyperandrogenism must be excluded prior to classification.'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(hero_html, unsafe_allow_html=True)

    st.markdown(
        '<div class="ov-box">'
        '<div class="ov-eyebrow">Clinical relevance</div>'
        '<div class="ov-box-title">Why Phenotypes?</div>'
        '<p class="ov-box-body">'
        'Because PCOS can present with different combinations of the three Rotterdam features, '
        'the criteria define <strong>four distinct phenotypes</strong>. Each represents a specific feature '
        'combination and carries different implications for metabolic risk, fertility outcomes, '
        'and treatment approach.'
        '</p>'
        '<p class="ov-box-body" style="margin-top:0.7rem;">'
        'This tool uses a <strong>two-stage approach</strong>: '
        '<span class="ov-inline-pill">Stage 1 — Rotterdam Rules</span> applies the official clinical '
        'criteria to determine whether PCOS is present and which phenotype best fits the data. '
        '<span class="ov-inline-pill">Stage 2 — ML Model</span> then confirms and refines the '
        'phenotype classification using a trained Random Forest pipeline, with confidence scores and a '
        'full clinical dashboard.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="margin-bottom:0.9rem;">'
        '<div class="ov-eyebrow">Rotterdam classification</div>'
        '<div class="ov-section-title">The four PCOS phenotypes</div>'
        '<div class="ov-severity-row">'
        '<span class="ov-sev-label">Higher metabolic risk</span>'
        '<div class="ov-sev-track">'
        '<div class="ov-sev-fill"></div>'
        '<div class="ov-sev-badges">'
        '<span class="ov-sev-badge" style="background:#7c52cc;">A</span>'
        '<span class="ov-sev-badge" style="background:#b83232;">B</span>'
        '<span class="ov-sev-badge" style="background:#1a6e3c;">C</span>'
        '<span class="ov-sev-badge" style="background:#9a7010;">D</span>'
        '</div>'
        '</div>'
        '<span class="ov-sev-label">Lower metabolic risk</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div style="margin:0.4rem 0 1.2rem;border-radius:12px;overflow:hidden;border:1px solid #ded5f0;'
        f'box-shadow:0 4px 24px rgba(108,63,197,0.10);position:relative;">'
        f'<div style="position:absolute;top:0;left:0;right:0;height:3px;'
        f'background:linear-gradient(90deg,#7c52cc,#b08af5,#7c52cc);"></div>'
        f'<img src="data:image/jpeg;base64,{PHENOTYPES_IMG}" style="width:100%;display:block;" />'
        f'<div style="padding:0.55rem 0.9rem;background:#faf8fe;border-top:1px solid #ede6f5;">'
        f'<span style="font-size:0.65rem;color:#9580b8;font-weight:600;letter-spacing:0.1em;'
        f'text-transform:uppercase;">Fig. 2 — Rotterdam Phenotype Classification (A–D)</span>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    ph_data = [
        ("A", "#7c52cc", "rgba(124,82,204,0.08)", "#5a38b0",
         "Full Classic PCOS",
         "Anovulation + Hyperandrogenism + Polycystic Ovaries",
         "All three Rotterdam criteria are met. Associated with the highest prevalence of metabolic comorbidities including insulin resistance.",
         ["Irregular or absent menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Polycystic ovarian morphology on ultrasound", "LH and AMH frequently elevated"],
         "Highest associated metabolic risk &middot; All 3 criteria met"),
        ("B", "#b83232", "rgba(184,50,50,0.08)", "#8a2020",
         "Classic without PCOM",
         "Anovulation + Hyperandrogenism &middot; Normal ovarian morphology",
         "Cycle irregularity and androgen excess are present, but ovarian morphology on ultrasound is within normal limits.",
         ["Irregular or absent menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Normal ovarian morphology on ultrasound", "Biochemical workup required for diagnosis"],
         "Elevated metabolic risk &middot; No polycystic morphology"),
        ("C", "#1a6e3c", "rgba(26,110,60,0.08)", "#0e4c28",
         "Ovulatory PCOS",
         "Hyperandrogenism + Polycystic Ovaries &middot; Regular cycles",
         "Ovulation is preserved despite androgen excess and polycystic ovarian morphology.",
         ["Regular menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Polycystic ovarian morphology on ultrasound", "Fertility often preserved"],
         "Lower metabolic risk &middot; Ovulation preserved"),
        ("D", "#9a7010", "rgba(154,112,16,0.08)", "#6b4e08",
         "Non-Androgenic PCOS",
         "Anovulation + Polycystic Ovaries &middot; No hyperandrogenism",
         "Cycle irregularity and polycystic ovarian morphology are present, but androgen levels are within normal limits.",
         ["Irregular or absent menstrual cycles", "Normal androgen levels (clinical and biochemical)",
          "Polycystic ovarian morphology on ultrasound", "Androgen-related symptoms absent"],
         "Lowest associated metabolic risk &middot; Classification debated"),
    ]

    col1, col2 = st.columns(2, gap="medium")
    for idx, (ph, color, bg, txt_color, sublabel, combo, description, features, note) in enumerate(ph_data):
        col = col1 if idx % 2 == 0 else col2
        with col:
            feats_html = ""
            for f in features:
                feats_html += (
                    '<div class="ov-ph-feat">'
                    '<span class="ov-ph-dot" style="background:' + color + ';"></span>'
                    + f + '</div>'
                )
            card_html = (
                '<div class="ov-ph-card">'
                '<div class="ov-ph-bar" style="background:' + color + ';"></div>'
                '<div class="ov-ph-head">'
                '<div class="ov-ph-badge" style="background:' + color + ';">' + ph + '</div>'
                '<div>'
                '<div class="ov-ph-label">Phenotype ' + ph + ' &mdash; ' + sublabel + '</div>'
                '<div class="ov-ph-combo">' + combo + '</div>'
                '</div></div>'
                '<p class="ov-ph-desc">' + description + '</p>'
                '<div class="ov-ph-features">' + feats_html + '</div>'
                '<div class="ov-ph-note" style="background:' + bg + ';color:' + txt_color + ';">' + note + '</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    steps_data = [
        ("01", "Enter clinical data",
         "Six sections: anthropometrics, vitals, menstrual history, laboratory values (Beta-HCG I, AMH, RBS), ultrasound follicle counts, and reported symptoms.",
         "~3 min to complete", "#f5f0fe", "#7a5caa", "#ddd0f5"),
        ("02", "Rotterdam rule check",
         "The three Rotterdam criteria (OA, HA, PCOM) are evaluated. If at least two criteria are met, PCOS is confirmed and the phenotype (A–D) is determined by the specific combination present.",
         "Rule-based · OA / HA / PCOM flags", "#f5f0fe", "#5a38b0", "rgba(124,82,204,0.25)"),
        ("03", "ML phenotype refinement",
         "A Random Forest pipeline trained on 26 TOMIM-selected features confirms the phenotype classification with per-class probability scores and a full clinical dashboard.",
         "Multi-class · A / B / C / D", "#f5f0fe", "#5a38b0", "rgba(124,82,204,0.25)"),
    ]
    how_html = ""
    for num, title, body, tag, tag_bg, tag_color, tag_border in steps_data:
        how_html += (
            '<div class="ov-how-step">'
            '<div class="ov-how-num">' + num + '</div>'
            '<div class="ov-how-title">' + title + '</div>'
            '<p class="ov-how-body">' + body + '</p>'
            '<div class="ov-how-tag" style="background:' + tag_bg + ';color:' + tag_color + ';border-color:' + tag_border + ';">' + tag + '</div>'
            '</div>'
        )

    st.markdown(
        '<div class="ov-box" style="margin-top:0.4rem;">'
        '<div class="ov-eyebrow">Assessment pipeline</div>'
        '<div class="ov-box-title">How this tool works</div>'
        '<div class="ov-how-grid">' + how_html + '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ov-disclaimer">'
        '<strong>Clinical disclaimer</strong> &mdash; This tool is intended for research and educational purposes only. '
        'It does not constitute a medical diagnosis. All outputs must be reviewed and confirmed by a licensed '
        'OB-GYN, reproductive endocrinologist, or qualified healthcare professional. ML models were trained on '
        'a specific clinical dataset and may not generalise to all patient populations.'
        '</div>',
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([3, 1])
    with col_r:
        if st.button("Begin Assessment \u2192", use_container_width=True):
            st.session_state.app_step = "form"
            st.rerun()

# ─────────────────────────────────────────────────────────
# FORM
# ─────────────────────────────────────────────────────────
elif app_step == "form":
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
            age    = c1.number_input("Age (years)",  min_value=15,    max_value=55,    value=sd.get("age",    None), placeholder="Type here")
            weight = c2.number_input("Weight (kg)",  min_value=30.0,  max_value=150.0, value=sd.get("weight", None), step=0.5, placeholder="Type here")
            height = c3.number_input("Height (cm)",  min_value=130.0, max_value=200.0, value=sd.get("height", None), step=0.5, placeholder="Type here")

            bmi = sd.get("_calc_bmi")
            whr = sd.get("_calc_whr")

            c1, c2, c3, c4, c5 = st.columns(5)
            hip   = c1.number_input("Hip (inch)",   min_value=20.0, max_value=60.0, value=sd.get("hip",   None), step=0.5, placeholder="Type here")
            waist = c2.number_input("Waist (inch)", min_value=20.0, max_value=60.0, value=sd.get("waist", None), step=0.5, placeholder="Type here")

            bg_opts     = ["A+","A-","B+","B-","O+","O-","AB+","AB-"]
            blood_group = c3.selectbox("Blood Type", bg_opts, index=bg_opts.index(sd.get("blood_group","A+")))

            if weight and height and height > 0:
                bmi = round(weight / ((height / 100) ** 2), 2)
            if waist and hip and hip > 0:
                whr = round(waist / hip, 3)

            bmi_display = f"{bmi:.1f} kg/m²" if bmi else "—"
            whr_display = f"{whr:.3f}" if whr else "—"

            c4.markdown(f'<div class="auto-pill">BMI<span>{bmi_display}</span></div>', unsafe_allow_html=True)
            c5.markdown(f'<div class="auto-pill">Waist:Hip Ratio<span>{whr_display}</span></div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            calc_col, _, next_col = st.columns([1, 0.1, 3.9])
            with calc_col:
                calc = st.form_submit_button("Calculate", use_container_width=True)
            with next_col:
                nxt = st.form_submit_button("Next — Vitals", use_container_width=True)

        if calc:
            _bmi = round(weight / ((height/100)**2), 2) if (weight and height) else None
            _whr = round(waist/hip, 3) if (waist and hip and hip > 0) else None
            sd.update({
                "_calc_bmi": _bmi, "_calc_whr": _whr,
                "age": age, "weight": weight, "height": height,
                "hip": hip, "waist": waist, "blood_group": blood_group,
            })
            st.rerun()

        if nxt:
            missing = [f for f, v in [("Age",age),("Weight",weight),("Height",height),("Hip",hip),("Waist",waist)] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                _bmi = round(weight / ((height/100)**2), 2)
                _whr = round(waist/hip, 3)
                bg_map = {"A+":11,"A-":12,"B+":13,"B-":14,"O+":15,"O-":16,"AB+":17,"AB-":18}
                sd.update({
                    "age":age,"weight":weight,"height":height,"bmi":_bmi,
                    "hip":hip,"waist":waist,"whr":_whr,"blood_group":blood_group,
                    "bg_code":bg_map[blood_group],
                    "_calc_bmi":_bmi,"_calc_whr":_whr,
                })
                st.session_state.active_section = 1; st.rerun()

    elif active_sec == 1:
        with st.form("sec_1"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">02</div>
                <div><div class="section-title">Vitals</div>
                <div class="section-desc">Pulse and blood pressure</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            pulse  = c1.number_input("Pulse Rate (bpm)",    min_value=40, max_value=130, value=sd.get("pulse",  None), placeholder="Type here")
            bp_sys = c2.number_input("BP Systolic (mmHg)",  min_value=70, max_value=200, value=sd.get("bp_sys", None), placeholder="Type here")
            bp_dia = c3.number_input("BP Diastolic (mmHg)", min_value=40, max_value=130, value=sd.get("bp_dia", None), placeholder="Type here")

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_1", 0, "Next — Menstrual History")

        if nxt:
            missing = [f for f, v in [("Pulse Rate",pulse),("BP Systolic",bp_sys),("BP Diastolic",bp_dia)] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
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
            marriage  = c2.number_input("Marriage Duration (years)", min_value=0, max_value=40, value=sd.get("marriage_yr", None), placeholder="Type here")
            pr_opts   = ["No","Yes"]
            pregnant  = c3.selectbox("Currently Pregnant?", pr_opts, index=pr_opts.index(sd.get("pregnant_label","No")))
            abortions = c4.number_input("No. of Abortions", min_value=0, max_value=10, value=sd.get("abortions", None), placeholder="Type here")

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_2", 1, "Next — Laboratory Values")

        if nxt:
            missing = [f for f, v in [("Marriage Duration",marriage),("No. of Abortions",abortions)] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                sd.update({
                    "cycle_ri_label": cycle_ri,
                    "cycle_ri":       1 if cycle_ri=="Irregular" else 0,
                    "cycle_24":       4 if cycle_ri=="Irregular" else 2,
                    "marriage_yr":    marriage,
                    "pregnant_label": pregnant,
                    "pregnant":       1 if pregnant=="Yes" else 0,
                    "abortions":      abortions,
                })
                st.session_state.active_section = 3; st.rerun()
        if back: st.session_state.active_section = 1; st.rerun()

    elif active_sec == 3:
        with st.form("sec_3"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">04</div>
                <div><div class="section-title">Laboratory Values</div>
                <div class="section-desc">Beta-HCG I, AMH, and Random Blood Sugar</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            beta_hcg_i = c1.number_input("Beta-HCG I (mIU/mL)",       min_value=0.0,  max_value=500.0, value=sd.get("beta_hcg_i", None), step=0.1, placeholder="Type here")
            amh        = c2.number_input("AMH (ng/mL)",                min_value=0.0,  max_value=15.0,  value=sd.get("amh",        None), step=0.1, placeholder="Type here")
            rbs        = c3.number_input("Random Blood Sugar (mg/dl)", min_value=50.0, max_value=400.0, value=sd.get("rbs",        None), step=1.0, placeholder="Type here")

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_3", 2, "Next — Ultrasound Findings")

        if nxt:
            missing = [f for f, v in [("Beta-HCG I",beta_hcg_i),("AMH",amh),("Random Blood Sugar",rbs)] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                sd.update({"beta_hcg_i":beta_hcg_i, "amh":amh, "rbs":rbs})
                st.session_state.active_section = 4; st.rerun()
        if back: st.session_state.active_section = 2; st.rerun()

    elif active_sec == 4:
        with st.form("sec_4"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">05</div>
                <div><div class="section-title">Ultrasound Findings</div>
                <div class="section-desc">Antral follicle count for left and right ovary</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            follicle_l = c1.number_input("Follicle No. (Left Ovary)",  min_value=0, max_value=30, value=sd.get("follicle_l", None), placeholder="Type here")
            follicle_r = c2.number_input("Follicle No. (Right Ovary)", min_value=0, max_value=30, value=sd.get("follicle_r", None), placeholder="Type here")

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_4", 3, "Next — Clinical Symptoms")

        if nxt:
            missing = [f for f, v in [("Follicle No. Left",follicle_l),("Follicle No. Right",follicle_r)] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                sd.update({"follicle_l":follicle_l, "follicle_r":follicle_r})
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
            yn          = ["No","Yes"]
            weight_gain = c1.radio("Weight Gain?",        yn, index=yn.index(sd.get("weight_gain_label","No")), horizontal=True)
            hair_growth = c2.radio("Excess Hair Growth?", yn, index=yn.index(sd.get("hair_growth_label","No")), horizontal=True)
            pimples     = c3.radio("Pimples / Acne?",     yn, index=yn.index(sd.get("pimples_label","No")),     horizontal=True)

            c1, c2, c3 = st.columns(3)
            fast_food = c1.radio("Fast Food (regularly)?", yn, index=yn.index(sd.get("fast_food_label","No")), horizontal=True)
            exercise  = c2.radio("Regular Exercise?",      yn, index=yn.index(sd.get("exercise_label","No")),  horizontal=True)
            skin_dark = c3.radio("Skin Darkening?",        yn, index=yn.index(sd.get("skin_dark_label","No")), horizontal=True)

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_5", 4, "Run Diagnostic")

        if nxt:
            sd.update({
                "weight_gain_label": weight_gain, "weight_gain":    1 if weight_gain=="Yes" else 0,
                "hair_growth_label": hair_growth, "hair_growth":    1 if hair_growth=="Yes" else 0,
                "pimples_label":     pimples,     "pimples":        1 if pimples=="Yes"     else 0,
                "skin_dark_label":   skin_dark,   "skin_darkening": 1 if skin_dark=="Yes"   else 0,
                "fast_food_label":   fast_food,   "fast_food":      1 if fast_food=="Yes"   else 0,
                "exercise_label":    exercise,    "exercise":       1 if exercise=="Yes"    else 0,
            })
            s = sd

            inp = {
                "age":                   s["age"],
                "weight":                s["weight"],
                "height":                s["height"],
                "bmi":                   s["bmi"],
                "blood group":           s["bg_code"],
                "pulse rate (bpm)":      s["pulse"],
                "cycle (2/4)":           s["cycle_24"],
                "marraige status (yrs)": s["marriage_yr"],
                "pregnant (1/0)":        s["pregnant"],
                "no. of abortions":      s["abortions"],
                "i   beta-hcg(miu/ml)":  s["beta_hcg_i"],
                "hip (inch)":            s["hip"],
                "waist (inch)":          s["waist"],
                "waist:hip ratio":       s["whr"],
                "amh (ng/ml)":           s["amh"],
                "rbs (mg/dl)":           s["rbs"],
                "weight gain (1/0)":     s["weight_gain"],
                "hair growth (1/0)":     s["hair_growth"],
                "pimples (1/0)":         s["pimples"],
                "fast food (1/0)":       s["fast_food"],
                "reg.exercise (1/0)":    s["exercise"],
                "bp _systolic (mmhg)":   s["bp_sys"],
                "bp _diastolic (mmhg)":  s["bp_dia"],
                "follicle no. (l)":      s["follicle_l"],
                "follicle no. (r)":      s["follicle_r"],
                "cycle_ri":              s["cycle_ri"],
                "skin darkening (1/0)":  s["skin_darkening"],
            }
            st.session_state.inputs = inp

            oa, ha, pcom   = evaluate_rotterdam(inp)
            rule_phenotype = classify_phenotype_rule(oa, ha, pcom)
            pcos_positive  = rule_phenotype is not None
            st.session_state.pcos_result     = pcos_positive
            st.session_state.rotterdam_flags = (oa, ha, pcom)

            if pcos_positive:
                ph_ml, probs = predict_phenotype(inp)
                st.session_state.phenotype_result = (rule_phenotype, probs)
            else:
                st.session_state.phenotype_result = None

            st.session_state.app_step = "result"
            st.rerun()

        if back: st.session_state.active_section = 4; st.rerun()

# ─────────────────────────────────────────────────────────
# RESULT PAGE
# ─────────────────────────────────────────────────────────
elif app_step == "result":
    pcos_pos     = st.session_state.pcos_result
    oa, ha, pcom = st.session_state.rotterdam_flags
    inp          = st.session_state.inputs

    def crit_badge(label, met, detail):
        if met:
            bg, border, text, icon = "#f0ebff","#7c52cc","#4a2c9e","✓"
        else:
            bg, border, text, icon = "#f9f9f9","#d0d0d0","#999999","✗"
        return (
            f'<div style="background:{bg};border:1.5px solid {border};border-radius:10px;'
            f'padding:0.7rem 1rem;flex:1;min-width:160px;">'
            f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{"#7c52cc" if met else "#999"};font-weight:700;margin-bottom:0.25rem;">'
            f'{icon} {label}</div>'
            f'<div style="font-size:0.8rem;color:{text};line-height:1.4;">{detail}</div>'
            f'</div>'
        )

    fl = inp.get("follicle no. (l)", 0) or 0
    fr = inp.get("follicle no. (r)", 0) or 0

    oa_detail  = "Irregular menstrual cycle" if oa else "Regular menstrual cycle"
    ha_detail  = "Hair growth / acne / skin darkening present" if ha else "No hyperandrogenism signs"
    pco_detail = f"Follicles L:{fl} R:{fr} (≥12 threshold)" if pcom else f"Follicles L:{fl} R:{fr} (below threshold)"

    badges_html = (
        '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;margin-bottom:1.2rem;">'
        + crit_badge("OA — Anovulation",          oa,   oa_detail)
        + crit_badge("HA — Hyperandrogenism",     ha,   ha_detail)
        + crit_badge("PCOM — Polycystic Ovaries", pcom, pco_detail)
        + '</div>'
    )
    st.markdown(badges_html, unsafe_allow_html=True)

    if pcos_pos:
        ph, probs = st.session_state.phenotype_result
        info      = PHENOTYPE_INFO[ph]
        ph_colors = {"A":"#7c52cc","B":"#b83232","C":"#1a6e3c","D":"#9a7010"}

        st.markdown(f"""
        <div class="result-positive" style="text-align:center;padding:1.6rem 2rem;">
            <div style="width:60px;height:60px;background:{info['color']};border-radius:50%;
                        display:flex;align-items:center;justify-content:center;margin:0 auto 0.8rem;
                        color:white;font-family:'Libre Baskerville',serif;font-size:1.4rem;font-weight:700;">
                {ph}
            </div>
            <div class="result-title">PCOS Detected — {info['label']}</div>
            <div class="result-subtitle">{info['sublabel']} &nbsp;·&nbsp; {info['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2, gap="large")

        with col1:
            crit_items = []
            if oa:   crit_items.append(("OA",   "Oligo / Anovulation", "Irregular menstrual cycle confirmed"))
            if ha:   crit_items.append(("HA",   "Hyperandrogenism",    "Hair growth, acne, or skin darkening"))
            if pcom: crit_items.append(("PCOM", "Polycystic Ovaries",  f"Follicles L:{fl}  R:{fr} (≥12 threshold)"))

            crit_html  = '<div style="margin-bottom:1.2rem;">'
            crit_html += (
                '<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.13em;'
                'color:#9580b8;font-weight:700;margin-bottom:0.6rem;">Rotterdam Criteria Met</div>'
            )
            for code, title, detail in crit_items:
                crit_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:0.65rem;'
                    f'margin-bottom:0.5rem;padding:0.6rem 0.8rem;'
                    f'background:rgba(108,63,197,0.06);border:1px solid #ddd0f5;border-radius:9px;">'
                    f'<div style="width:28px;height:28px;background:#1a0e36;border-radius:6px;'
                    f'display:flex;align-items:center;justify-content:center;flex-shrink:0;'
                    f'font-size:0.58rem;font-weight:700;color:#b08af5;letter-spacing:0.02em;">{code}</div>'
                    f'<div>'
                    f'<div style="font-size:0.8rem;font-weight:600;color:#1e1040;line-height:1.2;">{title}</div>'
                    f'<div style="font-size:0.7rem;color:#9580b8;margin-top:1px;">{detail}</div>'
                    f'</div></div>'
                )
            crit_html += '</div>'
            st.markdown(crit_html, unsafe_allow_html=True)

            feat_html = (
                '<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.13em;'
                'color:#9580b8;font-weight:700;margin-bottom:0.6rem;">Phenotype Clinical Features</div>'
            )
            for feat in info["features"]:
                feat_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:0.55rem;'
                    f'margin-bottom:0.4rem;font-size:0.8rem;color:#4a3a6e;line-height:1.5;">'
                    f'<span style="width:5px;height:5px;border-radius:50%;background:#7c52cc;'
                    f'flex-shrink:0;margin-top:0.4rem;display:inline-block;"></span>'
                    f'{feat}</div>'
                )
            st.markdown(feat_html, unsafe_allow_html=True)

        with col2:
            ph_colors = {"A":"#6c3fc5","B":"#9f1239","C":"#166534","D":"#92400e"}
            st.markdown(
                '<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.13em;'
                'color:#9580b8;font-weight:700;margin-bottom:0.4rem;">P2 Model — Phenotype Confidence</div>'
                '<p style="font-size:0.73rem;color:#9580b8;margin-bottom:0.9rem;line-height:1.5;">'
                'Rotterdam rule-based classification confirmed. Bars show the Random Forest P2 pipeline\'s '
                'per-class probability scores.</p>',
                unsafe_allow_html=True,
            )
            for pk in ["A","B","C","D"]:
                pv        = probs[pk]
                bw        = int(pv * 100)
                is_sel    = pk == ph
                opacity   = "1" if is_sel else "0.45"
                weight    = "700" if is_sel else "400"
                bar_color = ph_colors[pk]
                st.markdown(f"""
                <div style="margin-bottom:0.85rem;opacity:{opacity};">
                  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:4px;">
                    <div style="display:flex;align-items:center;gap:0.5rem;">
                      <span style="width:9px;height:9px;border-radius:50%;background:{bar_color};
                                   display:inline-block;flex-shrink:0;"></span>
                      <span style="font-size:0.8rem;font-weight:{weight};color:#2e1a58;">
                        Phenotype {pk} &nbsp;
                        <span style="font-weight:400;color:#9580b8;font-size:0.72rem;">
                          {PHENOTYPE_INFO[pk]['sublabel']}
                        </span>
                      </span>
                    </div>
                    <span style="font-size:0.8rem;font-weight:{weight};color:{bar_color};">{bw}%</span>
                  </div>
                  <div style="background:#ede6f5;border-radius:4px;height:6px;overflow:hidden;">
                    <div style="width:{bw}%;height:6px;border-radius:4px;background:{bar_color};
                                transition:width 0.4s ease;"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("View Clinical Dashboard", use_container_width=True):
                st.session_state.app_step = "dashboard"; st.rerun()
        with c2:
            if st.button("Start Over with New Patient", use_container_width=True):
                reset(); st.rerun()

    else:
        criteria_met = sum([oa, ha, pcom])
        st.markdown(f"""
        <div class="result-negative">
            <div class="result-negative-accent">&#10003;</div>
            <div class="result-title" style="color:#154030;">No PCOS Detected</div>
            <div class="result-subtitle">
                Only <strong>{criteria_met} of 3</strong> Rotterdam criteria are present.
                At least 2 of 3 must be met for a PCOS diagnosis.
                Phenotype classification will not proceed.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Start Over with New Patient", use_container_width=True):
            reset(); st.rerun()

    st.markdown("""<div class="disclaimer"><strong>Clinical Disclaimer:</strong>
        This tool is for research and educational purposes only. Results do not constitute a medical diagnosis
        and must be confirmed by a licensed healthcare professional. Rotterdam criteria are applied to
        self-reported and entered clinical data only.</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────
elif app_step == "dashboard":
    inp          = st.session_state.inputs
    sd           = st.session_state.section_data
    ph, probs    = st.session_state.phenotype_result
    info         = PHENOTYPE_INFO[ph]
    oa, ha, pcom = st.session_state.rotterdam_flags

    C_PURPLE  = "#6c3fc5"
    C_NAVY    = "#1a0e36"
    C_BORDER  = "#e0d5f5"
    C_TEXT    = "#2e1a58"
    C_MUTED   = "#9580b8"
    C_HIGH    = "#b91c1c"
    C_LOW     = "#c2410c"
    C_OK      = "#166534"
    FONT_SORA = "Sora, sans-serif"
    FONT_CG   = "Cormorant Garamond, serif"

    # Purple palette for charts — FIX Q1
    C_PUR_DARK   = "#4a1fa8"   # deep anchor
    C_PUR_MID    = "#6c3fc5"   # primary
    C_PUR_BRIGHT = "#9b6fe8"   # mid accent
    C_PUR_LIGHT  = "#c4a8f5"   # light
    C_PUR_GHOST  = "#ede6f5"   # background tint

    # Waterfall colours stay red/blue for direction clarity, but use purple for neutral lines
    C_WF_POS  = "#7c3aed"   # purple-positive (toward phenotype)
    C_WF_NEG  = "#be185d"   # rose-negative (away from phenotype)

    BASE_LAYOUT = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_SORA, color=C_TEXT),
    )

    def axis_style(**kw):
        return dict(
            tickfont=dict(family=FONT_SORA, size=10, color=C_TEXT),
            gridcolor="rgba(180,165,220,0.18)",
            zeroline=False, showgrid=True, **kw,
        )

    def section_label(txt):
        st.markdown(
            f'<p style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.14em;'
            f'color:{C_MUTED};font-weight:700;margin:0.15rem 0 0.6rem;">{txt}</p>',
            unsafe_allow_html=True,
        )

    def val_status(v, lo, hi):
        if v is None: return C_TEXT
        if v > hi:    return C_HIGH
        if v < lo:    return C_LOW
        return C_OK

    # ── Rotterdam summary strip ───────────────────────────
    def crit_mini(label, met, detail):
        if met:
            bg, border, icon = "rgba(108,63,197,0.08)","#7c52cc","✓"
            tc = "#4a2c9e"
        else:
            bg, border, icon = "#f9f9f9","#d0d0d0","✗"
            tc = "#999"
        return (
            f'<div style="background:{bg};border:1.5px solid {border};border-radius:9px;'
            f'padding:0.5rem 0.9rem;flex:1;min-width:130px;text-align:center;">'
            f'<div style="font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{tc};font-weight:700;">{icon} {label}</div>'
            f'<div style="font-size:0.72rem;color:{tc};margin-top:0.1rem;">{detail}</div>'
            f'</div>'
        )

    fl = inp.get("follicle no. (l)", 0) or 0
    fr = inp.get("follicle no. (r)", 0) or 0
    section_label("Rotterdam Criteria Summary")
    st.markdown(
        '<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1rem;">'
        + crit_mini("OA — Anovulation",          oa,   "Irregular cycle" if oa else "Regular cycle")
        + crit_mini("HA — Hyperandrogenism",     ha,   "Signs present" if ha else "No signs")
        + crit_mini("PCOM — Polycystic Ovaries", pcom, f"Follicles L:{fl} R:{fr}")
        + f'<div style="background:rgba(108,63,197,0.08);border:1.5px solid #7c52cc;border-radius:9px;'
          f'padding:0.5rem 0.9rem;flex:1;min-width:130px;text-align:center;">'
          f'<div style="font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;'
          f'color:#4a2c9e;font-weight:700;">Phenotype</div>'
          f'<div style="font-size:0.72rem;color:#4a2c9e;font-weight:700;margin-top:0.1rem;">'
          f'{ph} — {info["sublabel"]}</div>'
          f'</div>'
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── KPI strip — FIX Q4: use fmt_kpi instead of :.2g ──
    section_label("Key Biomarkers")
    KPI_DEF = [
        ("BMI",       sd.get("bmi",        0) or 0, 18.5,  24.9,  "kg/m²"),
        ("AMH",       sd.get("amh",        0) or 0,  1.0,   3.5,  "ng/mL"),
        ("RBS",       sd.get("rbs",        0) or 0, 70.0, 140.0,  "mg/dl"),
        ("Waist:Hip", sd.get("whr",        0) or 0,  0.0,   0.85, "ratio"),
        ("β-HCG I",   sd.get("beta_hcg_i",0) or 0,  0.0,   5.0,  "mIU/mL"),
    ]

    def kpi_html(label, val, lo, hi, unit):
        color   = val_status(val, lo, hi)
        arrow   = " ↑" if val > hi else (" ↓" if val < lo else "")
        display = fmt_kpi(val)   # ← FIXED: no more :.2g truncation
        return (
            f'<div style="background:#ffffff;border:1px solid {C_BORDER};border-radius:10px;'
            f'padding:0.8rem 0.6rem;text-align:center;height:100%;">'
            f'<div style="font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{C_MUTED};font-weight:700;margin-bottom:0.3rem;">{label}</div>'
            f'<div style="font-family:{FONT_CG};font-size:1.45rem;font-weight:600;'
            f'color:{color};line-height:1;word-break:break-all;">{display}'
            f'<span style="font-size:0.68rem;">{arrow}</span></div>'
            f'<div style="font-size:0.58rem;color:{C_MUTED};margin-top:0.15rem;">{unit}</div>'
            f'</div>'
        )

    cells = "".join(kpi_html(l, v, lo, hi, u) for l, v, lo, hi, u in KPI_DEF)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.5rem;margin-bottom:1rem;">{cells}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

    # ── ROW 1: Phenotype donut | Biomarker bar ────────────
    col_ph, col_bio = st.columns([1, 1.65], gap="large")

    with col_ph:
        section_label("Phenotype Classification (P2 Model)")
        ph_colors    = {"A":"#6c3fc5","B":"#9f1239","C":"#166534","D":"#92400e"}
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        donut_labels = list(sorted_probs.keys())
        donut_vals   = list(sorted_probs.values())

        # ── FIX Q3: textinfo="none", annotations only inside hole ──
        fig_donut = go.Figure(go.Pie(
            labels=donut_labels,
            values=donut_vals,
            hole=0.72,
            marker=dict(
                colors=[ph_colors[k] for k in donut_labels],
                line=dict(color="#faf8fe", width=3),
            ),
            textinfo="none",          # ← no slice labels at all — eliminates overlap
            hovertemplate=(
                "<b>Phenotype %{label}</b><br>"
                "%{customdata}<br>"
                "Probability: %{percent}<extra></extra>"
            ),
            customdata=[PHENOTYPE_INFO[k]["sublabel"] for k in donut_labels],
            pull=[0.04 if k == ph else 0 for k in donut_labels],
            direction="clockwise",
            rotation=270,
            showlegend=False,
        ))
        # Centre annotations — phenotype letter, %, sublabel
        fig_donut.add_annotation(
            text=f"<b>{ph}</b>",
            x=0.5, y=0.58,
            font=dict(family=FONT_CG, size=50, color=C_NAVY),
            showarrow=False, xref="paper", yref="paper",
        )
        fig_donut.add_annotation(
            text=f"{int(probs[ph]*100)}%",
            x=0.5, y=0.43,
            font=dict(family=FONT_SORA, size=13, color=info["color"], ),
            showarrow=False, xref="paper", yref="paper",
        )
        fig_donut.add_annotation(
            text=info["sublabel"].replace(" ", "<br>"),
            x=0.5, y=0.28,
            font=dict(family=FONT_SORA, size=8, color=C_MUTED),
            showarrow=False, xref="paper", yref="paper",
        )
        fig_donut.update_layout(
            **BASE_LAYOUT,
            height=300,
            margin=dict(l=10, r=10, t=15, b=10),
        )
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

        # Legend below donut
        legend_html = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;justify-content:center;margin-top:-0.3rem;">'
        for k in ["A","B","C","D"]:
            bold = "font-weight:700;" if k == ph else "opacity:0.6;"
            legend_html += (
                f'<span style="display:inline-flex;align-items:center;gap:0.3rem;'
                f'font-size:0.7rem;color:{C_TEXT};{bold}">'
                f'<span style="width:8px;height:8px;border-radius:50%;background:{ph_colors[k]};display:inline-block;"></span>'
                f'<span>{k} — {int(probs[k]*100)}%</span></span>'
            )
        legend_html += "</div>"
        st.markdown(legend_html, unsafe_allow_html=True)

    with col_bio:
        section_label("Biomarker Status vs. Reference Range")
        bm_data = [
            ("BMI",       sd.get("bmi",        0) or 0, 18.5,  24.9,  "kg/m²"),
            ("AMH",       sd.get("amh",        0) or 0,  1.0,   3.5,  "ng/mL"),
            ("RBS",       sd.get("rbs",        0) or 0, 70.0, 140.0,  "mg/dl"),
            ("Waist:Hip", sd.get("whr",        0) or 0,  0.0,   0.85, "ratio"),
            ("β-HCG I",   sd.get("beta_hcg_i",0) or 0,  0.0,   5.0,  "mIU/mL"),
        ]
        names      = [d[0] for d in bm_data]
        vals       = [d[1] for d in bm_data]
        hi_vals    = [d[3] for d in bm_data]
        lo_vals    = [d[2] for d in bm_data]
        units      = [d[4] for d in bm_data]
        norm_pct   = [min(v / hi * 100, 155) if hi else 0 for v, hi in zip(vals, hi_vals)]
        norm_lo    = [lo / hi * 100 if hi else 0 for lo, hi in zip(lo_vals, hi_vals)]
        status_lbl = ["High ↑" if v > hi else ("Low ↓" if v < lo else "Normal")
                      for v, lo, hi in zip(vals, lo_vals, hi_vals)]

        fig_bio = go.Figure()
        fig_bio.add_trace(go.Bar(y=names, x=[155]*len(names), orientation="h",
            marker_color="rgba(108,63,197,0.07)", hoverinfo="skip", showlegend=False))
        fig_bio.add_trace(go.Bar(y=names, x=norm_pct, orientation="h",
            marker=dict(color=C_PURPLE, opacity=0.75, line=dict(color="rgba(255,255,255,0.3)", width=0.5)),
            text=[f"  {fmt_kpi(v)} {u}  ·  {s}" for v, u, s in zip(vals, units, status_lbl)],
            textposition="outside",
            textfont=dict(family=FONT_SORA, size=9.5, color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>"))
        fig_bio.add_trace(go.Scatter(y=names, x=norm_lo, mode="markers",
            marker=dict(symbol="line-ns", size=14, color=C_MUTED, line=dict(width=1.5, color=C_MUTED)),
            hoverinfo="skip", showlegend=False))
        fig_bio.update_layout(
            **BASE_LAYOUT, barmode="overlay", height=300,
            margin=dict(l=0, r=180, t=8, b=8),
            xaxis=dict(**axis_style(),
                title=dict(text="% of upper reference limit", font=dict(family=FONT_SORA, size=9, color=C_MUTED)),
                ticksuffix="%", range=[0, 200]),
            yaxis=dict(autorange="reversed",
                tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT), showgrid=False),
            shapes=[dict(type="line", x0=100, x1=100, y0=-0.5, y1=len(names)-0.5,
                line=dict(color=C_PURPLE, width=1.5, dash="dot"))],
        )
        st.plotly_chart(fig_bio, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── ROW 2: SHAP Waterfall | Radar ─────────────────────
    section_label("Model Explainability — SHAP Waterfall (This Patient)")

    with st.spinner("Computing SHAP values…"):
        shap_result = compute_shap_values(inp)
        shap_values, selected_names, shap_err = shap_result

    col_imp, col_radar = st.columns([1.4, 1], gap="large")

    with col_imp:
        if shap_values is not None:
            class_order = ["A", "B", "C", "D"]
            class_idx   = class_order.index(ph)
            sv_class    = shap_values[:, :, class_idx]

            label_map = {
                "i   beta-hcg(miu/ml)": "β-HCG I",
                "amh (ng/ml)":          "AMH",
                "rbs (mg/dl)":          "RBS",
                "follicle no. (l)":     "Follicles L",
                "follicle no. (r)":     "Follicles R",
                "waist:hip ratio":      "Waist : Hip",
                "bmi":                  "BMI",
                "cycle (2/4)":          "Cycle",
                "weight gain (1/0)":    "Weight Gain",
                "hair growth (1/0)":    "Hair Growth",
                "pimples (1/0)":        "Pimples",
                "skin darkening (1/0)": "Skin Darkening",
                "fast food (1/0)":      "Fast Food",
                "reg.exercise (1/0)":   "Exercise",
                "bp _systolic (mmhg)":  "BP Systolic",
                "bp _diastolic (mmhg)": "BP Diastolic",
                "pulse rate (bpm)":     "Pulse Rate",
                "waist (inch)":         "Waist",
                "hip (inch)":           "Hip",
                "blood group":          "Blood Group",
                "marraige status (yrs)":"Marriage (yrs)",
                "pregnant (1/0)":       "Pregnant",
                "no. of abortions":     "Abortions",
                "age":                  "Age",
                "weight":               "Weight",
                "height":               "Height",
            }

            raw_shap   = sv_class[0].values.tolist()
            base_val   = float(sv_class[0].base_values)
            feat_names = [label_map.get(n, n) for n in selected_names]
            feat_vals  = [inp.get(c, None) for c in P2_FEATURES
                          if c in [P2_FEATURES[i] for i in p2_model.named_steps["tomim"].selected_idx_]]

            paired   = sorted(zip(raw_shap, feat_names, feat_vals), key=lambda x: abs(x[0]), reverse=True)[:10]
            paired   = list(reversed(paired))

            shap_data = json.dumps([
                {"shap": round(s, 4), "label": l, "val": round(float(v), 3) if v is not None else None}
                for s, l, v in paired
            ])
            ph_color  = ph_colors[ph]
            pred_prob = int(probs[ph] * 100)

            # ── FIX Q1: Waterfall uses purple palette ──────
            WATERFALL_HTML = f"""
<style>
  .wf-wrap {{
    font-family: Sora, sans-serif;
    padding: 4px 0 12px;
    color: #2e1a58;
  }}
  .wf-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    margin-bottom: 18px;
    border-bottom: 1.5px solid #ede6f5;
    padding-bottom: 10px;
  }}
  .wf-title {{
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.13em;
    color: #9580b8;
  }}
  .wf-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {ph_color}18;
    border: 1.5px solid {ph_color}55;
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 10.5px;
    font-weight: 700;
    color: {ph_color};
  }}
  .wf-dot {{ width:7px; height:7px; border-radius:50%; background:{ph_color}; display:inline-block; }}
  .wf-chart {{ position: relative; }}
  .wf-row {{
    display: grid;
    grid-template-columns: 110px 1fr 56px;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
    min-height: 28px;
  }}
  .wf-feat {{
    font-size: 10.5px; color: #4a3a6e; text-align: right;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.3;
  }}
  .wf-feat-val {{ font-size: 9px; color: #9580b8; display: block; margin-top: 1px; }}
  .wf-bar-track {{ position: relative; height: 22px; display: flex; align-items: center; }}
  .wf-bar {{
    position: absolute; height: 16px; border-radius: 3px;
    transition: width 0.6s cubic-bezier(.4,0,.2,1);
    min-width: 3px;
  }}
  .wf-connector {{
    position: absolute; top: 50%; width: 1px; height: 28px;
    transform: translateY(-50%); opacity: 0.3;
  }}
  .wf-shap-val {{ font-size: 10.5px; font-weight: 700; text-align: left; white-space: nowrap; }}
  .wf-shap-val.pos {{ color: #6c3fc5; }}
  .wf-shap-val.neg {{ color: #be185d; }}
  .wf-baseline-row {{
    display: grid; grid-template-columns: 110px 1fr 56px; gap: 8px;
    margin-top: 4px; padding-top: 8px; border-top: 1.5px solid #ede6f5; align-items: center;
  }}
  .wf-baseline-label {{
    font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.1em;
    color: #9580b8; font-weight: 700; text-align: right;
  }}
  .wf-fx-row {{
    display: grid; grid-template-columns: 110px 1fr 56px; gap: 8px;
    margin-top: 6px; padding: 8px 0 4px;
    border-top: 2px solid {ph_color}55; align-items: center;
  }}
  .wf-fx-label {{
    font-size: 9.5px; text-transform: uppercase; letter-spacing: 0.1em;
    color: {ph_color}; font-weight: 700; text-align: right;
  }}
  .wf-fx-val {{ font-size: 11px; font-weight: 700; color: {ph_color}; }}
  .wf-legend {{
    display: flex; gap: 14px; margin-top: 14px; padding-top: 10px; border-top: 1px solid #ede6f5;
  }}
  .wf-legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 9.5px; color: #9580b8; }}
  .wf-legend-swatch {{ width: 22px; height: 7px; border-radius: 2px; }}
</style>

<div class="wf-wrap">
  <div class="wf-header">
    <div class="wf-title">SHAP Waterfall — Phenotype {ph}</div>
    <div class="wf-badge"><span class="wf-dot"></span>{pred_prob}% confidence</div>
  </div>
  <div class="wf-chart" id="wf-chart"></div>
  <div class="wf-legend">
    <div class="wf-legend-item">
      <div class="wf-legend-swatch" style="background:#6c3fc5;opacity:0.8;"></div>
      Pushes toward Phenotype {ph}
    </div>
    <div class="wf-legend-item">
      <div class="wf-legend-swatch" style="background:#be185d;opacity:0.8;"></div>
      Pushes away from Phenotype {ph}
    </div>
  </div>
</div>

<script>
(function() {{
  var data    = {shap_data};
  var baseVal = {round(base_val, 4)};
  // Purple for positive SHAP, rose for negative — both harmonious with the purple UI
  var POS_CLR = "#6c3fc5";
  var NEG_CLR = "#be185d";
  var chart   = document.getElementById('wf-chart');
  var SCALE   = 85;

  var cumulative = baseVal;
  var snapshots  = [];
  data.forEach(function(d) {{ snapshots.push(cumulative); cumulative += d.shap; }});
  var finalVal = cumulative;

  var allVals = snapshots.concat([finalVal, baseVal]);
  var minVal  = Math.min.apply(null, allVals);
  var maxVal  = Math.max.apply(null, allVals);
  var range   = (maxVal - minVal) || 1;

  function toX(v) {{ return ((v - minVal) / range * SCALE) + 2; }}

  // Baseline
  var blRow = document.createElement('div');
  blRow.className = 'wf-baseline-row';
  blRow.innerHTML =
    '<div class="wf-baseline-label">E[f(x)]<br>baseline</div>' +
    '<div style="position:relative;height:22px;">' +
      '<div style="position:absolute;top:50%;left:' + toX(baseVal).toFixed(1) + '%;' +
        'width:1.5px;height:18px;transform:translateY(-50%);background:#9580b8;opacity:0.5;"></div>' +
    '</div>' +
    '<div style="font-size:10.5px;color:#9580b8;font-weight:600;">' + baseVal.toFixed(3) + '</div>';
  chart.appendChild(blRow);

  // Feature rows
  data.forEach(function(d, i) {{
    var startX = toX(snapshots[i]);
    var endX   = toX(snapshots[i] + d.shap);
    var isPos  = d.shap >= 0;
    var color  = isPos ? POS_CLR : NEG_CLR;
    var left   = Math.min(startX, endX);
    var width  = Math.max(Math.abs(endX - startX), 0.8);
    var valStr = d.val !== null ? String(d.val) : "—";

    var row = document.createElement('div');
    row.className = 'wf-row';
    var connLeft = toX(snapshots[i] + d.shap);

    row.innerHTML =
      '<div class="wf-feat">' + d.label +
        '<span class="wf-feat-val">= ' + valStr + '</span>' +
      '</div>' +
      '<div class="wf-bar-track">' +
        '<div class="wf-bar" style="left:' + left.toFixed(1) + '%;width:' + width.toFixed(1) + '%;background:' + color + ';opacity:0.82;"></div>' +
        '<div class="wf-connector" style="left:' + connLeft.toFixed(1) + '%;background:' + color + ';"></div>' +
      '</div>' +
      '<div class="wf-shap-val ' + (isPos ? 'pos' : 'neg') + '">' +
        (isPos ? '+' : '') + d.shap.toFixed(3) +
      '</div>';
    chart.appendChild(row);
  }});

  // f(x) row
  var fxRow = document.createElement('div');
  fxRow.className = 'wf-fx-row';
  fxRow.innerHTML =
    '<div class="wf-fx-label">f(x)<br>output</div>' +
    '<div style="position:relative;height:22px;">' +
      '<div style="position:absolute;top:50%;left:' + toX(finalVal).toFixed(1) + '%;' +
        'width:2px;height:20px;transform:translateY(-50%);background:{ph_color};border-radius:1px;"></div>' +
    '</div>' +
    '<div class="wf-fx-val">' + finalVal.toFixed(3) + '</div>';
  chart.appendChild(fxRow);
}})();
</script>
"""
            components.html(WATERFALL_HTML, height=520, scrolling=False)

            st.markdown(
                f'<p style="font-size:0.71rem;color:{C_MUTED};margin-top:0.3rem;line-height:1.65;">'
                f'Each bar shows how much a feature shifts the model output from baseline E[f(x)] '
                f'toward the final prediction f(x) for Phenotype {ph}. '
                f'<span style="color:#6c3fc5;font-weight:600;">Purple = pushes toward {ph}.</span> '
                f'<span style="color:#be185d;font-weight:600;">Rose = pushes away.</span>'
                f'</p>',
                unsafe_allow_html=True,
            )

        else:
            st.markdown(
                f'<div style="border:1px solid #f5c08a;border-radius:10px;'
                f'padding:1rem 1.2rem;font-size:0.82rem;color:#7a3a10;">'
                f'⚠ SHAP unavailable.<br>'
                f'<pre style="font-size:0.72rem;white-space:pre-wrap;margin-top:0.5rem;">'
                f'{shap_err}</pre></div>',
                unsafe_allow_html=True,
            )

    # ── RADAR — FIX Q1: full purple palette ───────────────
    with col_radar:
        section_label("Biomarker Radar")
        radar_defs = [
            ("AMH",         sd.get("amh",        0) or 0,  1.0,  3.5,   "ng/mL"),
            ("BMI",         sd.get("bmi",        0) or 0,  18.5, 24.9,  "kg/m²"),
            ("Waist:Hip",   sd.get("whr",        0) or 0,  0.0,  0.85,  "ratio"),
            ("RBS",         sd.get("rbs",        0) or 0,  70.0, 140.0, "mg/dl"),
            ("β-HCG I",     sd.get("beta_hcg_i",0) or 0,  0.0,  5.0,   "mIU/mL"),
            ("Follicles L", sd.get("follicle_l", 0) or 0,  0.0,  12.0,  "count"),
            ("Follicles R", sd.get("follicle_r", 0) or 0,  0.0,  12.0,  "count"),
        ]

        r_labels = [d[0] for d in radar_defs]

        def norm_radar(val, lo, hi):
            if hi == lo: return 0
            return min(max((val - lo) / (hi - lo) * 100, 0), 150)

        def radar_status(val, lo, hi):
            if hi > 0 and val > hi: return "High ↑"
            if lo > 0 and val < lo: return "Low ↓"
            return "Normal"

        r_vals  = [norm_radar(d[1], d[2], d[3]) for d in radar_defs]
        r_hover = [
            f"{d[0]}: {fmt_kpi(d[1])} {d[4]} — {radar_status(d[1], d[2], d[3])}"
            for d in radar_defs
        ]

        r_labels_closed = r_labels + [r_labels[0]]
        r_vals_closed   = r_vals   + [r_vals[0]]
        r_hover_closed  = r_hover  + [r_hover[0]]

        any_high = any(norm_radar(d[1], d[2], d[3]) > 100 for d in radar_defs)

        # Always purple — elevated areas use a deeper/more saturated purple
        fill_color = "rgba(124,58,237,0.18)" if any_high else "rgba(108,63,197,0.13)"
        line_color = "#7c3aed" if any_high else C_PURPLE
        marker_clr = "#9b6fe8"

        fig_radar = go.Figure()
        # Dotted reference ring
        fig_radar.add_trace(go.Scatterpolar(
            r=[100] * len(r_labels_closed), theta=r_labels_closed,
            fill=None, mode="lines",
            line=dict(color=C_PUR_LIGHT, width=1.5, dash="dot"),
            name="Upper normal limit", hoverinfo="skip", showlegend=True,
        ))
        # Patient data
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals_closed, theta=r_labels_closed,
            fill="toself", fillcolor=fill_color,
            mode="lines+markers",
            line=dict(color=line_color, width=2.5),
            marker=dict(size=6, color=marker_clr,
                        line=dict(color="#ffffff", width=1.5)),
            name="Patient values",
            text=r_hover_closed,
            hovertemplate="%{text}<extra></extra>",
            showlegend=True,
        ))
        fig_radar.update_layout(
            **BASE_LAYOUT, height=440,
            margin=dict(l=40, r=40, t=30, b=60),
            polar=dict(
                bgcolor="rgba(250,248,254,0.6)",
                radialaxis=dict(
                    visible=True, range=[0, 150],
                    tickvals=[0, 50, 100, 150],
                    ticktext=["0%", "50%", "100%", "150%"],
                    tickfont=dict(family=FONT_SORA, size=8, color=C_MUTED),
                    gridcolor="rgba(180,165,220,0.30)",
                    linecolor="rgba(180,165,220,0.30)",
                ),
                angularaxis=dict(
                    tickfont=dict(family=FONT_SORA, size=10.5, color=C_TEXT),
                    gridcolor="rgba(180,165,220,0.30)",
                    linecolor="rgba(180,165,220,0.25)",
                ),
            ),
            legend=dict(
                font=dict(family=FONT_SORA, size=9, color=C_MUTED),
                orientation="h", yanchor="bottom", y=-0.14,
                xanchor="center", x=0.5,
            ),
        )
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})

        elevated_note = (
            f"<span style='color:{C_PUR_DARK};font-weight:600;'>⚠ One or more values are elevated.</span>"
            if any_high else "All values within normal range."
        )
        st.markdown(
            f'<p style="font-size:0.72rem;color:{C_MUTED};margin-top:-0.5rem;text-align:center;">'
            f'100% = upper normal limit. Values beyond the dotted ring are elevated. {elevated_note}</p>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)

    # ── ROW 3: Follicle bar | Driver bar ──────────────────
    # ── FIX Q2: Driver features now come from SHAP p2_model ──
    section_label("Ultrasound & Clinical Thresholds")
    col_foll, col_sym = st.columns(2, gap="large")

    with col_foll:
        section_label("Ultrasound — Follicle Count")
        fl = sd.get("follicle_l", 0) or 0
        fr = sd.get("follicle_r", 0) or 0
        fig_foll = go.Figure()
        fig_foll.add_trace(go.Bar(
            x=["Left Ovary","Right Ovary"], y=[fl, fr],
            marker=dict(
                color=[C_PUR_MID, C_PUR_BRIGHT],
                line=dict(color="rgba(255,255,255,0.3)", width=0.5),
                cornerradius=4,
            ),
            text=[f"<b>{fl}</b>", f"<b>{fr}</b>"], textposition="outside",
            textfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
            hovertemplate="<b>%{x}</b><br>Follicles: %{y}<extra></extra>", width=0.35))
        fig_foll.add_hline(y=12, line=dict(color=C_HIGH, width=1.5, dash="dash"),
            annotation_text="Polycystic threshold (≥12)", annotation_position="top right",
            annotation_font=dict(size=8.5, color=C_HIGH))
        fig_foll.update_layout(**BASE_LAYOUT, height=280,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT), showgrid=False),
            yaxis=dict(**axis_style(),
                title=dict(text="Follicle count", font=dict(family=FONT_SORA, size=9, color=C_MUTED)),
                range=[0, max(max(fl, fr) * 1.45, 15)]),
            showlegend=False)
        st.plotly_chart(fig_foll, use_container_width=True, config={"displayModeBar": False})

    with col_sym:
        # ── SHAP-driven phenotype driver features ─────────
        section_label("Phenotype Driver Features (SHAP-ranked)")

        with st.spinner(""):
            driver_features, is_shap = get_shap_driver_features(inp, ph, top_n=6)

        if is_shap:
            source_note = "Ranked by absolute SHAP contribution to Phenotype " + ph + " from P2 model."
        else:
            source_note = "SHAP unavailable — showing clinical threshold comparison."

        drv_labels = [d[0] for d in driver_features]
        drv_raw    = [d[1] for d in driver_features]
        drv_shap   = [d[2] for d in driver_features]  # None if fallback
        drv_ref    = [d[3] for d in driver_features]

        # Build display: if SHAP available show SHAP magnitude bars, else % of threshold
        if is_shap and all(s is not None for s in drv_shap):
            max_abs = max(abs(s) for s in drv_shap) or 1
            drv_pct = [abs(s) / max_abs * 100 for s in drv_shap]
            drv_colors = [
                C_PUR_MID if s >= 0 else C_PUR_BRIGHT
                for s in drv_shap
            ]
            x_title = "SHAP importance (normalised)"
            hover_sfx = [f"SHAP: {'+' if s>=0 else ''}{s:.4f}" for s in drv_shap]
        else:
            drv_pct    = [min(v / r * 100, 160) if r else 0 for v, r in zip(drv_raw, drv_ref)]
            drv_colors = [C_PUR_MID if p >= 100 else f"rgba(108,63,197,{max(0.25, p/100*0.7):.2f})"
                          for p in drv_pct]
            x_title    = "% of clinical threshold"
            hover_sfx  = [f"{pct:.1f}% of threshold" for pct in drv_pct]

        fig_drv = go.Figure()

        if is_shap:
            # No threshold line for SHAP view
            pass
        else:
            fig_drv.add_vline(x=100, line=dict(color=C_MUTED, width=1.2, dash="dot"),
                annotation_text="threshold", annotation_position="top right",
                annotation_font=dict(size=8, color=C_MUTED))

        fig_drv.add_trace(go.Bar(
            y=drv_labels[::-1], x=drv_pct[::-1], orientation="h",
            marker=dict(
                color=drv_colors[::-1],
                line=dict(color="rgba(255,255,255,0.2)", width=0.5),
                cornerradius=3,
            ),
            text=[f"  {fmt_kpi(v)}" for v in drv_raw[::-1]],
            textposition="outside",
            textfont=dict(family=FONT_SORA, size=9.5, color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
        ))
        fig_drv.update_layout(**BASE_LAYOUT, height=280,
            margin=dict(l=0, r=70, t=20, b=10),
            xaxis=dict(**axis_style(),
                ticksuffix="" if is_shap else "%",
                range=[0, max(drv_pct) * 1.35 if drv_pct else 120],
                title=dict(text=x_title, font=dict(family=FONT_SORA, size=9, color=C_MUTED))),
            yaxis=dict(tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
                showgrid=False, autorange=True),
            showlegend=False)
        st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})

        if is_shap:
            legend_drv = (
                f'<div style="display:flex;gap:1rem;margin-top:-0.2rem;">'
                f'<span style="font-size:0.7rem;color:{C_MUTED};display:flex;align-items:center;gap:0.3rem;">'
                f'<span style="width:10px;height:10px;border-radius:3px;background:{C_PUR_MID};display:inline-block;"></span>'
                f'Pushes toward {ph}</span>'
                f'<span style="font-size:0.7rem;color:{C_MUTED};display:flex;align-items:center;gap:0.3rem;">'
                f'<span style="width:10px;height:10px;border-radius:3px;background:{C_PUR_BRIGHT};display:inline-block;"></span>'
                f'Pushes away from {ph}</span>'
                f'</div>'
            )
            st.markdown(legend_drv, unsafe_allow_html=True)

        st.markdown(
            f'<p style="font-size:0.72rem;color:{C_MUTED};margin-top:0.2rem;">{source_note}</p>',
            unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Full data record ──────────────────────────────────
    with st.expander("📋 Full Patient Data Record"):
        NICE = {
            "age":"Age (yrs)", "weight":"Weight (kg)", "height":"Height (cm)",
            "bmi":"BMI", "hip":"Hip (in)", "waist":"Waist (in)",
            "whr":"Waist:Hip", "blood_group":"Blood Type",
            "pulse":"Pulse (bpm)", "bp_sys":"BP Systolic", "bp_dia":"BP Diastolic",
            "cycle_ri_label":"Cycle", "marriage_yr":"Marriage (yrs)",
            "pregnant_label":"Pregnant", "abortions":"Abortions",
            "beta_hcg_i":"β-HCG I", "amh":"AMH", "rbs":"RBS",
            "follicle_l":"Follicles L", "follicle_r":"Follicles R",
            "weight_gain_label":"Weight Gain", "hair_growth_label":"Hair Growth",
            "skin_dark_label":"Skin Dark.", "pimples_label":"Pimples",
            "fast_food_label":"Fast Food", "exercise_label":"Exercises",
        }
        SECS = {
            "Anthropometric": ["age","weight","height","bmi","hip","waist","whr","blood_group"],
            "Vitals":         ["pulse","bp_sys","bp_dia"],
            "Menstrual":      ["cycle_ri_label","marriage_yr","pregnant_label","abortions"],
            "Laboratory":     ["beta_hcg_i","amh","rbs"],
            "Ultrasound":     ["follicle_l","follicle_r"],
            "Symptoms":       ["weight_gain_label","hair_growth_label","skin_dark_label","pimples_label","fast_food_label","exercise_label"],
        }
        RANGES = {
            "bmi":(18.5,24.9),"whr":(0,0.85),
            "amh":(1,3.5),"rbs":(70,140),
            "pulse":(60,100),"bp_sys":(90,120),"bp_dia":(60,80),
        }
        for sec_title, keys in SECS.items():
            items = {k: sd[k] for k in keys if k in sd}
            if not items: continue
            st.markdown(
                f'<div style="font-size:0.6rem;text-transform:uppercase;letter-spacing:0.12em;'
                f'font-weight:700;color:{C_MUTED};margin:1rem 0 0.5rem;padding-bottom:0.3rem;'
                f'border-bottom:1px solid {C_BORDER};">{sec_title}</div>',
                unsafe_allow_html=True)
            pills = '<div style="display:flex;flex-wrap:wrap;gap:0.4rem;">'
            for k, v in items.items():
                lo, hi = RANGES.get(k, (None, None))
                try:
                    fv    = float(v)
                    color = val_status(fv, lo, hi) if lo is not None else C_TEXT
                    disp  = fmt_kpi(fv)
                except (TypeError, ValueError):
                    color = C_TEXT; disp = str(v)
                pills += (
                    f'<div style="border:1px solid {C_BORDER};border-radius:8px;'
                    f'padding:0.45rem 0.8rem;min-width:90px;flex:1;max-width:160px;">'
                    f'<div style="font-size:0.56rem;text-transform:uppercase;letter-spacing:0.09em;'
                    f'color:{C_MUTED};font-weight:700;margin-bottom:0.15rem;">{NICE.get(k,k)}</div>'
                    f'<div style="font-size:1rem;font-weight:600;color:{color};'
                    f'font-family:{FONT_CG};line-height:1.1;">{disp}</div>'
                    f'</div>'
                )
            pills += "</div>"
            st.markdown(pills, unsafe_allow_html=True)

    # ── Actions ───────────────────────────────────────────
    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Back to Results", use_container_width=True):
            st.session_state.app_step = "result"; st.rerun()
    with c2:
        if st.button("Start New Patient", use_container_width=True):
            reset(); st.rerun()

    st.markdown("""
    <div class="disclaimer">
      <strong>Clinical Disclaimer:</strong>
      This dashboard is for research and educational purposes only. All findings must be interpreted
      by a licensed OB-GYN or reproductive endocrinologist. PCOS detection is based on Rotterdam 2003
      criteria applied to entered clinical data. Feature importance scores reflect P2 model behaviour,
      not clinical causation.
    </div>""", unsafe_allow_html=True)