import streamlit as st
import numpy as np
import pandas as pd
import pickle
import traceback
import plotly.graph_objects as go
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
    "app_step": "overview",
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
    p1_features = pickle.load(open("p1_features.pkl", "rb"))
    return p1, p2, p1_features

p1_model, p2_model, P1_FEATURES = load_models()

# ─────────────────────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────────────────────
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
    X = pd.DataFrame([[inp.get(c, np.nan) for c in P1_FEATURES]], columns=P1_FEATURES)
    prob = p1_model.predict_proba(X)[0]
    positive_class_index = list(p1_model.classes_).index(1)
    return prob[positive_class_index] >= 0.5

def predict_phenotype(inp):
    X = np.array([[inp.get(c, np.nan) for c in P2_FEATURES]])
    probs_arr = p2_model.predict_proba(X)[0]
    probs = {c: round(float(p), 3) for c, p in zip(["A","B","C","D"], probs_arr)}
    return max(probs, key=probs.get), probs

def compute_shap_values(inp):
    try:
        model    = p1_model.named_steps["model"]
        selector = p1_model.named_steps["selector"]
        selected_names = [P1_FEATURES[i] for i in selector.selected_idx_]
        booster    = model.get_booster()
        importance = booster.get_score(importance_type="gain")
        scores = {}
        for fname, score in importance.items():
            try:
                idx = int(fname.replace("f", ""))
                if idx < len(selected_names):
                    scores[selected_names[idx]] = float(score)
            except ValueError:
                if fname in selected_names:
                    scores[fname] = float(score)
        for name in selected_names:
            if name not in scores:
                scores[name] = 0.0
        return scores, None
    except Exception:
        return None, traceback.format_exc()

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
    html += stage("P1", "PCOS Detection",          "Binary positive / negative", p1_state)
    html += stage("P2", "Phenotype Classification", "Types A / B / C / D",       p2_state)
    html += stage("DB", "Clinical Dashboard",       "Charts, SHAP & summary",     dash_state)
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
        elif app_step == "overview":
            state = "locked"
        else:
            state = "done"
        rows_html += _nav_row(num, label, state)
    st.markdown(rows_html, unsafe_allow_html=True)

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    if app_step == "overview":
        _pipeline_stages(p1_state="locked", p2_state="locked", dash_state="locked")
    elif app_step == "form":
        _pipeline_stages(p1_state="locked", p2_state="locked", dash_state="locked")
    elif app_step == "pcos_result":
        _pipeline_stages(p1_state="active", p2_state="locked", dash_state="locked")
    elif app_step == "phenotype_result":
        _pipeline_stages(p1_state="done",   p2_state="active", dash_state="locked")
    elif app_step == "dashboard":
        _pipeline_stages(p1_state="done",   p2_state="done",   dash_state="active")

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
elif app_step == "pcos_result":
    st.markdown('<div class="main-header"><h1>PCOS Detection Result</h1><p>Based on all entered clinical data.</p></div>', unsafe_allow_html=True)
elif app_step == "phenotype_result":
    st.markdown('<div class="main-header"><h1>Phenotype Classification Result</h1><p>PCOS confirmed — classifying phenotype.</p></div>', unsafe_allow_html=True)
elif app_step == "dashboard":
    st.markdown('<div class="main-header"><h1>Clinical Dashboard</h1><p>Full diagnostic summary, biomarker analysis, and model explainability.</p></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# OVERVIEW PAGE
# ─────────────────────────────────────────────────────────
if app_step == "overview":

    # ── Hero ─────────────────────────────────────────────
    rotterdam_criteria = [
        ("1", "Oligo / Anovulation",  "Irregular or absent menstrual cycles due to infrequent or absent ovulation"),
        ("2", "Hyperandrogenism",      "Elevated androgens — assessed clinically (acne, hirsutism) or biochemically"),
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

    # ── Why phenotypes ───────────────────────────────────
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
        'This tool uses a two-stage ML pipeline: <span class="ov-inline-pill">Pipeline 1</span> determines '
        'whether the entered clinical data is consistent with a PCOS diagnosis, then '
        '<span class="ov-inline-pill">Pipeline 2</span> classifies which phenotype applies &mdash; with '
        'confidence scores and a full clinical dashboard.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Severity bar ─────────────────────────────────────
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

    # ── Phenotype cards ───────────────────────────────────
    ph_data = [
        ("A", "#7c52cc", "rgba(124,82,204,0.08)", "#5a38b0",
         "Full Classic PCOS",
         "Anovulation + Hyperandrogenism + Polycystic Ovaries",
         "All three Rotterdam criteria are met. Associated with the highest prevalence of metabolic comorbidities including insulin resistance and dyslipidaemia.",
         ["Irregular or absent menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Polycystic ovarian morphology on ultrasound", "LH and AMH frequently elevated"],
         "Highest associated metabolic risk &middot; All 3 criteria met"),
        ("B", "#b83232", "rgba(184,50,50,0.08)", "#8a2020",
         "Classic without PCO",
         "Anovulation + Hyperandrogenism &middot; Normal ovarian morphology",
         "Cycle irregularity and androgen excess are present, but ovarian morphology on ultrasound is within normal limits. Biochemical workup is required for identification.",
         ["Irregular or absent menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Normal ovarian morphology on ultrasound", "Biochemical workup required for diagnosis"],
         "Elevated metabolic risk &middot; No polycystic morphology"),
        ("C", "#1a6e3c", "rgba(26,110,60,0.08)", "#0e4c28",
         "Ovulatory PCOS",
         "Hyperandrogenism + Polycystic Ovaries &middot; Regular cycles",
         "Ovulation is preserved despite androgen excess and polycystic ovarian morphology. May be identified only through ultrasound and androgen testing in the absence of cycle irregularity.",
         ["Regular menstrual cycles", "Clinical or biochemical hyperandrogenism",
          "Polycystic ovarian morphology on ultrasound", "Fertility often preserved"],
         "Lower metabolic risk &middot; Ovulation preserved"),
        ("D", "#9a7010", "rgba(154,112,16,0.08)", "#6b4e08",
         "Non-Androgenic PCOS",
         "Anovulation + Polycystic Ovaries &middot; No hyperandrogenism",
         "Cycle irregularity and polycystic ovarian morphology are present, but androgen levels are within normal limits. This phenotype is the subject of ongoing debate in the literature.",
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
                    + f +
                    '</div>'
                )
            card_html = (
                '<div class="ov-ph-card">'
                '<div class="ov-ph-bar" style="background:' + color + ';"></div>'
                '<div class="ov-ph-head">'
                '<div class="ov-ph-badge" style="background:' + color + ';">' + ph + '</div>'
                '<div>'
                '<div class="ov-ph-label">Phenotype ' + ph + ' &mdash; ' + sublabel + '</div>'
                '<div class="ov-ph-combo">' + combo + '</div>'
                '</div>'
                '</div>'
                '<p class="ov-ph-desc">' + description + '</p>'
                '<div class="ov-ph-features">' + feats_html + '</div>'
                '<div class="ov-ph-note" style="background:' + bg + ';color:' + txt_color + ';">' + note + '</div>'
                '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # ── How this tool works ──────────────────────────────
    steps_data = [
        ("01", "Enter clinical data",
         "Six sections: anthropometrics, vitals, menstrual history, laboratory values, ultrasound findings, and reported symptoms.",
         "~5 min to complete", "#f5f0fe", "#7a5caa", "#ddd0f5"),
        ("P1", "PCOS detection",
         "An XGBoost classifier trained on 16 selected features returns a binary positive or negative prediction based on the entered data.",
         "Binary classifier", "#f5f0fe", "#5a38b0", "rgba(124,82,204,0.25)"),
        ("P2", "Phenotype classification",
         "If positive, a Random Forest pipeline classifies the case into one of four Rotterdam phenotypes with per-class probability scores and a clinical dashboard.",
         "Multi-class &middot; A / B / C / D", "#f5f0fe", "#5a38b0", "rgba(124,82,204,0.25)"),
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

    # ── Disclaimer ────────────────────────────────────────
    st.markdown(
        '<div class="ov-disclaimer">'
        '<strong>Clinical disclaimer</strong> &mdash; This tool is intended for research and educational purposes only. '
        'It does not constitute a medical diagnosis. All outputs must be reviewed and confirmed by a licensed '
        'OB-GYN, reproductive endocrinologist, or qualified healthcare professional. ML models were trained on '
        'a specific clinical dataset and may not generalise to all patient populations.'
        '</div>',
        unsafe_allow_html=True
    )

    # ── CTA ───────────────────────────────────────────────
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

    # ── SECTION 0: ANTHROPOMETRIC ───────────────────────
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

    # ── SECTION 1: VITALS ────────────────────────────────
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

    # ── SECTION 2: MENSTRUAL ─────────────────────────────
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
                    "cycle_ri_label":cycle_ri,
                    "cycle_ri":1 if cycle_ri=="Irregular" else 0,
                    "cycle_24":4 if cycle_ri=="Irregular" else 2,
                    "marriage_yr":marriage,
                    "pregnant_label":pregnant,
                    "pregnant":1 if pregnant=="Yes" else 0,
                    "abortions":abortions,
                })
                st.session_state.active_section = 3; st.rerun()
        if back: st.session_state.active_section = 1; st.rerun()

    # ── SECTION 3: LABS ──────────────────────────────────
    elif active_sec == 3:
        with st.form("sec_3"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">04</div>
                <div><div class="section-title">Laboratory Values</div>
                <div class="section-desc">Hormones and blood markers</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            hb          = c1.number_input("Hemoglobin (g/dl)",     min_value=5.0,  max_value=20.0,  value=sd.get("hb",          None), step=0.1, placeholder="Type here")
            beta_hcg_i  = c2.number_input("Beta-HCG I (mIU/mL)",  min_value=0.0,  max_value=500.0, value=sd.get("beta_hcg_i",  None), step=0.1, placeholder="Type here")
            beta_hcg_ii = c3.number_input("Beta-HCG II (mIU/mL)", min_value=0.0,  max_value=500.0, value=sd.get("beta_hcg_ii", None), step=0.1, placeholder="Type here")
            fsh         = c4.number_input("FSH (mIU/mL)",          min_value=0.0,  max_value=30.0,  value=sd.get("fsh",         None), step=0.1, placeholder="Type here")

            c1, c2, c3, c4 = st.columns(4)
            lh    = c1.number_input("LH (mIU/mL)",  min_value=0.0, max_value=50.0,  value=sd.get("lh",  None), step=0.1, placeholder="Type here")
            tsh   = c2.number_input("TSH (mIU/L)",  min_value=0.0, max_value=10.0,  value=sd.get("tsh", None), step=0.1, placeholder="Type here")
            prl   = c3.number_input("PRL (ng/mL)",  min_value=0.0, max_value=100.0, value=sd.get("prl", None), step=0.1, placeholder="Type here")
            amh   = c4.number_input("AMH (ng/mL)",  min_value=0.0, max_value=15.0,  value=sd.get("amh", None), step=0.1, placeholder="Type here")

            c1, c2, c3, c4 = st.columns(4)
            vit_d = c1.number_input("Vitamin D3 (ng/mL)",         min_value=0.0, max_value=100.0, value=sd.get("vit_d", None), step=0.1, placeholder="Type here")
            prg   = c2.number_input("Progesterone (ng/mL)",       min_value=0.0, max_value=30.0,  value=sd.get("prg",   None), step=0.1, placeholder="Type here")
            rbs   = c3.number_input("Random Blood Sugar (mg/dl)", min_value=50.0,max_value=400.0, value=sd.get("rbs",   None), step=1.0, placeholder="Type here")

            fsh_lh = round(fsh / lh, 3) if (fsh and lh and lh > 0) else sd.get("_calc_fshlh")
            fsh_lh_display = f"{fsh_lh:.3f}" if fsh_lh else "—"
            c4.markdown(f'<div class="auto-pill">FSH/LH Ratio<span>{fsh_lh_display}</span></div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            calc_col, _, next_col = st.columns([1, 0.1, 3.9])
            with calc_col:
                calc = st.form_submit_button("Calculate", use_container_width=True)
            with next_col:
                nxt = st.form_submit_button("Next — Ultrasound Findings", use_container_width=True)
            back = st.form_submit_button("Back")

        if calc:
            _fshlh = round(fsh/lh, 3) if (fsh and lh and lh > 0) else None
            sd.update({
                "hb":hb,"beta_hcg_i":beta_hcg_i,"beta_hcg_ii":beta_hcg_ii,
                "fsh":fsh,"lh":lh,"tsh":tsh,"prl":prl,"amh":amh,
                "vit_d":vit_d,"prg":prg,"rbs":rbs,"_calc_fshlh":_fshlh,
            })
            st.rerun()

        if nxt:
            lab_fields = [
                ("Hemoglobin",hb),("Beta-HCG I",beta_hcg_i),("Beta-HCG II",beta_hcg_ii),
                ("FSH",fsh),("LH",lh),("TSH",tsh),("PRL",prl),("AMH",amh),
                ("Vitamin D3",vit_d),("Progesterone",prg),("Random Blood Sugar",rbs),
            ]
            missing = [f for f, v in lab_fields if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                _fshlh = round(fsh/lh, 3) if (fsh and lh and lh > 0) else None
                sd.update({
                    "hb":hb,"beta_hcg_i":beta_hcg_i,"beta_hcg_ii":beta_hcg_ii,
                    "fsh":fsh,"lh":lh,"fsh_lh":_fshlh,"tsh":tsh,"prl":prl,
                    "amh":amh,"vit_d":vit_d,"prg":prg,"rbs":rbs,"_calc_fshlh":_fshlh,
                })
                st.session_state.active_section = 4; st.rerun()
        if back: st.session_state.active_section = 2; st.rerun()

    # ── SECTION 4: ULTRASOUND ────────────────────────────
    elif active_sec == 4:
        with st.form("sec_4"):
            st.markdown("""<div class="section-card"><div class="section-header">
                <div class="section-icon">05</div>
                <div><div class="section-title">Ultrasound Findings</div>
                <div class="section-desc">Follicle count and size, endometrium thickness</div></div>
            </div>""", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            follicle_l  = c1.number_input("Follicle No. (Left)",       min_value=0,   max_value=30,   value=sd.get("follicle_l", None), placeholder="Type here")
            follicle_r  = c2.number_input("Follicle No. (Right)",      min_value=0,   max_value=30,   value=sd.get("follicle_r", None), placeholder="Type here")
            avg_f_l     = c3.number_input("Avg. Follicle Size L (mm)", min_value=0.0, max_value=30.0, value=sd.get("avg_f_l",   None), step=0.5, placeholder="Type here")
            avg_f_r     = c4.number_input("Avg. Follicle Size R (mm)", min_value=0.0, max_value=30.0, value=sd.get("avg_f_r",   None), step=0.5, placeholder="Type here")
            endometrium = st.number_input("Endometrium Thickness (mm)", min_value=0.0, max_value=20.0, value=sd.get("endometrium", None), step=0.1, placeholder="Type here")

            st.markdown("</div>", unsafe_allow_html=True)
            back, nxt = nav_buttons("sec_4", 3, "Next — Clinical Symptoms")

        if nxt:
            missing = [f for f, v in [
                ("Follicle No. Left",follicle_l),("Follicle No. Right",follicle_r),
                ("Avg. Follicle Size L",avg_f_l),("Avg. Follicle Size R",avg_f_r),
                ("Endometrium Thickness",endometrium),
            ] if v is None]
            if missing:
                st.error("Some fields are incomplete. Please review all inputs before continuing.")
            else:
                sd.update({
                    "follicle_l":follicle_l,"follicle_r":follicle_r,
                    "avg_f_l":avg_f_l,"avg_f_r":avg_f_r,"endometrium":endometrium,
                })
                st.session_state.active_section = 5; st.rerun()
        if back: st.session_state.active_section = 3; st.rerun()

    # ── SECTION 5: SYMPTOMS ──────────────────────────────
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
                "LH(mIU/mL)":s["lh"], "FSH/LH":s.get("fsh_lh"), "Waist(inch)":s["waist"],
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
                    color:white;font-family:'Libre Baskerville',serif;font-size:1.2rem;font-weight:700;">
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
    inp  = st.session_state.inputs
    sd   = st.session_state.section_data
    ph, probs = st.session_state.phenotype_result
    info = PHENOTYPE_INFO[ph]

    C_PURPLE = "#6c3fc5"
    C_NAVY   = "#1a0e36"
    C_BORDER = "#e0d5f5"
    C_TEXT   = "#2e1a58"
    C_MUTED  = "#9580b8"
    C_HIGH   = "#b91c1c"
    C_LOW    = "#c2410c"
    C_OK     = "#166534"
    FONT_SORA = "Sora, sans-serif"
    FONT_CG   = "Cormorant Garamond, serif"

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

    # ── KPI strip ─────────────────────────────────────────
    section_label("Key Biomarkers")
    KPI_DEF = [
        ("BMI",         sd.get("bmi",  0) or 0, 18.5, 24.9,  "kg/m²"),
        ("AMH",         sd.get("amh",  0) or 0,  1.0,  3.5,  "ng/mL"),
        ("FSH",         sd.get("fsh",  0) or 0,  3.0, 10.0,  "mIU/mL"),
        ("LH",          sd.get("lh",   0) or 0,  2.0, 15.0,  "mIU/mL"),
        ("FSH/LH",      sd.get("fsh_lh",0) or 0, 1.0,  3.0,  "ratio"),
        ("TSH",         sd.get("tsh",  0) or 0,  0.4,  4.0,  "mIU/L"),
        ("PRL",         sd.get("prl",  0) or 0,  2.0, 29.0,  "ng/mL"),
        ("Vitamin D3",  sd.get("vit_d",0) or 0, 20.0, 50.0,  "ng/mL"),
        ("RBS",         sd.get("rbs",  0) or 0, 70.0,140.0,  "mg/dl"),
        ("Hemoglobin",  sd.get("hb",   0) or 0, 12.0, 16.0,  "g/dl"),
    ]

    def kpi_html(label, val, lo, hi, unit):
        color = val_status(val, lo, hi)
        arrow = " ↑" if val > hi else (" ↓" if val < lo else "")
        try:    display = f"{float(val):.2g}"
        except: display = str(val)
        return (
            f'<div style="background:#ffffff;border:1px solid {C_BORDER};border-radius:10px;'
            f'padding:0.8rem 0.6rem;text-align:center;height:100%;">'
            f'<div style="font-size:0.55rem;text-transform:uppercase;letter-spacing:0.1em;'
            f'color:{C_MUTED};font-weight:700;margin-bottom:0.3rem;">{label}</div>'
            f'<div style="font-family:{FONT_CG};font-size:1.55rem;font-weight:600;'
            f'color:{color};line-height:1;">{display}<span style="font-size:0.7rem;">{arrow}</span></div>'
            f'<div style="font-size:0.58rem;color:{C_MUTED};margin-top:0.15rem;">{unit}</div>'
            f'</div>'
        )

    for row_items in [KPI_DEF[:5], KPI_DEF[5:]]:
        cells = "".join(kpi_html(l, v, lo, hi, u) for l, v, lo, hi, u in row_items)
        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:0.5rem;margin-bottom:0.5rem;">{cells}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)

    # ── Phenotype donut | Biomarker bar ───────────────────
    col_ph, col_bio = st.columns([1, 1.65], gap="large")

    with col_ph:
        section_label("Phenotype Classification")
        ph_colors = {"A": "#6c3fc5", "B": "#9f1239", "C": "#166534", "D": "#92400e"}
        sorted_probs = dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
        donut_labels = list(sorted_probs.keys())
        donut_vals   = list(sorted_probs.values())

        fig_donut = go.Figure(go.Pie(
            labels=donut_labels, values=donut_vals, hole=0.68,
            marker=dict(colors=[ph_colors[k] for k in donut_labels], line=dict(color="#ffffff", width=3)),
            textinfo="percent", textposition="outside",
            textfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
            outsidetextfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
            hovertemplate="<b>Phenotype %{label}</b><br>%{customdata}<br>Probability: %{percent}<extra></extra>",
            customdata=[PHENOTYPE_INFO[k]["sublabel"] for k in donut_labels],
            pull=[0.06 if k == ph else 0 for k in donut_labels],
            direction="clockwise", rotation=90, showlegend=False,
        ))
        fig_donut.add_annotation(text=f"<b>{ph}</b>", x=0.5, y=0.55,
            font=dict(family=FONT_CG, size=46, color=C_NAVY), showarrow=False)
        fig_donut.add_annotation(text=info["sublabel"].replace(" ", "<br>"), x=0.5, y=0.35,
            font=dict(family=FONT_SORA, size=9, color=C_MUTED), showarrow=False)
        fig_donut.update_layout(**BASE_LAYOUT, height=300, margin=dict(l=30, r=30, t=30, b=30))
        st.plotly_chart(fig_donut, use_container_width=True, config={"displayModeBar": False})

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
            ("BMI",          sd.get("bmi",  0) or 0, 18.5, 24.9,  "kg/m²"),
            ("AMH",          sd.get("amh",  0) or 0,  1.0,  3.5,  "ng/mL"),
            ("FSH",          sd.get("fsh",  0) or 0,  3.0, 10.0,  "mIU/mL"),
            ("LH",           sd.get("lh",   0) or 0,  2.0, 15.0,  "mIU/mL"),
            ("TSH",          sd.get("tsh",  0) or 0,  0.4,  4.0,  "mIU/L"),
            ("PRL",          sd.get("prl",  0) or 0,  2.0, 29.0,  "ng/mL"),
            ("Vitamin D3",   sd.get("vit_d",0) or 0, 20.0, 50.0,  "ng/mL"),
            ("Progesterone", sd.get("prg",  0) or 0,  1.0, 25.0,  "ng/mL"),
            ("RBS",          sd.get("rbs",  0) or 0, 70.0,140.0,  "mg/dl"),
            ("Hemoglobin",   sd.get("hb",   0) or 0, 12.0, 16.0,  "g/dl"),
        ]
        names   = [d[0] for d in bm_data]
        vals    = [d[1] for d in bm_data]
        hi_vals = [d[3] for d in bm_data]
        lo_vals = [d[2] for d in bm_data]
        units   = [d[4] for d in bm_data]
        norm_pct = [min(v / hi * 100, 155) for v, hi in zip(vals, hi_vals)]
        norm_lo  = [lo / hi * 100 for lo, hi in zip(lo_vals, hi_vals)]
        status_lbl = ["High ↑" if v > hi else ("Low ↓" if v < lo else "Normal")
                      for v, lo, hi in zip(vals, lo_vals, hi_vals)]

        fig_bio = go.Figure()
        fig_bio.add_trace(go.Bar(y=names, x=[155]*len(names), orientation="h",
            marker_color="rgba(108,63,197,0.07)", hoverinfo="skip", showlegend=False))
        fig_bio.add_trace(go.Bar(y=names, x=norm_pct, orientation="h",
            marker=dict(color=C_PURPLE, opacity=0.75, line=dict(color="rgba(255,255,255,0.3)", width=0.5)),
            text=[f"  {v:.3g} {u}  ·  {s}" for v, u, s in zip(vals, units, status_lbl)],
            textposition="outside",
            textfont=dict(family=FONT_SORA, size=9.5, color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>"))
        fig_bio.add_trace(go.Scatter(y=names, x=norm_lo, mode="markers",
            marker=dict(symbol="line-ns", size=14, color=C_MUTED, line=dict(width=1.5, color=C_MUTED)),
            hoverinfo="skip", showlegend=False))
        fig_bio.update_layout(
            **BASE_LAYOUT, barmode="overlay", height=370,
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

    # ── XAI | Radar ───────────────────────────────────────
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    section_label("Model Explainability")

    with st.spinner("Computing feature importance…"):
        shap_result, base_or_err = compute_shap_values(inp)

    col_imp, col_radar = st.columns([1.4, 1], gap="large")

    with col_imp:
        if shap_result:
            sorted_imp  = sorted(shap_result.items(), key=lambda x: x[1], reverse=True)[:10]
            feat_labels = [k for k, _ in sorted_imp]
            feat_vals   = [v for _, v in sorted_imp]
            max_v       = max(feat_vals) if feat_vals else 1
            norm_imp    = [v / max_v * 100 for v in feat_vals]
            bar_opacities = [max(0.45, 0.9 - i * 0.045) for i in range(len(feat_labels))]

            fig_imp = go.Figure()
            for i in range(0, len(feat_labels), 2):
                fig_imp.add_shape(type="rect", xref="paper", yref="y",
                    x0=0, x1=1, y0=i-0.5, y1=i+0.5,
                    fillcolor="rgba(108,63,197,0.03)", line=dict(width=0), layer="below")
            fig_imp.add_trace(go.Bar(
                y=feat_labels[::-1], x=norm_imp[::-1], orientation="h",
                marker=dict(
                    color=[f"rgba(108,63,197,{bar_opacities[::-1][i]:.2f})" for i in range(len(feat_labels))],
                    line=dict(color="rgba(255,255,255,0.2)", width=0.5), cornerradius=3),
                text=[f"<b>{v:.1f}</b>" for v in norm_imp[::-1]],
                textposition="outside",
                textfont=dict(family=FONT_SORA, size=10, color=C_TEXT),
                hovertemplate="<b>%{y}</b><br>Relative importance: %{x:.1f}<extra></extra>"))
            fig_imp.add_vline(x=100, line=dict(color=C_PURPLE, width=1, dash="dot"),
                annotation_text="top feature", annotation_position="top right",
                annotation_font=dict(size=8, color=C_MUTED))
            fig_imp.update_layout(**BASE_LAYOUT, height=370,
                margin=dict(l=0, r=70, t=10, b=30),
                xaxis=dict(**axis_style(),
                    title=dict(text="Relative importance (normalised, 0–100)",
                        font=dict(family=FONT_SORA, size=9, color=C_MUTED)), range=[0, 125]),
                yaxis=dict(tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
                    showgrid=False, autorange=True))
            st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
            top_feat = sorted_imp[0][0]
            st.markdown(
                f'<p style="font-size:0.72rem;color:{C_MUTED};margin-top:-0.3rem;">'
                f'Bars show average decision gain per feature split. '
                f'<b style="color:{C_TEXT};">{top_feat}</b> is the most influential predictor.</p>',
                unsafe_allow_html=True)
        else:
            st.warning("Feature importance could not be computed for this model configuration.")

    with col_radar:
        radar_defs = [
            ("AMH",       sd.get("amh",    0) or 0, 3.5),
            ("FSH/LH",    sd.get("fsh_lh", 0) or 0, 3.0),
            ("BMI",       sd.get("bmi",    0) or 0, 24.9),
            ("Waist:Hip", sd.get("whr",    0) or 0, 0.85),
            ("RBS",       sd.get("rbs",    0) or 0, 140.0),
            ("LH",        sd.get("lh",     0) or 0, 15.0),
        ]
        r_labels = [d[0] for d in radar_defs]
        r_vals   = [min(d[1] / d[2] * 100, 140) for d in radar_defs]
        r_labels_closed = r_labels + [r_labels[0]]
        r_vals_closed   = r_vals   + [r_vals[0]]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=[100]*len(r_labels_closed), theta=r_labels_closed, fill=None, mode="lines",
            line=dict(color=C_BORDER, width=1.5, dash="dot"),
            name="Reference (upper normal)", hoverinfo="skip", showlegend=True))
        fig_radar.add_trace(go.Scatterpolar(
            r=r_vals_closed, theta=r_labels_closed, fill="toself",
            fillcolor="rgba(108,63,197,0.12)", mode="lines+markers",
            line=dict(color=C_PURPLE, width=2), marker=dict(size=5, color=C_PURPLE),
            name="Patient values",
            hovertemplate="<b>%{theta}</b><br>%{r:.1f}% of upper limit<extra></extra>",
            showlegend=True))
        fig_radar.update_layout(
            **BASE_LAYOUT, height=370, margin=dict(l=30, r=30, t=30, b=30),
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 140],
                    tickfont=dict(family=FONT_SORA, size=8, color=C_MUTED),
                    gridcolor="rgba(180,165,220,0.2)", linecolor="rgba(180,165,220,0.2)",
                    ticksuffix="%"),
                angularaxis=dict(tickfont=dict(family=FONT_SORA, size=10, color=C_TEXT),
                    gridcolor="rgba(180,165,220,0.2)", linecolor="rgba(180,165,220,0.2)")),
            legend=dict(font=dict(family=FONT_SORA, size=9, color=C_MUTED),
                orientation="h", yanchor="bottom", y=-0.08, xanchor="center", x=0.5))
        st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<p style="font-size:0.72rem;color:{C_MUTED};margin-top:-0.3rem;text-align:center;">'
            f'Values shown as % of upper reference limit. Dotted ring = normal ceiling.</p>',
            unsafe_allow_html=True)

    # ── Follicle | Phenotype drivers ──────────────────────
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    col_foll, col_sym = st.columns(2, gap="large")

    with col_foll:
        section_label("Ultrasound — Follicle Count")
        fl = sd.get("follicle_l", 0) or 0
        fr = sd.get("follicle_r", 0) or 0
        fig_foll = go.Figure()
        fig_foll.add_trace(go.Bar(
            x=["Left Ovary","Right Ovary"], y=[fl, fr],
            marker=dict(color=[C_PURPLE, "rgba(108,63,197,0.5)"],
                line=dict(color="rgba(255,255,255,0.3)", width=0.5), cornerradius=4),
            text=[f"<b>{fl}</b>", f"<b>{fr}</b>"], textposition="outside",
            textfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
            hovertemplate="<b>%{x}</b><br>Follicles: %{y}<extra></extra>", width=0.35))
        fig_foll.add_hline(y=12, line=dict(color=C_HIGH, width=1.5, dash="dash"),
            annotation_text="Polycystic threshold (≥12)", annotation_position="top right",
            annotation_font=dict(size=8.5, color=C_HIGH))
        fig_foll.update_layout(**BASE_LAYOUT, height=260,
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT), showgrid=False),
            yaxis=dict(**axis_style(),
                title=dict(text="Follicle count", font=dict(family=FONT_SORA, size=9, color=C_MUTED)),
                range=[0, max(max(fl, fr) * 1.45, 15)]),
            showlegend=False)
        st.plotly_chart(fig_foll, use_container_width=True, config={"displayModeBar": False})

    with col_sym:
        section_label("Phenotype Driver Features")
        driver_defs = [
            ("AMH",         sd.get("amh",       0) or 0, 3.5,  "ng/mL"),
            ("LH",          sd.get("lh",        0) or 0, 15.0, "mIU/mL"),
            ("FSH/LH",      sd.get("fsh_lh",    0) or 0, 3.0,  "ratio"),
            ("Follicles L", sd.get("follicle_l",0) or 0, 12.0, "count"),
            ("Follicles R", sd.get("follicle_r",0) or 0, 12.0, "count"),
            ("Waist:Hip",   sd.get("whr",       0) or 0, 0.85, "ratio"),
        ]
        drv_labels = [d[0] for d in driver_defs]
        drv_raw    = [d[1] for d in driver_defs]
        drv_ref    = [d[2] for d in driver_defs]
        drv_units  = [d[3] for d in driver_defs]
        drv_pct    = [min(v/ref*100, 160) if ref else 0 for v, ref in zip(drv_raw, drv_ref)]
        drv_colors = [C_PURPLE if pct >= 100 else f"rgba(108,63,197,{max(0.25, pct/100*0.7):.2f})"
                      for pct in drv_pct]

        fig_drv = go.Figure()
        fig_drv.add_vline(x=100, line=dict(color=C_MUTED, width=1.2, dash="dot"),
            annotation_text="threshold", annotation_position="top right",
            annotation_font=dict(size=8, color=C_MUTED))
        fig_drv.add_trace(go.Bar(
            y=drv_labels[::-1], x=drv_pct[::-1], orientation="h",
            marker=dict(color=drv_colors[::-1],
                line=dict(color="rgba(255,255,255,0.2)", width=0.5), cornerradius=3),
            text=[f"  {v:.3g} {u}" for v, u in zip(drv_raw[::-1], drv_units[::-1])],
            textposition="outside",
            textfont=dict(family=FONT_SORA, size=9.5, color=C_TEXT),
            hovertemplate="<b>%{y}</b><br>%{x:.1f}% of threshold<extra></extra>"))
        fig_drv.update_layout(**BASE_LAYOUT, height=260,
            margin=dict(l=0, r=90, t=20, b=10),
            xaxis=dict(**axis_style(), ticksuffix="%", range=[0, 195],
                title=dict(text="% of clinical threshold",
                    font=dict(family=FONT_SORA, size=9, color=C_MUTED))),
            yaxis=dict(tickfont=dict(family=FONT_SORA, size=11, color=C_TEXT),
                showgrid=False, autorange=True),
            showlegend=False)
        st.plotly_chart(fig_drv, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f'<p style="font-size:0.72rem;color:{C_MUTED};margin-top:-0.3rem;">'
            f'Bars show patient value as % of upper clinical threshold. Filled bars exceed threshold.</p>',
            unsafe_allow_html=True)

    # ── Full data record ──────────────────────────────────
    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    with st.expander("📋 Full Patient Data Record"):
        NICE = {
            "age":"Age (yrs)", "weight":"Weight (kg)", "height":"Height (cm)",
            "bmi":"BMI", "hip":"Hip (in)", "waist":"Waist (in)",
            "whr":"Waist:Hip", "blood_group":"Blood Type",
            "pulse":"Pulse (bpm)", "bp_sys":"BP Systolic", "bp_dia":"BP Diastolic",
            "cycle_ri_label":"Cycle", "marriage_yr":"Marriage (yrs)",
            "pregnant_label":"Pregnant", "abortions":"Abortions",
            "hb":"Hemoglobin", "beta_hcg_i":"β-HCG I", "beta_hcg_ii":"β-HCG II",
            "fsh":"FSH", "lh":"LH", "fsh_lh":"FSH/LH", "tsh":"TSH",
            "prl":"PRL", "amh":"AMH", "vit_d":"Vit D3",
            "prg":"Progesterone", "rbs":"RBS",
            "follicle_l":"Follicles L", "follicle_r":"Follicles R",
            "avg_f_l":"Avg Size L", "avg_f_r":"Avg Size R",
            "endometrium":"Endometrium",
            "weight_gain_label":"Weight Gain", "hair_growth_label":"Hair Growth",
            "skin_dark_label":"Skin Dark.", "pimples_label":"Pimples",
            "fast_food_label":"Fast Food", "exercise_label":"Exercises",
        }
        SECS = {
            "Anthropometric": ["age","weight","height","bmi","hip","waist","whr","blood_group"],
            "Vitals":         ["pulse","bp_sys","bp_dia"],
            "Menstrual":      ["cycle_ri_label","marriage_yr","pregnant_label","abortions"],
            "Laboratory":     ["hb","beta_hcg_i","beta_hcg_ii","fsh","lh","fsh_lh","tsh","prl","amh","vit_d","prg","rbs"],
            "Ultrasound":     ["follicle_l","follicle_r","avg_f_l","avg_f_r","endometrium"],
            "Symptoms":       ["weight_gain_label","hair_growth_label","skin_dark_label","pimples_label","fast_food_label","exercise_label"],
        }
        RANGES = {
            "bmi":(18.5,24.9),"whr":(0,0.85),"hb":(12,16),"fsh":(3,10),
            "lh":(2,15),"fsh_lh":(1,3),"tsh":(0.4,4),"prl":(2,29),
            "amh":(1,3.5),"vit_d":(20,50),"prg":(1,25),"rbs":(70,140),
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
                    fv = float(v); color = val_status(fv, lo, hi) if lo is not None else C_TEXT
                    disp = f"{fv:.3g}"
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
            st.session_state.app_step = "phenotype_result"; st.rerun()
    with c2:
        if st.button("Start New Patient", use_container_width=True):
            reset(); st.rerun()

    st.markdown("""
    <div class="disclaimer">
      <strong>Clinical Disclaimer:</strong>
      This dashboard is for research and educational purposes only. All findings must be interpreted
      by a licensed OB-GYN or reproductive endocrinologist. Feature importance scores reflect model
      behaviour, not clinical causation.
    </div>""", unsafe_allow_html=True)