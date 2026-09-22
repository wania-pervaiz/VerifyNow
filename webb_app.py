
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# ==========================================
# 1. PREMIUM UI CONFIGURATION (CSS)
# ==========================================
st.set_page_config(
    page_title="VerifyNow | Fake News Detector",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for a clean, professional "Journal" aesthetic
st.markdown("""
    <style>
    /* Global Background - Using Pearl for a clean, high-end look */
    .stApp {
        background-color: #FCEABC !important; 
        background-image: 
            radial-gradient(at 0% 0%, rgba(243, 213, 141, 0.3) 0px, transparent 50%),   /* Buff */
            radial-gradient(at 100% 100%, rgba(38, 67, 101, 0.05) 0px, transparent 50%) !important; /* Police Blue */
        font-family: 'Inter', -apple-system, sans-serif;
    }
div[data-testid="stMainBlockContainer"] {
    padding-top: 0px !important;
}
   .header-banner-container {
    background-color: #B5D8FF !important; /* Kept your exact Azure Sky Blue */
    text-align: center;
    padding: 50px 20px;
    margin-top: 0px;
    margin-bottom: 40px;
    width: 100vw !important; /* Forces layout to span exact viewport width */
    position: relative;
    left: 50%;
    right: 50%;
    margin-left: -50vw;
    margin-right: -50vw;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
}
    .main-title {
        color: #264365 !important; /* High contrast Police Blue text */
        font-size: 52px;
        font-weight: 800;
        letter-spacing: -1.5px;
        margin: 0;
    }
    .sub-title {
        color: #8A3B08 !important; /* High contrast Citrine Brown text */
        font-size: 14px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-top: 10px;
    }

    /* Input Workspace Panel */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.6) !important;
        border: 1px solid #F3D58D !important; /* Buff Border */
        border-radius: 20px !important;
        padding: 20px !important;
        margin-bottom: 0px;
        box-shadow: 0 20px 40px rgba(38, 67, 101, 0.05) !important;
    }

    /* Diagnostic Cards */
    .metric-card {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 24px;
        border-left: 5px solid #F3D58D; /* Buff Accent */
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }

    /* Result Box - Solid Marigold Outline for New Output */
    .new-live-output-card {
        background: #ffffff !important;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 25px;
        box-shadow: 0 10px 30px rgba(229, 146, 44, 0.1);
    }

    /* Text Colors - Strictly Police Blue for Readability */
    h1, h2, h3, h4, h5, p, span, label {
        color: #264365 !important; 
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 4px;
        font-size: 10px;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 10px;
        color: #ffffff !important;
    }
    .badge-real { background-color: green !important; } 
    .new-live-output-card-real { border: 1px solid green !important;} 
    .badge-fake { background-color: red !important; } 
    .new-live-output-card-fake { border: 1px solid red !important;} 
    .badge-warning { background-color: #E5922C !important; } /* Marigold */

    /* FIXED: High-specificity query override for the Streamlit Form Button to ensure color applies */
    div[data-testid="stFormSubmitButton"] button, div.stButton > button {
        background-color: #B5D8FF !important; /* Solid Brown Accent */
        color: WHITE !important; 
        border: none !important;
        font-weight: 300 !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
    }
    div[data-testid="stFormSubmitButton"] button:hover, div.stButton > button:hover {
        background-color: #88b3e3 !important; /* Darker brown on focus hover */
        color: #ffffff !important;
    }

    /* Text Area */
    textarea {
        background-color: #ffffff !important;
        border: 1px solid #EBDDC5 !important;
        color: #264365 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. BRANDING (Header Banner Render)
st.markdown("""
    <div class="header-banner-container">
        <h1 class="main-title">VerifyNow🔍︎</h1>
        <p class="sub-title">Advanced NLP Detection & Linguistic Analysis</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 3. BACKEND (Passive Aggressive Classifier)
# ==========================================
@st.cache_resource
def load_and_train_model():
    fake_df = pd.read_csv("Fake.csv")
    true_df = pd.read_csv("True.csv")
    fake_df['label'] = 'FAKE'
    true_df['label'] = 'REAL'
    df = pd.concat([fake_df, true_df]).sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(10000) 

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        return text

    df['text'] = df['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(df['text'])
    y = df['label']
    model = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    model.fit(X, y)
    return model, vectorizer, clean_text

with st.spinner("Initializing neural verification layers..."):
    model, vectorizer, clean_text = load_and_train_model()

# ==========================================
# 4. INTERFACE
# ==========================================
# Setting up columns for a "centered" workspace
pad_l, main_col, pad_r = st.columns([1, 4, 1])

with main_col:
    with st.form("main_form"):
        user_input = st.text_area("Analysis Workspace", placeholder="Enter news headline or full article text here...", height=150, label_visibility="collapsed")
        c1, c2 = st.columns([4, 1])
        with c2:
            submit = st.form_submit_button("ANALYZE PATTERN")

    st.markdown("---")

    grid_left, grid_right = st.columns([2.5, 1.5])

    with grid_left:
        st.markdown("<h3 style='font-size:22px;'>System Diagnostics</h3>", unsafe_allow_html=True)

        if submit:
            if not user_input.strip():
                st.warning("Please provide input text.")
            else:
                cleaned = clean_text(user_input)
                vectorized = vectorizer.transform([cleaned])
                result = model.predict(vectorized)[0]
                snippet = user_input[:90] + "..." if len(user_input) > 90 else user_input

                if result == 'REAL':
                    st.markdown(f"""
                        <div class="new-live-output-card new-live-output-card-real">
                            <span class="badge badge-real">Verified Authentic</span>
                            <h4>Detection Result</h4>
                            <p style="font-style:italic; font-size:14px; color:#555;">"{snippet}"</p>
                            <p style="font-weight:700; color:#264365;">The text matches verified institutional reporting patterns.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="new-live-output-card new-live-output-card-fake">
                            <span class="badge badge-fake">Structural Anomaly</span>
                            <h4>Detection Result</h4>
                            <p style="font-style:italic; font-size:14px; color:#555;">"{snippet}"</p>
                            <p style="font-weight:700; color:#8A3B08;">Warning: Linguistic markers indicate synthetic or unverified origin.</p>
                        </div>
                    """, unsafe_allow_html=True)

        # Baseline Data
        st.markdown("""
            <div class="metric-card">
                <span class="badge badge-warning">Recent Check</span>
                <h4 style="margin:0;">Global Economic Review</h4>
                <p style="font-size:13px; margin:5px 0;">Linguistic structure verified against official 2024 treasury reports.</p>
            </div>
            <div class="metric-card">
                <span class="badge badge-warning">Recent Check</span>
                <h4 style="margin:0;">Climate Accord Claims</h4>
                <p style="font-size:13px; margin:5px 0;">Analysis flagged semantic drift in social media redistribution pipelines.</p>
            </div>
        """, unsafe_allow_html=True)

    with grid_right:
        st.markdown("<h3 style='font-size:22px;'>Trending</h3>", unsafe_allow_html=True)
        st.markdown(f"""
            <div style="background: #F3D58D; padding:20px; border-radius:12px;">
                <p style="font-weight:700; font-size:14px; margin-bottom:5px;">Deepfake Election Alerts</p>
                <p style="font-size:12px; margin-bottom:15px;">Monitoring surge in unverified video transcripts across social platforms.</p>

                <p style="font-weight:700; font-size:14px; margin-bottom:5px;">AI Bias Baselines</p>
                <p style="font-size:12px; margin-bottom:15px;">Measuring sentiment drift in automated news generators.</p>

                <p style="font-weight:700; font-size:14px; margin-bottom:5px;">Source Transparency</p>
                <p style="font-size:12px;">Updates to linguistic tracking for cross-border publishing pipelines.</p>
            </div>
        """, unsafe_allow_html=True)
