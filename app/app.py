# app/app.py
# Streamlit-based Web Application for Phishing URL Detection
# Usage:
# streamlit run app/app.py
# or
# python -m streamlit run app/app.py

import sys
from pathlib import Path

import joblib
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
SRC_DIR = BASE_DIR / "src"
MODEL_PATH = BASE_DIR / "artifacts" / "phishing_rf_pipeline.joblib"

for p in (BASE_DIR, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.features import extract_features

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

def normalize_url(url: str) -> str:
    url = (url or "").strip()
    if url and not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

# UI helpers
PAGES = ["Home", "Scan", "Safety Guide", "About", "Contact"]

def ui_setup():
    st.set_page_config(
        page_title="Phishing URL Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_css()

    if "page" not in st.session_state:
        st.session_state.page = "Home"

    if "nav_choice" not in st.session_state:
        st.session_state.nav_choice = st.session_state.page

    # Scan page state
    st.session_state.setdefault("scan_url", "")
    st.session_state.setdefault("scan_has_result", False)
    st.session_state.setdefault("scan_result", {})

def inject_css():
    st.markdown(
        """
        <style>
          .block-container { max-width: 1180px; padding-top: 2.8rem; padding-bottom: 2.2rem; }

          /* White background */
          [data-testid="stAppViewContainer"] { background: #ffffff; }
          [data-testid="stSidebar"] {
            background: #fbfbfc;
            border-right: 1px solid #eef0f3;
          }

          /* Hide Streamlit menu/footer */
          #MainMenu { visibility: hidden; }
          footer { visibility: hidden; }

          .topbar {
            border-radius: 16px;
            border: 1px solid #e8edf3;
            background: #ffffff;
            padding: 18px 20px;
            margin-bottom: 22px;
            box-shadow: 0 1px 0 rgba(15,23,42,0.02);
            outline: 1px solid rgba(232,237,243,0.0);
            outline-offset: 2px;
          }

          .brand { display: flex; align-items: center; gap: 12px; }

          .logo {
            width: 42px; height: 42px; 
            border-radius: 12px;
            display: inline-flex; align-items: center; justify-content: center;
            background: linear-gradient(135deg, #0ea5a6, #2563eb);
            color: #ffffff;
            font-size: 20px;          
            flex: 0 0 auto;
          }

          .brand-text { min-width: 0; }

          .brand-title {
            font-weight: 800;
            letter-spacing: .2px;
            font-size: 17px;  
            color: #0f172a;
            line-height: 1.3; 
            word-break: break-word;
          }

          .brand-sub { color: #64748b; font-size: 12.5px; margin-top: 2px; line-height: 1.35; }

          /* Hero */
          .hero {
            border-radius: 20px;
            border: 1px solid #e8edf3;
            background: linear-gradient(180deg, #ffffff, #f8fafc);
            padding: 26px;
          }
          .hero h1 { margin: 0; font-size: 34px; line-height: 1.1; color: #0f172a; }
          .hero p {
            margin: 10px 0 0 0;
            color: #475569;
            font-size: 14px;
            max-width: 78ch;
            line-height: 1.65;
          }

          /* Cards */
          .card {
            border-radius: 16px;
            border: 1px solid #e8edf3;
            background: #ffffff;
            padding: 16px;
            box-shadow: 0 8px 24px rgba(15,23,42,0.04);
          }
          .card h3 { margin: 0 0 6px 0; font-size: 16px; color: #0f172a; }
          .muted { color: #475569; font-size: 13px; line-height: 1.55; }

          /* Placeholder image box */
          .ph {
            border-radius: 16px;
            border: 1px dashed #cbd5e1;
            background: #ffffff;
            padding: 18px;
            min-height: 240px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #64748b;
            text-align: center;
            font-size: 13px;
            line-height: 1.5;
          }

          /* Result banner */
          .result {
            border-radius: 16px;
            padding: 14px 16px;
            border: 1px solid #e8edf3;
            background: #ffffff;
            box-shadow: 0 8px 24px rgba(15,23,42,0.04);
          }
          .badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            font-weight: 800;
            font-size: 13px;
            border: 1px solid #e8edf3;
            background: #f8fafc;
            color: #0f172a;
          }
          .badge.safe { border-color: #bfe3da; }
          .badge.warn { border-color: #f5c2c7; }

          /* Footer */
          .footer {
            margin-top: 26px;
            padding: 14px 16px;
            border-radius: 14px;
            border: 1px solid #e8edf3;
            background: #ffffff;
            color: #64748b;
            font-size: 12px;
            text-align: center;
          }

          /* Buttons */
          div.stButton > button {
            border-radius: 12px !important;
            padding: 0.55rem 0.95rem !important;
            border: 1px solid #dbe3ee !important;
            background: #ffffff !important;
            color: #0f172a !important;
          }
          div.stButton > button:hover {
            border-color: #bcd0f5 !important;
            background: #f8fafc !important;
          }

          pre { border-radius: 12px !important; }
          
          [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
          }

          h1, h2, h3, h4 { letter-spacing: -0.2px; }
          .hero h1 { letter-spacing: -0.6px; }
          .hero p, .muted, .brand-sub { text-rendering: geometricPrecision; }

          .card, .hero, .result, .footer, .topbar, .ph {
            transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease;
          }

          .card:hover {
            transform: translateY(-2px);
            box-shadow: 0 14px 34px rgba(15,23,42,0.08);
            border-color: #d7e3f8;
          }

          .hero {
            box-shadow: 0 10px 28px rgba(15,23,42,0.06);
          }

          .ph:hover {
            border-color: #bcd0f5;
          }

          [data-testid="stTextInput"] input {
            border-radius: 12px !important;
            border: 1px solid #dbe3ee !important;
            padding: 0.7rem 0.9rem !important;
            background: #ffffff !important;
          }

          [data-testid="stTextInput"] input:focus {
            border-color: #9fbef7 !important;
            box-shadow: 0 0 0 3px rgba(159,190,247,0.35) !important;
            outline: none !important;
          }

          div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb, #0ea5a6) !important;
            color: #ffffff !important;
            border: none !important;
          }

          div.stButton > button[kind="primary"]:hover {
            filter: brightness(0.98);
          }

          div.stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(15,23,42,0.06);
          }

          section[data-testid="stSidebar"] [role="radiogroup"] label {
            padding: 6px 10px;
            border-radius: 10px;
          }

          section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
            background: #f4f7fb;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

def header_bar():
    st.markdown(
        """
        <div class="topbar">
          <div class="brand">
            <div class="logo">🛡️</div>
            <div class="brand-text">
              <div class="brand-title">Phishing URL Detector</div>
              <div class="brand-sub">Scan links before you click</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def clear_scan_result():
    st.session_state.scan_has_result = False
    st.session_state.scan_result = {}

def goto_page(page_name: str):
    previous = st.session_state.page
    st.session_state.page = page_name

    if previous == "Scan" and page_name != "Scan":
        clear_scan_result()

    st.rerun()

def sidebar_nav():
    st.session_state.nav_choice = st.session_state.page

    def on_nav_change():
        prev = st.session_state.page
        new_page = st.session_state.nav_choice
        st.session_state.page = new_page
        if prev == "Scan" and new_page != "Scan":
            clear_scan_result()

    st.sidebar.markdown("### Pages")
    st.sidebar.radio(
        "Navigation",
        options=PAGES,
        key="nav_choice",
        label_visibility="collapsed",
        on_change=on_nav_change,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Project")
    st.sidebar.caption("🛡️ Phishing URL Detector")

def footer():
    st.markdown(
        """
        <div class="footer">
          © 2026 Phishing Detection Project • <span style="opacity:.85">Abdulsalam Abu Gharbieh</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def ensure_model_or_stop():
    if not MODEL_PATH.exists():
        st.error(f"Model not found at: {MODEL_PATH}\n\nRun: `python src/train.py`")
        st.stop()

def is_phishing_prediction(prediction: str) -> bool:
    p = (prediction or "").strip().lower()
    return any(k in p for k in ["phish", "malicious", "bad", "attack", "fraud"])

# Pages
def page_home():
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown(
            """
            <div class="hero">
              <h1>Welcome to Phishing URL Detector</h1>
              <p>
                Check whether a website link looks suspicious before you visit it.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("Start scanning", use_container_width=True, type="primary"):
            goto_page("Scan")

        st.write("")
        a, b, c = st.columns(3, gap="medium")
        with a:
            st.markdown(
                """
                <div class="card">
                  <h3>Fast checks</h3>
                  <div class="muted">Scan URLs in seconds using your trained model pipeline.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with b:
            st.markdown(
                """
                <div class="card">
                  <h3>Simple output</h3>
                  <div class="muted">Clear “safe” or “warning” messaging for quick decisions.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c:
            st.markdown(
                """
                <div class="card">
                  <h3>Privacy-first demo</h3>
                  <div class="muted">No accounts required. Paste a link and review the result.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        img_path = BASE_DIR / "assets" / "home.png"
        st.image(str(img_path), use_container_width=True)

def page_scan():
    ensure_model_or_stop()

    st.markdown(
        """
        <div class="hero">
          <h1>Scan a Website</h1>
          <p>Enter a URL to check whether it appears suspicious.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    has_result = bool(st.session_state.get("scan_has_result")) and bool(st.session_state.get("scan_result"))

    # Layout behaviour
    if not has_result:
        left, mid, right = st.columns([1.25, 1.6, 1.25], gap="large")
        form_container = mid
        result_container = None
    else:
        form_container, result_container = st.columns([1.05, 0.95], gap="large")

    with form_container:
        st.markdown(
            """
            <div class="card">
              <h3>URL scanner</h3>
              <div class="muted">Paste a full link or domain. It will be normalized automatically.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write("")

        st.text_input(
            "Website URL",
            placeholder="e.g., https://example.com/login",
            label_visibility="collapsed",
            key="scan_url",
        )

        scan_clicked = st.button("Scan now", use_container_width=True, type="primary")

        if scan_clicked:
            url = normalize_url(st.session_state.scan_url)

            if not url:
                st.error("Error: URL is empty.")
                st.stop()

            try:
                features_df = extract_features(url)
                model = load_model()
                prediction = model.predict(features_df)[0]

                probabilities = model.predict_proba(features_df)[0]
                class_to_prob = {cls: float(prob) for cls, prob in zip(model.classes_, probabilities)}
                confidence = class_to_prob[prediction]

            except Exception as e:
                st.exception(e)
                st.stop()

            phishing = is_phishing_prediction(str(prediction))

            st.session_state.scan_has_result = True
            st.session_state.scan_result = {
                "url": url,
                "prediction": str(prediction),
                "phishing": phishing,
                "confidence": float(confidence),
            }
            st.rerun()

    if has_result:
        res = st.session_state.scan_result

        with result_container:
            st.markdown("#### Scan result")

            if res["phishing"]:
                st.markdown(
                    '<span class="badge warn">⚠️ Warning: potential phishing detected</span>',
                    unsafe_allow_html=True,
                )
                st.caption(
                    "This website appears suspicious. Avoid entering personal information and verify the domain carefully."
                )
            else:
                st.markdown(
                    '<span class="badge safe">✅ Result: no strong phishing indicators</span>',
                    unsafe_allow_html=True,
                )
                st.caption("The model did not flag strong phishing indicators. Still use caution with unfamiliar links.")

            st.markdown("**Scanned URL:**")
            st.code(res["url"], language="text")

            st.markdown("**Prediction:**")
            st.code(res["prediction"].upper(), language="text")

            st.markdown("**Confidence:**")

            confidence_pct = int(res["confidence"] * 100)
            bar_color = "#e55353" if res["phishing"] else "#2ecc71"

            st.markdown(
                f"""
                <div style="
                    background:rgba(243,244,246,0.6);
                    border-radius:10px;
                    padding:14px 16px;
                    margin-bottom:18px;
                    border:1px solid rgba(0,0,0,0.05);
                ">
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="flex:1; background:#e5e7eb; border-radius:8px; height:14px;">
                            <div style="
                                width:{confidence_pct}%;
                                background:{bar_color};
                                height:14px;
                                border-radius:8px;
                                transition:width 0.3s ease;">
                            </div>
                        </div>
                        <div style="
                            min-width:45px;
                            font-size:1rem;
                            font-weight:600;
                            color:#374151;">
                            {confidence_pct}%
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="card">
          <h3>Quick safety reminders</h3>
          <div class="muted">
            • Watch for lookalike domains (misspellings or extra words).<br/>
            • Be careful with shortened links or random subdomains.<br/>
            • If a site asks for credentials unexpectedly, stop and verify using the official website.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def page_safety_guide():
    st.markdown(
        """
        <div class="hero">
          <h1>How to protect yourself from phishing</h1>
          <p>
            Phishing is a common online threat. Attackers try to trick people into revealing passwords,
            payment details, or account access by pretending to be a trusted service.
            Use the steps below to reduce your risk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            """
            <div class="card">
              <h3>1) Be careful with emails</h3>
              <div class="muted">
                Don’t click unknown links or open unexpected attachments.
                If an email claims to be urgent, verify via official channels.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card">
              <h3>2) Check URLs closely</h3>
              <div class="muted">
                Look for subtle misspellings, odd subdomains, or unusual endings.
                When unsure, type the site address manually instead of clicking.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            """
            <div class="card">
              <h3>3) Use two-factor authentication</h3>
              <div class="muted">
                2FA adds protection even if a password is compromised.
                Prefer authenticator apps when available.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    c4, c5, c6 = st.columns(3, gap="medium")
    with c4:
        st.markdown(
            """
            <div class="card">
              <h3>4) Avoid suspicious downloads</h3>
              <div class="muted">
                Don’t install software from pop-ups or unfamiliar sites.
                Keep your browser and operating system updated.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            """
            <div class="card">
              <h3>5) Use safety tools</h3>
              <div class="muted">
                Scan links before opening them, especially when messages feel urgent or unusual.
                Combine scanning with browser protection features for better coverage.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c6:
        st.markdown(
            """
            <div class="card">
              <h3>6) Report suspicious links</h3>
              <div class="muted">
                If you receive a suspicious email or message, report it to your organisation’s 
                IT/security team. Early reporting helps prevent others from falling victim.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def page_about():
    st.write("")

    left, right = st.columns([1.2, 0.8], gap="large")

    with left:

        # About this tool
        st.markdown(
            """
            <div class="card">
              <h1>About this tool</h1>
              <div class="muted">
                This project checks URLs before you visit them. It uses a trained machine-learning pipeline
                to identify suspicious patterns associated with phishing.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        # What it does
        st.markdown(
            """
            <div class="card">
              <h3>What it does</h3>
              <div class="muted">
                <ul>
                    <li>Accepts a URL input and normalises the link</li>
                    <li>Extracts structural URL-based features used during model training</li>
                    <li>Passes the extracted features to the trained Random Forest model</li>
                    <li>Displays a clear prediction indicating whether the URL is legitimate or phishing</li>
                </ul>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        # How the model works
        st.markdown(
            """
            <div class="card">
            <h3>How the model works</h3>

            <div class="muted">

            <p>
            This application uses a Random Forest machine learning model trained on a labelled phishing URL dataset.
            The dataset contains thousands of URLs classified as legitimate or phishing using extracted structural features.
            </p>

            <p>
            After training, the full pipeline was saved as a serialised model file 
            (<code>phishing_rf_pipeline.joblib</code>) and is loaded by the application.
            </p>

            <p>
            The Streamlit application performs real-time inference using the saved model,
            meaning the model does not retrain each time the app runs.
            </p>

            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # Model evaluation (metrics + confusion matrix)
    st.write("")

    metrics_path = BASE_DIR / "results" / "metrics.json"
    cm_path = BASE_DIR / "results" / "confusion_matrix.png"

    with st.expander("Model evaluation (10-fold cross-validation)", expanded=False):

        st.caption(
            "Results from 10-fold cross-validation on a labelled phishing URL dataset "
            "(12,573 samples, 51 features)."
        )

        if metrics_path.exists():
            import json

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

            scores = metrics.get("scores", {})

            def mean_std(metric_key: str):
                obj = scores.get(metric_key, {})
                mean = obj.get("mean", None)
                std = obj.get("std", None)
                return mean, std

            acc_mean, acc_std = mean_std("accuracy")
            prec_mean, prec_std = mean_std("precision")
            rec_mean, rec_std = mean_std("recall")
            f1_mean, f1_std = mean_std("f1")

            st.markdown(
                f"""
                <div class="card">
                  <h3>Metrics (mean ± std)</h3>
                  <div class="muted">
                    <b>Accuracy:</b> {acc_mean:.4f} ± {acc_std:.4f}<br/>
                    <b>Precision:</b> {prec_mean:.4f} ± {prec_std:.4f}<br/>
                    <b>Recall:</b> {rec_mean:.4f} ± {rec_std:.4f}<br/>
                    <b>F1-score:</b> {f1_mean:.4f} ± {f1_std:.4f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            if cm_path.exists():
                st.image(
                    str(cm_path),
                    caption="Confusion matrix",
                    use_container_width=True
                )
            else:
                st.info("Confusion matrix image not found in /results.")

        else:
            st.info("metrics.json not found in /results. Run your evaluation script to generate it.")

def page_contact():
    st.markdown(
        """
        <div class="hero">
          <h1>Contact</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown(
            """
            <div class="card">
              <h3>Contact developer</h3>
              <div class="muted">
                <b>Developer name:</b> Abdulsalam Abu Gharbieh<br/>
                <b>Email:</b> a.abugharbieh1@uni.brighton.ac.uk<br/>
                <b>Phone:</b> +44 7775 085009<br/>
                <br/>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_page():
    page = st.session_state.page

    if page == "Home":
        page_home()
    elif page == "Scan":
        page_scan()
    elif page == "Safety Guide":
        page_safety_guide()
    elif page == "About":
        page_about()
    elif page == "Contact":
        page_contact()
    else:
        st.session_state.page = "Home"
        st.session_state.nav_choice = "Home"
        clear_scan_result()
        st.rerun()

def main():
    ui_setup()
    sidebar_nav()
    header_bar()
    render_page()
    footer()

if __name__ == "__main__":
    main()