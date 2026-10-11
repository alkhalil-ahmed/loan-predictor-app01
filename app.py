"""Loan eligibility prediction dashboard."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps


st.set_page_config(
    page_title="Loan Eligibility | Risk Desk",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --ink: #f3f1eb;
        --muted: #a6a6ae;
        --line: #353944;
        --paper: #111318;
        --white: #f7fffc;
        --teal: #f2b35f;
        --teal-soft: #3b2b1a;
        --green: #72d5e4;
        --green-soft: #153842;
        --red: #f07d89;
        --red-soft: #40232c;
    }

    .stApp {
        background: radial-gradient(circle at 88% -12%, rgba(219, 145, 57, 0.14), transparent 27rem),
            radial-gradient(circle at 6% 108%, rgba(51, 122, 151, 0.2), transparent 25rem),
            linear-gradient(135deg, #111318 0%, #171a22 52%, #101217 100%);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid='stHeader'] { background: transparent; }
    [data-testid='stToolbar'] { right: 1rem; }

    [data-testid='stSidebar'] {
        background: #13201d;
        border-right: 1px solid #293b35;
    }

    [data-testid='stSidebar'] * { color: #e9f3ef; }
    [data-testid='stSidebar'] .stCaption { color: #9bb0a8; }
    [data-testid='stSidebar'] hr { border-color: #30443d; }
    [data-testid='stSidebar'] .stSuccess { background: #1a4436; border: 1px solid #2d6653; }

    .block-container { max-width: 1440px; padding: 1.15rem 2.2rem 1.3rem; }
    [data-testid='stVerticalBlock'] { gap: 0.35rem; }
    [data-testid='stForm'] [data-testid='stVerticalBlock'] { gap: 0.18rem; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
    h1 { font-size: clamp(1.8rem, 3vw, 2.65rem); letter-spacing: 0; line-height: 1.05; margin: 0.1rem 0 0.35rem; }
    h2 { font-size: 1.25rem; margin: 0.1rem 0 0.2rem; }
    h4 { margin: 0.8rem 0 0.45rem; }
    .eyebrow, .section-label, .decision-kicker {
        color: var(--teal); font-size: 0.72rem; font-weight: 700; letter-spacing: 0.12em;
    }
    .lede { color: var(--muted); font-size: 0.88rem; line-height: 1.45; max-width: 650px; margin: 0; }
    .section-label { border-bottom: 1px solid rgba(72, 223, 193, 0.18); padding-bottom: 0.55rem; margin: 0 0 0.7rem; }
    .result-label { margin-top: 0.9rem; }
    .header-bar { align-items: center; display: flex; justify-content: space-between; margin-bottom: 1rem; }
    .profile-badge { align-items: center; background: rgba(30, 32, 41, 0.88); border: 1px solid rgba(242, 179, 95, 0.5); border-radius: 999px; box-shadow: 0 0 24px rgba(242, 179, 95, 0.12), inset 0 1px rgba(255,255,255,0.09); display: flex; gap: 0.9rem; padding: 0.48rem 1.25rem 0.48rem 0.55rem; }
    .profile-badge img { border: 2px solid #f2b35f; border-radius: 50%; box-shadow: 0 0 18px rgba(242, 179, 95, 0.3); height: 3.8rem; object-fit: cover; object-position: 50% 22%; width: 3.8rem; }
    .profile-badge strong { color: var(--ink); display: block; font-size: 0.92rem; }
    .profile-badge small { color: var(--muted); display: block; font-size: 0.65rem; }
    .brand-mark { display: flex; align-items: center; gap: 0.7rem; margin: 0.25rem 0 1.3rem; }
    .brand-mark > span { color: #70d4be; font-size: 1.85rem; line-height: 1; }
    .brand-mark strong { display: block; font-family: 'Space Grotesk', sans-serif; font-size: 0.9rem; letter-spacing: 0.16em; }
    .brand-mark small { color: #94aaa1; font-size: 0.7rem; }
    .side-stat { border-bottom: 1px solid #30443d; display: flex; flex-direction: column; gap: 0.15rem; padding: 0.75rem 0; }
    .side-stat span { color: #8fa69d; font-size: 0.72rem; }
    .side-stat strong { font-size: 0.9rem; }
    [data-testid='stForm'] { background: transparent; border: 0; padding: 0; }
    [data-testid='stVerticalBlockBorderWrapper'] { background: linear-gradient(145deg, rgba(31, 34, 43, 0.92), rgba(22, 24, 31, 0.9)); border: 1px solid rgba(242, 179, 95, 0.22); border-radius: 13px; padding: 0.7rem 0.9rem 0.35rem; box-shadow: 0 18px 42px rgba(0, 0, 0, 0.28), inset 0 1px rgba(255,255,255,0.06); }
    label, [data-testid='stNumberInput'] label, [data-testid='stSelectbox'] label { color: #dfdee1 !important; font-weight: 600 !important; font-size: 0.73rem !important; }
    [data-testid='stSelectbox'], [data-testid='stNumberInput'] { margin-bottom: 0.2rem; }
    [data-baseweb='select'] > div, [data-testid='stNumberInput'] input { border: 1px solid #4b4d58; border-radius: 7px; background: rgba(18, 20, 27, 0.96); color: #f4f1eb; }
    [data-baseweb='select'] > div:hover, [data-testid='stNumberInput'] input:focus { border-color: var(--teal); box-shadow: 0 0 0 1px var(--teal), 0 0 13px rgba(242, 179, 95, 0.14); }
    [data-testid='stSelectbox'] [data-baseweb='select'] > div, [data-testid='stSelectbox'] [role='combobox'] { background: #171a22 !important; color: #f4f1eb !important; }
    [data-testid='stSelectbox'] [data-baseweb='select'] > div > div { background: #171a22 !important; color: #f4f1eb !important; }
    [data-testid='stSelectbox'] [data-baseweb='select'] > div > div > div { background: #171a22 !important; }
    [data-testid='stSelectbox'] [data-baseweb='select'] * { background-color: #171a22 !important; color: #f4f1eb !important; }
    [data-testid='stSelectbox'] [data-baseweb='select'] span { color: #f4f1eb !important; }
    [data-testid='stNumberInput'] > div, [data-testid='stNumberInput'] [data-baseweb='input'], [data-testid='stNumberInput'] [data-baseweb='input'] > div { background: #171a22 !important; border: 1px solid #4b4d58; border-radius: 7px; }
    [data-testid='stNumberInput'] [data-baseweb='input'] > div > div { background: #171a22 !important; }
    [data-testid='stNumberInput'] * { background-color: #171a22 !important; color: #f4f1eb !important; }
    [data-testid='stNumberInput'] input { background: transparent !important; border: 0; color: #f4f1eb !important; }
    [data-testid='stNumberInput'] button { background: #252833 !important; border-color: #4b4d58 !important; color: var(--teal) !important; }
    [data-baseweb='select'] svg { fill: var(--teal); }
    [data-baseweb='select'] > div, [data-testid='stNumberInput'] input { min-height: 2.15rem; }
    [data-testid='stFormSubmitButton'] button { background: linear-gradient(100deg, #e9a952, #f5ca78); border: 0; border-radius: 8px; box-shadow: 0 0 18px rgba(242, 179, 95, 0.18); color: #24170a; font-weight: 800; min-height: 2.55rem; margin-top: 0.75rem; transition: transform 160ms ease, box-shadow 160ms ease, filter 160ms ease; }
    [data-testid='stFormSubmitButton'] button:hover { box-shadow: 0 0 25px rgba(242, 179, 95, 0.34); filter: brightness(1.06); transform: translateY(-1px); }
    [data-testid='stDownloadButton'] button { background: #252833 !important; border: 1px solid #f2b35f !important; border-radius: 8px !important; color: #f5ca78 !important; font-weight: 700 !important; min-height: 2.35rem !important; }
    [data-testid='stDownloadButton'] button:hover { background: #34313a !important; box-shadow: 0 0 16px rgba(242, 179, 95, 0.2) !important; color: #ffe2a8 !important; }
    [data-testid='stDownloadButton'] { margin-top: 0.55rem; }
    .empty-state { align-items: center; background: var(--teal-soft); border: 1px dashed #acd8cd; border-radius: 8px; color: #336d63; display: flex; gap: 1rem; margin-top: 0.8rem; padding: 0.75rem 1rem; }
    .empty-icon { font-size: 1.8rem; }
    .empty-state small { color: #56877e; }
    .mini-note { color: #789e94; font-size: 0.68rem; line-height: 1.4; margin-top: 0.75rem; }
    .decision-card { align-items: center; border-radius: 10px; display: flex; gap: 1rem; justify-content: space-between; padding: 0.62rem 1.2rem; }
    .decision-card.approved { background: linear-gradient(100deg, rgba(13, 67, 49, 0.92), rgba(10, 39, 31, 0.9)); border: 1px solid rgba(67, 230, 155, 0.65); box-shadow: 0 0 28px rgba(67, 230, 155, 0.13), inset 0 1px rgba(125, 255, 193, 0.1); color: #baffd9; }
    .decision-card.rejected { background: linear-gradient(100deg, rgba(73, 19, 31, 0.94), rgba(43, 13, 22, 0.9)); border: 1px solid rgba(255, 100, 119, 0.68); box-shadow: 0 0 28px rgba(255, 70, 101, 0.14), inset 0 1px rgba(255, 157, 170, 0.1); color: #ffd0d5; }
    .decision-card h2 { color: inherit; }
    .decision-card p { color: inherit; font-size: 0.78rem; margin: 0; opacity: 0.82; }
    .decision-icon { align-items: center; border-radius: 50%; display: flex; flex: 0 0 2.25rem; font-size: 1.2rem; font-weight: 700; height: 2.25rem; justify-content: center; }
    .approved .decision-icon { background: rgba(67, 230, 155, 0.18); color: var(--green); box-shadow: 0 0 16px rgba(67, 230, 155, 0.28); }
    .rejected .decision-icon { background: rgba(255, 100, 119, 0.18); color: var(--red); box-shadow: 0 0 16px rgba(255, 100, 119, 0.28); }
    .decision-kicker { color: inherit; opacity: 0.72; }
    .confidence { border-left: 1px solid currentColor; display: flex; flex-direction: column; min-width: 8.5rem; padding-left: 1.35rem; text-align: right; }
    .confidence strong { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; }
    .confidence span { font-size: 0.72rem; opacity: 0.75; }
    .gauge-track { background: rgba(255,255,255,0.1); border-radius: 99px; height: 0.24rem; margin-top: 0.28rem; overflow: hidden; width: 7.5rem; }
    .gauge-track span { background: var(--green); border-radius: inherit; box-shadow: 0 0 8px var(--green); display: block; height: 100%; }
    .rejected .gauge-track span { background: var(--red); box-shadow: 0 0 8px var(--red); }
    .download-spacer { height: 0.9rem; }
    [data-testid='stSelectbox'] svg, [data-testid='stNumberInput'] svg { height: 1.05rem; width: 1.05rem; }
    [data-testid='stSelectbox'] button, [data-testid='stNumberInput'] button { min-height: 1.35rem; min-width: 1.35rem; }
    [data-testid='stDataFrame'] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
    @media (max-width: 700px) {
        .block-container { padding: 1rem 0.8rem 2rem; }
        [data-testid='stForm'] { padding: 0.4rem 1rem 1.3rem; }
        .decision-card { align-items: flex-start; flex-wrap: wrap; }
        .confidence { border-left: 0; border-top: 1px solid currentColor; flex-basis: 100%; padding: 0.8rem 0 0; text-align: left; }
    }
    @media print {
        @page { margin: 0.45in; size: landscape; }
        html, body, .stApp { background: #111318 !important; color: #f3f1eb !important; -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
        [data-testid='stHeader'], [data-testid='stToolbar'], [data-testid='stForm'], [data-testid='stDownloadButton'], button { display: none !important; }
        .block-container { max-width: none !important; padding: 0 !important; }
        .header-bar { margin-bottom: 0.35in; }
        .decision-card, .profile-badge { -webkit-print-color-adjust: exact !important; print-color-adjust: exact !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(
    os.getenv(
        "LOAN_MODEL_PATH",
        str(PROJECT_DIR / "logistic_regression_model.pkl"),
    )
)
SCALER_PATH = Path(
    os.getenv("LOAN_SCALER_PATH", str(PROJECT_DIR / "scaler.pkl"))
)
FEATURE_COLUMNS = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
    "Property_Area_Semiurban",
    "Property_Area_Urban",
]


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, scaler_path: str) -> tuple[Any, Any]:
    """Load and cache the serialized model artifacts."""
    return joblib.load(model_path), joblib.load(scaler_path)


def build_feature_frame(values: dict[str, Any]) -> pd.DataFrame:
    """Build the model input in the exact order used during training."""
    property_area = values["property_area"]
    row = {
        "Gender": 1 if values["gender"] == "Male" else 0,
        "Married": 1 if values["married"] == "Yes" else 0,
        "Dependents": 3 if values["dependents"] == "3+" else int(values["dependents"]),
        "Education": 0 if values["education"] == "Graduate" else 1,
        "Self_Employed": 1 if values["self_employed"] == "Yes" else 0,
        "ApplicantIncome": float(values["applicant_income"]),
        "CoapplicantIncome": float(values["coapplicant_income"]),
        "LoanAmount": float(values["loan_amount"]),
        "Loan_Amount_Term": float(values["loan_term"]),
        "Credit_History": float(values["credit_history"]),
        "Property_Area_Semiurban": int(property_area == "Semiurban"),
        "Property_Area_Urban": int(property_area == "Urban"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)


def format_currency(value: float) -> str:
    """Format a numeric value as a US dollar amount."""
    return f"${value:,.0f}"


def avatar_data_uri() -> str:
    """Return the provided avatar as an embeddable browser data URI."""
    avatar_path = PROJECT_DIR / "unnamed.jpg"
    if not avatar_path.exists():
        return ""
    encoded_image = base64.b64encode(avatar_path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded_image}"


def build_result_png(values: dict[str, Any], prediction: int, probability: float) -> bytes:
    """Create the high-resolution full-dashboard artwork."""
    canvas = Image.new("RGB", (2400, 1350), "#111318")
    draw = ImageDraw.Draw(canvas)
    font_dir = Path(r"C:\Windows\Fonts")

    def font(name: str, size: int) -> ImageFont.FreeTypeFont:
        path = font_dir / name
        if path.exists():
            return ImageFont.truetype(str(path), size)
        return ImageFont.load_default()

    title_font = font("segoeuib.ttf", 58)
    heading_font = font("segoeuib.ttf", 38)
    body_font = font("segoeui.ttf", 27)
    small_font = font("segoeui.ttf", 22)
    accent = "#f2b35f"
    approved = prediction == 1
    status_color = "#72d5e4" if approved else "#f07d89"
    status_text = "LOAN APPROVED" if approved else "LOAN NOT APPROVED"

    draw.text((100, 58), "RISK DESK  /  CREDIT DECISIONING", font=small_font, fill=accent)
    draw.text((100, 100), "Loan eligibility assessment", font=title_font, fill="#f3f1eb")
    draw.text((100, 175), "Prepared for Alkhalil  |  Logistic Regression decision support", font=body_font, fill="#a6a6ae")

    avatar_path = PROJECT_DIR / "unnamed.jpg"
    if avatar_path.exists():
        avatar = ImageOps.fit(Image.open(avatar_path).convert("RGB"), (150, 150), centering=(0.5, 0.22))
        mask = Image.new("L", (150, 150), 0)
        ImageDraw.Draw(mask).ellipse((0, 0, 149, 149), fill=255)
        canvas.paste(avatar, (2015, 50), mask)
    draw.ellipse((2013, 48, 2167, 202), outline=accent, width=4)
    draw.text((2200, 85), "Alkhalil", font=heading_font, fill="#f3f1eb")
    draw.text((2200, 135), "Application author", font=small_font, fill="#a6a6ae")

    card_y = 270
    card_height = 445
    card_width = 700
    card_positions = [(100, "01 / PERSONAL PROFILE"), (850, "02 / LOAN DETAILS"), (1600, "03 / PROPERTY & FINANCIAL INFO")]
    for x, label in card_positions:
        draw.rounded_rectangle((x, card_y, x + card_width, card_y + card_height), radius=22, fill="#1f222b", outline="#3d414c", width=2)
        draw.text((x + 32, card_y + 28), label, font=small_font, fill=accent)

    profile_details = [
        ("Gender", str(values["gender"])),
        ("Married status", str(values["married"])),
        ("Dependents", str(values["dependents"])),
        ("Education", str(values["education"])),
    ]
    loan_details = [
        ("Applicant income", f"${float(values['applicant_income']):,.0f}"),
        ("Coapplicant income", f"${float(values['coapplicant_income']):,.0f}"),
        ("Loan amount", f"${float(values['loan_amount']):,.0f}k"),
        ("Loan term", f"{float(values['loan_term']):,.0f} days"),
    ]
    property_details = [
        ("Self-employed", str(values["self_employed"])),
        ("Credit history", "Meets requirements" if values["credit_history"] else "Does not meet requirements"),
        ("Property area", str(values["property_area"])),
    ]
    for x, details in zip((100, 850, 1600), (profile_details, loan_details, property_details)):
        for index, (label, value) in enumerate(details):
            y = card_y + 92 + index * 78
            draw.text((x + 32, y), label, font=small_font, fill="#a6a6ae")
            draw.text((x + 32, y + 30), value, font=body_font, fill="#f3f1eb")

    draw.rounded_rectangle((850, 750, 1550, 815), radius=14, fill="#e9a952")
    draw.text((1200, 782), "PREDICT LOAN ELIGIBILITY", font=small_font, fill="#24170a", anchor="mm")

    card = (100, 865, 2300, 1175)
    draw.rounded_rectangle(card, radius=24, fill="#153842" if approved else "#40232c", outline=status_color, width=3)
    draw.ellipse((155, 955, 265, 1065), fill=status_color)
    draw.text((210, 1010), "OK" if approved else "!", font=heading_font, fill="#111318", anchor="mm")
    draw.text((335, 925), "RECOMMENDATION", font=small_font, fill=status_color)
    draw.text((335, 970), status_text, font=heading_font, fill="#f3f1eb")
    draw.text((335, 1035), "The application meets the model's approval threshold." if approved else "The application falls below the model's approval threshold.", font=body_font, fill="#c8d1d0")
    probability_percent = probability * 100
    confidence_label = "approval confidence" if approved else "approval probability"
    draw.text((1775, 935), f"{probability_percent:.1f}%", font=title_font, fill=status_color)
    draw.text((1780, 1010), confidence_label, font=small_font, fill="#a6a6ae")
    gauge = (1775, 1065, 2200, 1088)
    draw.rounded_rectangle(gauge, radius=10, fill="#353944")
    draw.rounded_rectangle(
        (gauge[0], gauge[1], gauge[0] + int(425 * probability), gauge[3]),
        radius=10,
        fill=status_color,
    )

    draw.text((100, 1250), "Model validation accuracy: 85.4%  |  Decision-support output  |  High-resolution export", font=small_font, fill="#70747e")
    buffer = BytesIO()
    canvas.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def render_sidebar() -> None:
    """Render project details and model metadata."""
    with st.sidebar:
        st.markdown(
            '<div class="brand-mark"><span>◈</span><div><strong>RISK DESK</strong>'
            '<small>Credit decisioning</small></div></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### Model profile")
        st.markdown(
            '<div class="side-stat"><span>Model type</span><strong>Logistic Regression</strong></div>'
            '<div class="side-stat"><span>Validation accuracy</span><strong>85.4%</strong></div>'
            '<div class="side-stat"><span>Decision output</span><strong>Binary + probability</strong></div>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### System status")
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            st.success("Model services online")
        else:
            st.error("Model artifacts unavailable")
        st.caption("Artifacts are loaded locally and cached for the session.")
        st.divider()
        st.markdown("### Ownership")
        st.caption("Loan Eligibility Intelligence\nPrepared by the Risk Analytics team")
        st.caption("Version 1.0 · October 2026")


def main() -> None:
    avatar_uri = avatar_data_uri()
    st.markdown(
        '<div class="header-bar"><div><div class="eyebrow">RISK DESK · CREDIT DECISIONING</div>'
        '<h1>Loan eligibility assessment</h1><p class="lede">A focused underwriting signal '
        "for faster, more consistent lending decisions.</p></div>"
        f'<div class="profile-badge"><img src="{avatar_uri}" alt="Author avatar">'
        '<div><strong>Alkhalil</strong><small>Application author</small></div></div></div>',
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        st.error(
            "Model artifacts could not be found. Confirm the configured paths for "
            "`logistic_regression_model.pkl` and `scaler.pkl`."
        )
        st.stop()

    with st.form("loan_application_form"):
        personal_card, loan_card, property_card = st.columns(3, gap="medium")
        with personal_card:
            with st.container(border=True):
                st.markdown('<div class="section-label">01 / Personal profile</div>', unsafe_allow_html=True)
                gender = st.selectbox("Gender", ["Male", "Female"], help="Select the applicant's reported gender.")
                married = st.selectbox("Married status", ["Yes", "No"], help="Whether the applicant is married.")
                dependents = st.selectbox("Number of dependents", ["0", "1", "2", "3+"], help="Use 3+ for three or more dependents.")
                education = st.selectbox("Education level", ["Graduate", "Not Graduate"], help="Highest reported education level.")
        with loan_card:
            with st.container(border=True):
                st.markdown('<div class="section-label">02 / Loan details</div>', unsafe_allow_html=True)
                applicant_income = st.number_input("Applicant income (USD)", min_value=0.0, value=5_000.0, step=500.0, help="Gross applicant income in US dollars.")
                coapplicant_income = st.number_input("Coapplicant income (USD)", min_value=0.0, value=1_500.0, step=500.0, help="Gross coapplicant income in US dollars. Enter 0 when not applicable.")
                loan_amount = st.number_input("Loan amount (thousands of USD)", min_value=0.0, value=150.0, step=5.0, help="Requested loan principal, expressed in thousands of US dollars.")
                loan_term = st.number_input("Loan term (days)", min_value=30.0, max_value=3_650.0, value=360.0, step=30.0, help="Requested repayment term in days.")
        with property_card:
            with st.container(border=True):
                st.markdown('<div class="section-label">03 / Property & financial info</div>', unsafe_allow_html=True)
                self_employed = st.selectbox("Self-employed status", ["No", "Yes"], help="Whether the applicant is self-employed.")
                credit_history = st.selectbox("Credit history", ["1 - Meets requirements", "0 - Does not meet requirements"], help="Select 1 when the applicant is in good standing.")
                property_area = st.selectbox("Property area", ["Urban", "Semiurban", "Rural"], help="The property's geographic classification.")
                st.markdown('<div class="mini-note">Property area is encoded into the two model features required by the scaler.</div>', unsafe_allow_html=True)

        button_left, button_center, button_right = st.columns([1, 1.3, 1])
        with button_center:
            submitted = st.form_submit_button("Predict loan eligibility  →", use_container_width=True)

    if not submitted:
        return

    values = {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "credit_history": 1.0 if credit_history.startswith("1") else 0.0,
        "property_area": property_area,
    }

    try:
        model, scaler = load_artifacts(str(MODEL_PATH), str(SCALER_PATH))
        feature_frame = build_feature_frame(values)
        scaled_features = scaler.transform(feature_frame)
        prediction = int(model.predict(scaled_features)[0])
        probability = float(model.predict_proba(scaled_features)[0, 1])
    except (ValueError, AttributeError, OSError) as error:
        st.error(f"The assessment could not be completed: {error}")
        return

    st.markdown('<div class="section-label result-label">03 / Decision signal</div>', unsafe_allow_html=True)
    probability_percent = probability * 100
    if prediction == 1:
        st.markdown(
            f'<div class="decision-card approved"><div class="decision-icon">✓</div>'
            f'<div><span class="decision-kicker">RECOMMENDATION</span>'
            f'<h2>Loan approved</h2><p>The application meets the model\'s approval threshold.</p></div>'
            f'<div class="confidence"><strong>{probability_percent:.1f}%</strong><span>approval confidence</span>'
            f'<div class="gauge-track"><span style="width: {probability_percent:.1f}%"></span></div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        risk_level = "High" if probability_percent < 30 else "Moderate"
        st.markdown(
            f'<div class="decision-card rejected"><div class="decision-icon">!</div>'
            f'<div><span class="decision-kicker">RECOMMENDATION</span>'
            f'<h2>Loan not approved</h2><p>The application falls below the model\'s approval threshold.</p></div>'
            f'<div class="confidence"><strong>{risk_level}</strong><span>risk level · {probability_percent:.1f}% approval</span>'
            f'<div class="gauge-track"><span style="width: {probability_percent:.1f}%"></span></div></div></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()