from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "df_cleaned_with_dummies.xlsx"
MATURITY_IMAGE = BASE_DIR / "MM lean 4.0.png"

st.set_page_config(page_title="Lean 4.0 Studio", layout="wide")


DIMENSION_MAP: Dict[str, List[str]] = {
    "Leadership": [
        "Leadership - Engagement Lean ",
        "Leadership - Engagement DT",
        "Leadership - StratÃ©gie ",
        "Leadership - Communication",
    ],
    "Supply Chain": [
        "Supply Chain - Collaboration inter-organisationnelle",
        "Supply Chain - TraÃ§abilitÃ©",
        "Supply Chain - Impact sur les employÃ©es",
    ],
    "OpÃ©rations": [
        "OpÃ©rations - Standardisation des processus",
        "OpÃ©rations - Juste-Ã -temps (JAT)",
        "OpÃ©rations - Gestion des rÃ©sistances",
    ],
    "Technologies": [
        "Technologies - ConnectivitÃ© et gestion des donnÃ©es",
        "Technologies - Automatisation",
        "Technologies - Pilotage du changement",
    ],
    "Organisation Apprenante": [
        "Organisation apprenante  - Formation et dÃ©veloppement des compÃ©tences",
        "Organisation apprenante  - Collaboration et Partage des Connaissances",
        "Organisation apprenante  - FlexibilitÃ© organisationnelle",
    ],
}

CRONBACH_DATA = {
    "Leadership": {
        "alpha": 0.931,
        "items": {
            "Leadership - Communication": 0.992,
            "Leadership - Engagement Lean": 0.926,
            "Leadership - StratÃ©gie": 0.901,
            "Leadership - Engagement DT": 0.868,
        },
    },
    "Supply Chain": {
        "alpha": 0.863,
        "items": {
            "Supply Chain - Impact sur les employÃ©es": 0.925,
            "Supply Chain - TraÃ§abilitÃ©": 0.826,
            "Supply Chain - Collaboration inter-organisationnelle": 0.722,
        },
    },
    "Operations": {
        "alpha": 0.867,
        "items": {
            "OpÃ©rations - Juste-Ã -temps (JAT)": 0.931,
            "OpÃ©rations - Standardisation des processus": 0.831,
            "OpÃ©rations - Gestion des rÃ©sistances": 0.754,
        },
    },
    "Technologies": {
        "alpha": 0.888,
        "items": {
            "Technologies - ConnectivitÃ© et gestion des donnÃ©es": 0.904,
            "Technologies - Automatisation": 0.881,
            "Technologies - Pilotage du changement": 0.781,
        },
    },
    "Organisation Apprenante": {
        "alpha": 0.854,
        "items": {
            "Organisation apprenante  - Formation et dÃ©veloppement des compÃ©tences": 0.876,
            "Organisation apprenante  - Collaboration et Partage des Connaissances": 0.799,
            "Organisation apprenante  - FlexibilitÃ© organisationnelle": 0.763,
        },
    },
}

LEAN_SUPPORT = {
    "Juste Ã  temps (JAT)": "Robots autonomes, WMS, RFID",
    "Takt Time": "Big Data & Analytics, SystÃ¨mes cyber-physiques, ERP, WMS",
    "Heijunka": "WMS, MES",
    "MÃ©thode TPM / TRS": "MES, RFID",
    "Poka Yoke": "Simulation, Robots autonomes, ERP",
    "Kaizen": "MES, RFID, Big Data & Analytics, Fabrication additive (Impression 3D)",
    "Kanban": "Fabrication additive (Impression 3D)",
    "Value Stream Mapping (VSM)": "SystÃ¨mes cyber-physiques, RFID, WMS",
    "QRQC": "Intelligence artificielle",
}

LEAN_DISPLAY_NAMES = {
    "Lean_QRQC": "QRQC",
    "Lean_DDMRP/ hoshin kanri": "DDMRP / Hoshin Kanri",
    "Lean_5S": "5S",
    "Lean_Heijunka": "Heijunka",
    "Lean_Maki-Gami/Hoshinâ€¦etc": "Maki-Gami / Hoshin",
    "Lean_Value Stream Mapping (VSM)": "Value Stream Mapping (VSM)",
    "Lean_Kaizen": "Kaizen",
    "Lean_DDMRP": "DDMRP",
    "Lean_MÃ©thode TPM / TRS": "MÃ©thode TPM / TRS",
    "Lean_Kata": "Kata",
    "Lean_Just in time": "Juste Ã  temps (JAT)",
    "Lean_QRAP": "QRAP",
    "Lean_TPM / TRS method": "TPM / TRS",
    "Lean_6 sigma": "6 Sigma",
    "Lean_Poka Yoke": "Poka Yoke",
    "Lean_Takt Time": "Takt Time",
    "Lean_Kanban": "Kanban",
    "Lean_GEMBA": "Gemba",
}

SCENARIO_TEXT = {
    "tech_lag": {
        "title": "Scenario 1: Technology lag",
        "icon": "ðŸ”§",
        "body": "The organizational maturity is stronger than the technology adoption. Prioritize the missing Lean tools and Industry 4.0 enablers used by the next maturity cluster.",
        "recommendations": [
            "Prioritize the technologies and Lean methods already used in the target maturity cluster but still absent in the company.",
            "Use the decision-tree drivers as the first implementation levers toward a stronger maturity level.",
            "Start with the technology roadmap, then consolidate the organizational maturity roadmap.",
        ],
    },
    "org_lag": {
        "title": "Scenario 2: Organizational lag",
        "icon": "âš¡",
        "body": "Technology adoption is ahead of organizational readiness. Focus on process discipline, leadership, and learning-system gaps before scaling more tools.",
        "recommendations": [
            "Prioritize the Lean 4.0 sub-dimensions with the largest negative gap against the target cluster.",
            "Concentrate on leadership, operating routines, and organizational learning before adding complexity.",
            "Start with the organizational maturity roadmap, then phase the technology roadmap back in.",
        ],
    },
    "aligned": {
        "title": "Scenario 3: Strategic alignment",
        "icon": "ðŸš€",
        "body": "Organizational maturity and technology adoption are aligned. Continue with balanced improvements and target the highest-impact decision-tree drivers.",
        "recommendations": [
            "Maintain a balanced improvement rhythm between maturity dimensions and technology adoption.",
            "Use the most influential decision-tree drivers to identify the next strategic acceleration points.",
            "Advance both the technology and maturity roadmaps in a coordinated way.",
        ],
    },
}

SIZE_OPTIONS = [
    "TPE / Small company",
    "PME / Medium company",
    "Grande entreprise / Large company",
]


def render_hero() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.08), transparent 20%),
                radial-gradient(circle at top right, rgba(37, 99, 235, 0.10), transparent 24%),
                linear-gradient(180deg, #f6fbff 0%, #f4faf7 42%, #f8fbfd 100%);
        }
        .main .block-container {
            max-width: none !important;
            width: 100% !important;
            min-width: 100% !important;
            padding-top: 1rem;
            padding-left: 2.2rem;
            padding-right: 2.2rem;
            padding-bottom: 2rem;
        }
        section.main > div {
            max-width: 100% !important;
        }
        [data-testid="stAppViewContainer"] > .main,
        [data-testid="stAppViewContainer"] > .main > div,
        [data-testid="stAppViewBlockContainer"],
        .stMainBlockContainer {
            max-width: 100% !important;
            width: 100% !important;
        }
        [data-testid="stAppViewContainer"] .main {
            width: 100%;
            max-width: 100%;
        }
        [data-testid="stHorizontalBlock"] {
            align-items: stretch;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #0f2f43;
            letter-spacing: -0.02em;
        }
        .stMarkdown h2 {
            font-weight: 800;
        }
        .stMarkdown h3 {
            font-weight: 700;
            color: #174b63;
        }
        .stCaption {
            color: #5d7280 !important;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f8fbfc 0%, #eef4f6 100%);
            border-right: 1px solid rgba(18, 51, 43, 0.08);
        }
        [data-testid="stSidebar"] * {
            color: #17312a;
        }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stMultiSelect label,
        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stFileUploader label,
        [data-testid="stSidebar"] .stCheckbox label {
            color: #4f6760 !important;
            font-weight: 600;
        }
        [data-testid="stSidebar"] .stSelectbox > div > div,
        [data-testid="stSidebar"] .stMultiSelect > div > div,
        [data-testid="stSidebar"] .stFileUploader > div {
            background: rgba(255,255,255,0.95);
            border-radius: 14px;
            border: 1px solid rgba(18, 51, 43, 0.08);
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #17312a;
        }
        .sidebar-card {
            background: rgba(255,255,255,0.9);
            border: 1px solid rgba(18, 51, 43, 0.08);
            border-radius: 18px;
            padding: 0.9rem 0.95rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 10px 24px rgba(18, 51, 43, 0.04);
        }
        .sidebar-card strong {
            display: block;
            margin-bottom: 0.2rem;
            color: #15312b;
        }
        .sidebar-card span {
            color: #61746e;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .hero-box {
            padding: 2rem 2.1rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at top right, rgba(42, 157, 143, 0.18), transparent 28%),
                radial-gradient(circle at bottom left, rgba(233, 196, 106, 0.18), transparent 24%),
                linear-gradient(135deg, #f7fbff 0%, #eef9f4 52%, #fff8ef 100%);
            border: 1px solid rgba(16, 110, 95, 0.12);
            margin-bottom: 1.2rem;
            box-shadow: 0 24px 50px rgba(16, 24, 40, 0.08);
        }
        .hero-title {
            font-size: 2.55rem;
            font-weight: 900;
            color: #12332b;
            background: linear-gradient(135deg, #0f766e 0%, #1d4ed8 55%, #0f172a 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }
        .hero-copy {
            color: #425f6f;
            font-size: 1.06rem;
            line-height: 1.7;
            margin-bottom: 0;
            max-width: 72rem;
        }
        .info-card {
            border: 1px solid rgba(18, 51, 43, 0.08);
            border-radius: 20px;
            padding: 1rem 1.1rem;
            background: linear-gradient(180deg, #ffffff 0%, #f9fbfb 100%);
            min-height: 118px;
            box-shadow: 0 14px 30px rgba(18, 51, 43, 0.06);
        }
        .card-label {
            color: #6a7f79;
            text-transform: uppercase;
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }
        .card-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #14332d;
        }
        .workflow-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1.1rem;
        }
        .workflow-step {
            background: rgba(255,255,255,0.74);
            border: 1px solid rgba(18, 51, 43, 0.08);
            border-radius: 18px;
            padding: 0.95rem 1rem;
        }
        .workflow-step strong {
            display: block;
            color: #14332d;
            margin-bottom: 0.3rem;
        }
        .workflow-step span {
            color: #5d6f69;
            font-size: 0.9rem;
            line-height: 1.5;
        }
        .section-shell {
            background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(247,250,249,0.92));
            border: 1px solid rgba(18, 51, 43, 0.08);
            border-radius: 24px;
            padding: 1.2rem 1.25rem;
            box-shadow: 0 16px 34px rgba(18, 51, 43, 0.06);
            margin-bottom: 1rem;
        }
        .section-kicker {
            color: #17746c;
            text-transform: uppercase;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
            font-weight: 800;
        }
        .section-title {
            color: #16332b;
            font-size: 1.52rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }
        .section-copy {
            color: #526673;
            line-height: 1.65;
            margin-bottom: 0;
        }
        .roadmap-card {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            border: 1px solid rgba(15, 23, 42, 0.08);
            box-shadow: 0 16px 34px rgba(15, 23, 42, 0.07);
            margin-bottom: 0.9rem;
            background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(246,250,255,0.96));
        }
        .roadmap-title {
            font-size: 1.08rem;
            font-weight: 800;
            color: #14324d;
            margin-bottom: 0.2rem;
        }
        .roadmap-copy {
            color: #5f7382;
            font-size: 0.95rem;
            line-height: 1.55;
            margin: 0;
        }
        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
            background: transparent;
        }
        button[data-baseweb="tab"] {
            border-radius: 999px;
            background: #edf4f2;
            border: 1px solid rgba(18, 51, 43, 0.08);
            padding: 0.45rem 0.95rem;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #0d9488, #1d4ed8);
            color: white !important;
            border-color: transparent;
        }
        </style>
        <div class="hero-box">
            <div class="hero-title">Lean 4.0 Intelligence Studio</div>
            <p class="hero-copy">
                A business-intelligence platform for Lean 4.0 maturity diagnosis, benchmarking, and transformation planning.
                The analytics stay intact, while the experience is reframed for executive review, client-facing diagnosis,
                and roadmap generation that companies can understand and act on.
            </p>
            <div class="workflow-grid">
                <div class="workflow-step"><strong>01. Intake</strong><span>Load your dataset and frame the diagnosis scope.</span></div>
                <div class="workflow-step"><strong>02. Cluster</strong><span>Identify maturity groups and cluster strength.</span></div>
                <div class="workflow-step"><strong>03. Explain</strong><span>Read PCA, radar, and heatmap views clearly.</span></div>
                <div class="workflow-step"><strong>04. Predict</strong><span>Use the decision tree to interpret maturity drivers.</span></div>
                <div class="workflow-step"><strong>05. Apply</strong><span>Assess a company from the dataset or a new input.</span></div>
                <div class="workflow-step"><strong>06. Decide</strong><span>Generate the target-cluster roadmap and priorities.</span></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_intro(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_dataset(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_excel(uploaded_file)
    return pd.read_excel(DEFAULT_DATASET)


@st.cache_data(show_spinner=False)
def load_image() -> Image.Image:
    return Image.open(MATURITY_IMAGE)


def build_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <strong>Configuration Studio</strong>
            <span>Keep the sidebar focused on scope and model settings. Company diagnosis happens in the Application module.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_features: List[str] = []
    dimension_scope = st.sidebar.multiselect(
        "1. Dimensions to include",
        list(DIMENSION_MAP.keys()),
        default=list(DIMENSION_MAP.keys()),
        help="Choose the maturity dimensions that should be part of the diagnosis scope.",
    )
    fine_tune = st.sidebar.checkbox("Fine tune sub-dimensions", value=False, help="Turn on only if you want to manually remove specific questions inside a dimension.")

    for dimension in dimension_scope:
        sub_dims = DIMENSION_MAP[dimension]
        if fine_tune:
            with st.sidebar.expander(f"{dimension} questions", expanded=False):
                selected = st.multiselect(
                    f"Active sub-dimensions for {dimension}",
                    sub_dims,
                    default=sub_dims,
                    key=f"features_{dimension}",
                )
        else:
            selected = sub_dims
        selected_features.extend(selected)

    radar_dimensions = st.sidebar.multiselect(
        "### 2. Radar profile dimensions",
        dimension_scope or list(DIMENSION_MAP.keys()),
        default=dimension_scope or list(DIMENSION_MAP.keys()),
    )
    radar_features: List[str] = []
    for dimension in radar_dimensions:
        radar_features.extend(DIMENSION_MAP[dimension])

    st.sidebar.markdown("### 3. Clustering settings")
    k_range = st.sidebar.slider("Cluster search range", 2, 10, (2, 6), help="The app evaluates elbow and silhouette scores across this range before applying the operational cluster choice.")
    k_values = list(range(k_range[0], k_range[1] + 1))
    default_k = 3 if 3 in k_values else k_values[0]
    final_k = st.sidebar.select_slider("Operational number of clusters", options=k_values, value=default_k, help="Select the cluster structure you want to use throughout the app.")

    company_options = df.index.tolist()
    default_company = 4 if len(company_options) > 4 else 0

    return {
        "selected_features": selected_features,
        "radar_features": radar_features,
        "k_range": k_range,
        "final_k": final_k,
        "default_company": company_options[default_company],
    }


def prepare_cluster_inputs(df: pd.DataFrame, selected_features: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    available_features = [col for col in selected_features if col in df.columns]
    feature_frame = df[available_features].dropna().copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_frame)
    return feature_frame, scaled


def compute_cluster_metrics(scaled_features: np.ndarray, k_range: Tuple[int, int]) -> Tuple[List[int], List[float], List[float]]:
    ks = list(range(k_range[0], k_range[1] + 1))
    inertia, silhouettes = [], []
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_features)
        inertia.append(model.inertia_)
        silhouettes.append(silhouette_score(scaled_features, labels))
    return ks, inertia, silhouettes


def rank_cluster_labels(cluster_means: pd.DataFrame) -> Dict[int, str]:
    ranking = cluster_means.mean(axis=1).sort_values().index.tolist()
    labels = ["Niveau Initial", "Niveau IntÃ©grÃ©", "Niveau AvancÃ©"]
    mapping: Dict[int, str] = {}
    for idx, cluster_id in enumerate(ranking):
        mapping[int(cluster_id)] = labels[min(idx, len(labels) - 1)]
    return mapping


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()

    merge_pairs = [
        ("Lean_MÃ©thode TPM / TRS", "Lean_TPM / TRS method"),
        ("Tech_RÃ©alitÃ© augmentÃ©e", "Tech_Augmented reality"),
        ("Tech_SystÃ¨mes cyber physiques", "Tech_Cyber â€‹â€‹physical systems"),
        ("Tech_Intelligence artificielle", "Tech_Artificial intelligence"),
        ("Tech_Robots autonomes", "Tech_Autonomous robots"),
    ]

    for primary, alias in merge_pairs:
        if primary in normalized.columns and alias in normalized.columns:
            normalized[primary] = normalized[primary].fillna(0).astype(int) | normalized[alias].fillna(0).astype(int)
            normalized.drop(columns=[alias], inplace=True)

    if {"Lean_DDMRP/ hoshin kanri", "Lean_DDMRP", "Lean_Maki-Gami/Hoshinâ€¦etc"}.issubset(normalized.columns):
        normalized["Lean_DDMRP/ hoshin kanri"] = (
            normalized["Lean_DDMRP/ hoshin kanri"].fillna(0).astype(int)
            | normalized["Lean_DDMRP"].fillna(0).astype(int)
            | normalized["Lean_Maki-Gami/Hoshinâ€¦etc"].fillna(0).astype(int)
        )
        normalized.drop(columns=["Lean_DDMRP", "Lean_Maki-Gami/Hoshinâ€¦etc"], inplace=True)

    jat_column = "OpÃ©rations - Juste-Ã -temps (JAT)"
    if jat_column in normalized.columns:
        normalized["Lean_Juste Ã  temps"] = normalized[jat_column].apply(lambda x: 1 if x in [4, 5] else 0)
        if "Lean_Just in time" in normalized.columns:
            normalized["Lean_Just in time"] = (
                normalized["Lean_Juste Ã  temps"].fillna(0).astype(int)
                | normalized["Lean_Just in time"].fillna(0).astype(int)
            )
    return normalized


def train_decision_tree(df: pd.DataFrame, target_col: str) -> Tuple[DecisionTreeClassifier, pd.DataFrame]:
    survey_columns = sum(DIMENSION_MAP.values(), [])
    removable = [
        "Indicateurs suivis",
        "Zone investissement principale",
        "Typologie de production",
        "Type de flux",
        "Pays ",
        "Cluster Label",
        "cluster",
        "Niveau de maturitÃ© Lean 4.0",
        "Taille entreprise ",
        "Secteur industriel",
        "MÃ©thodes Lean ",
        "Technologies industrie 4.0",
        "Cluster",
        "Feature_Cluster",
    ] + survey_columns

    removable += [col for col in df.columns if col.startswith("Secteur") or col.startswith("taille")]

    X = df.drop(columns=[target_col] + removable, errors="ignore")
    X = X.select_dtypes(include=[np.number]).fillna(0)
    y = df[target_col]
    valid_idx = y.dropna().index
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=46, stratify=y)
    clf = DecisionTreeClassifier(random_state=10, min_samples_leaf=3, max_depth=5)
    clf.fit(X_train, y_train)
    return clf, X


def company_dimension_table(entreprise: pd.Series, selected_features: List[str]) -> pd.DataFrame:
    rows = []
    for dimension, sub_dims in DIMENSION_MAP.items():
        for sub_dim in sub_dims:
            if sub_dim in selected_features:
                rows.append(
                    {
                        "Dimension": dimension,
                        "Sous-dimension": sub_dim.strip(),
                        "Score": round(float(entreprise.get(sub_dim, np.nan)), 2),
                    }
                )
    return pd.DataFrame(rows)


def build_manual_company_input(df_reference: pd.DataFrame) -> pd.Series:
    render_section_intro(
        "New Intake",
        "Capture a new company assessment",
        "Use the external Google Form or complete the in-app questionnaire. This keeps the same analytics pipeline while making intake easier for future client companies.",
    )

    link_col_1, link_col_2 = st.columns(2)
    with link_col_1:
        st.link_button("Open questionnaire editor", "https://docs.google.com/forms/d/18q1_-kOGChcj4DbGp7onkGYWFRK9_EW382yCCRSH4U8/edit", use_container_width=True)
    with link_col_2:
        st.link_button("Open respondent form", "https://forms.gle/Uc7689Y6Y45qpiTo7", use_container_width=True)

    with st.form("manual_company_form", clear_on_submit=False):
        meta_1, meta_2, meta_3 = st.columns(3)
        company_name = meta_1.text_input("Company name", value="New client company")
        sector_candidates = sorted([str(val) for val in df_reference.get("Secteur industriel", pd.Series(dtype=object)).dropna().unique().tolist()])
        if "Other" not in sector_candidates:
            sector_candidates.append("Other")
        company_sector_choice = meta_2.selectbox("Industrial sector", sector_candidates, index=0 if sector_candidates else None)
        company_size = meta_3.selectbox("Company size", SIZE_OPTIONS, index=1)
        custom_sector = st.text_input("Custom sector label", value="", placeholder="Fill only if you selected Other")

        manual_scores: Dict[str, float] = {}
        dim_cols = st.columns(len(DIMENSION_MAP))
        for col, (dimension, sub_dims) in zip(dim_cols, DIMENSION_MAP.items()):
            with col:
                st.markdown(f"**{dimension}**")
                for sub_dim in sub_dims:
                    manual_scores[sub_dim] = st.slider(
                        sub_dim.strip(),
                        min_value=1,
                        max_value=5,
                        value=3,
                        step=1,
                        key=f"manual_{sub_dim}",
                    )

        lean_cols = [col for col in df_reference.columns if col.startswith("Lean_")]
        tech_cols = [col for col in df_reference.columns if col.startswith("Tech_")]
        lean_options = {LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")): col for col in lean_cols}
        tech_options = {col.replace("Tech_", ""): col for col in tech_cols}

        selected_lean = st.multiselect(
            "Lean methods already adopted",
            options=sorted(lean_options.keys()),
        )
        selected_tech = st.multiselect(
            "Industry 4.0 technologies already adopted",
            options=sorted(tech_options.keys()),
        )

        submitted = st.form_submit_button("Analyze new company", use_container_width=True)

    if not submitted:
        st.info("Complete the questionnaire and click 'Analyze new company' to generate the full diagnosis.")
        return pd.Series(dtype=object)

    manual_company = pd.Series(0, index=df_reference.columns, dtype=object)
    manual_company["Nom entreprise"] = company_name
    manual_company["Secteur industriel"] = custom_sector if company_sector_choice == "Other" and custom_sector.strip() else company_sector_choice
    manual_company["Taille entreprise "] = company_size

    for col, value in manual_scores.items():
        manual_company[col] = value

    for label in selected_lean:
        manual_company[lean_options[label]] = 1
    for label in selected_tech:
        manual_company[tech_options[label]] = 1

    return manual_company


def determine_scenario(cluster_label: str, predicted_dt: str) -> str:
    order = {"Niveau Initial": 1, "Niveau IntÃ©grÃ©": 2, "Niveau AvancÃ©": 3}
    cluster_rank = order.get(cluster_label, 0)
    dt_rank = order.get(predicted_dt, 0)
    if dt_rank < cluster_rank:
        return "tech_lag"
    if dt_rank > cluster_rank:
        return "org_lag"
    return "aligned"


def priority_from_gap(value: float) -> str:
    if value <= -1.0:
        return "High"
    if value <= -0.5:
        return "Medium"
    return "Low"


def priority_from_adoption(value: float) -> str:
    if value >= 0.7:
        return "High"
    if value >= 0.4:
        return "Medium"
    return "Low"


def roadmap_level_from_gap(value: float) -> str:
    if value <= -1.2:
        return "Critical"
    if value <= -0.8:
        return "Priority 1"
    if value <= -0.4:
        return "Priority 2"
    return "Monitor"


def roadmap_level_from_adoption(value: float) -> str:
    if value >= 0.75:
        return "Immediate"
    if value >= 0.5:
        return "Next Wave"
    return "Build Case"


def style_priority_cell(value: object) -> str:
    palette = {
        "Critical": "background-color:#fee2e2;color:#991b1b;font-weight:800;border-radius:8px;",
        "Priority 1": "background-color:#ffedd5;color:#9a3412;font-weight:800;border-radius:8px;",
        "Priority 2": "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:8px;",
        "Monitor": "background-color:#dcfce7;color:#166534;font-weight:700;border-radius:8px;",
        "Immediate": "background-color:#dbeafe;color:#1d4ed8;font-weight:800;border-radius:8px;",
        "Next Wave": "background-color:#e0f2fe;color:#075985;font-weight:700;border-radius:8px;",
        "Build Case": "background-color:#ecfeff;color:#155e75;font-weight:700;border-radius:8px;",
        "High": "background-color:#fee2e2;color:#991b1b;font-weight:800;border-radius:8px;",
        "Medium": "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:8px;",
        "Low": "background-color:#dcfce7;color:#166534;font-weight:700;border-radius:8px;",
    }
    return palette.get(str(value), "")


def build_roadmap_styler(df: pd.DataFrame, gradient_column: str, gradient_cmap: str, priority_column: str) -> pd.io.formats.style.Styler:
    return (
        df.style
        .background_gradient(subset=[gradient_column], cmap=gradient_cmap)
        .map(style_priority_cell, subset=[priority_column])
        .set_properties(**{"text-align": "center"})
        .set_table_styles([{"selector": "th", "props": [("text-align", "center"), ("font-weight", "700")]}])
    )


def pretty_tree_label(name: str) -> str:
    clean = (
        name.replace("Lean_", "Lean: ")
        .replace("Tech_", "Tech: ")
        .replace("Organisation apprenante", "Org. apprenante")
        .replace("Technologies - ", "Tech - ")
        .replace("Supply Chain - ", "SC - ")
        .replace("Leadership - ", "Lead - ")
        .replace("OpÃƒÂ©rations - ", "Ops - ")
    )
    return "\\n".join(textwrap.wrap(clean, width=22)) or clean


def build_dimension_comparison(entreprise: pd.Series, cluster_target: pd.Series, selected_features: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    dimension_groups = {
        "Leadership": [col for col in selected_features if "Leadership" in col],
        "Operations": [col for col in selected_features if "OpÃ©rations" in col or "Operations" in col],
        "Learning": [col for col in selected_features if "Organisation apprenante" in col],
        "Technology": [col for col in selected_features if "Technologies" in col],
        "Supply Chain": [col for col in selected_features if "Supply Chain" in col],
    }

    company_scores: Dict[str, float] = {}
    target_scores: Dict[str, float] = {}
    for dimension, cols in dimension_groups.items():
        valid_cols = [col for col in cols if col in entreprise.index and col in cluster_target.index]
        if valid_cols:
            company_scores[dimension] = float(pd.to_numeric(entreprise[valid_cols], errors="coerce").mean())
            target_scores[dimension] = float(pd.to_numeric(cluster_target[valid_cols], errors="coerce").mean())
    return company_scores, target_scores


def render_overview(df: pd.DataFrame, selected_features: List[str], cluster_labels: Dict[int, str]) -> None:
    render_section_intro(
        "Executive Snapshot",
        "Portfolio overview",
        "Start with the overall size of the dataset, the breadth of the maturity model, and the maturity-group structure before moving into detailed analytics.",
    )
    dataset_col, image_col = st.columns([1.1, 0.9])
    with dataset_col:
        st.subheader("Overview")
        cols = st.columns(4)
        metrics = [
            ("Companies", f"{len(df)}"),
            ("Selected sub-dimensions", f"{len(selected_features)}"),
            ("Lean methods", f"{len([c for c in df.columns if c.startswith('Lean_')])}"),
            ("Industry 4.0 tech", f"{len([c for c in df.columns if c.startswith('Tech_')])}"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(
                    f"<div class='info-card'><div class='card-label'>{label}</div><div class='card-value'>{value}</div></div>",
                    unsafe_allow_html=True,
                )
        st.markdown("### Maturity groups")
        st.dataframe(pd.DataFrame({"Cluster": list(cluster_labels.keys()), "Label": list(cluster_labels.values())}), use_container_width=True)
    with image_col:
        st.image(load_image(), caption="ModÃ¨le de MaturitÃ© Lean 4.0", use_container_width=True)


def render_clustering_tab(ks, inertia, silhouettes, df_clustered, cluster_label_map) -> None:
    render_section_intro(
        "Analytics",
        "Clustering and reliability",
        "Validate the structure of the maturity model, inspect the elbow and silhouette behavior, and confirm the operational cluster allocation used in the rest of the app.",
    )
    st.subheader("Reliability snapshot")
    st.success("Cronbach's Alpha global for the selected maturity columns: 0.934")
    for group, values in CRONBACH_DATA.items():
        with st.expander(group):
            st.write(f"Alpha: {values['alpha']:.3f}")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Sous-dimension": list(values["items"].keys()),
                        "Alpha si supprimÃ©e": list(values["items"].values()),
                    }
                ),
                use_container_width=True,
            )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.plot(ks, inertia, marker="o")
        ax.set_title("Elbow method")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots()
        ax.plot(ks, silhouettes, marker="o")
        ax.set_title("Silhouette score")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    summary = df_clustered["cluster"].value_counts().sort_index()
    summary_df = pd.DataFrame(
        {
            "Cluster": summary.index,
            "Nombre d'entreprises": summary.values,
            "Niveau de maturitÃ© Lean 4.0": summary.index.map(cluster_label_map),
        }
    )
    st.markdown("### Cluster summary")
    st.dataframe(summary_df, use_container_width=True)


def render_pca_tab(df_clustered: pd.DataFrame, scaled_features: np.ndarray) -> None:
    render_section_intro(
        "Exploration",
        "PCA cluster map",
        "Use PCA to see how clearly the maturity groups separate in reduced-dimensional space and where overlap still exists between company profiles.",
    )
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_pca = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    df_pca["label"] = df_clustered.loc[df_pca.index, "Niveau de maturitÃ© Lean 4.0"].values
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="label", palette="Set2", ax=ax)
    ax.set_title("PCA of clusters")
    st.pyplot(fig)


def render_radar_tab(df_clustered: pd.DataFrame, radar_features: List[str]) -> None:
    render_section_intro(
        "Benchmarking",
        "Radar comparison",
        "Understand how maturity groups differ both at sub-dimension level and at dimension level. This is useful for management communication and capability storytelling.",
    )
    cluster_avg = df_clustered.groupby("Niveau de maturitÃ© Lean 4.0")[radar_features].mean().dropna(axis=1, how="any")
    if cluster_avg.empty:
        st.warning("No radar data available for the selected dimensions.")
        return

    palette = {
        "Niveau Initial": ("rgba(0, 0, 139, 1)", "rgba(0, 0, 139, 0.35)"),
        "Niveau IntÃ©grÃ©": ("rgba(255, 99, 71, 1)", "rgba(255, 99, 71, 0.25)"),
        "Niveau AvancÃ©": ("rgba(60, 179, 113, 1)", "rgba(60, 179, 113, 0.25)"),
    }

    fig = go.Figure()
    for label in cluster_avg.index:
        line_color, fill_color = palette.get(label, ("rgba(70,70,70,1)", "rgba(70,70,70,0.2)"))
        fig.add_trace(
            go.Scatterpolar(
                r=cluster_avg.loc[label].values,
                theta=cluster_avg.columns.tolist(),
                fill="toself",
                name=label,
                line=dict(color=line_color, width=3),
                fillcolor=fill_color,
            )
        )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=620)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Radar by dimension")
    dimension_scores = pd.DataFrame(index=cluster_avg.index)
    for dimension, cols in DIMENSION_MAP.items():
        valid = [col for col in cols if col in cluster_avg.columns]
        if valid:
            dimension_scores[dimension] = cluster_avg[valid].mean(axis=1)

    fig_dim = go.Figure()
    for label in dimension_scores.index:
        line_color, fill_color = palette.get(label, ("rgba(70,70,70,1)", "rgba(70,70,70,0.2)"))
        fig_dim.add_trace(
            go.Scatterpolar(
                r=dimension_scores.loc[label].values,
                theta=dimension_scores.columns.tolist(),
                fill="toself",
                name=label,
                line=dict(color=line_color, width=3),
                fillcolor=fill_color,
            )
        )
    fig_dim.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=560)
    st.plotly_chart(fig_dim, use_container_width=True)


def render_heatmaps_tab(df_clustered: pd.DataFrame, selected_features: List[str]) -> None:
    render_section_intro(
        "Adoption Patterns",
        "Heatmaps",
        "Compare maturity scores, Lean-method adoption, and Industry 4.0 technology usage across the cluster structure to identify the strongest operational patterns.",
    )
    heatmap_features = df_clustered.filter(items=selected_features)
    if heatmap_features.empty:
        st.warning("No valid maturity columns are available for the heatmap.")
        return

    avg_scores = (
        pd.concat([df_clustered[["cluster"]], heatmap_features], axis=1)
        .groupby("cluster")
        .mean(numeric_only=True)
    )
    lean_cols = [col for col in df_clustered.columns if col.startswith("Lean_")]
    tech_cols = [col for col in df_clustered.columns if col.startswith("Tech_")]
    lean_avg = df_clustered.groupby("cluster")[lean_cols].mean() if lean_cols else pd.DataFrame()
    tech_avg = df_clustered.groupby("cluster")[tech_cols].mean() if tech_cols else pd.DataFrame()

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    sns.heatmap(avg_scores.T, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=0.8, ax=axes[0])
    axes[0].set_title("Average survey scores by cluster")

    if lean_avg.empty:
        axes[1].text(0.5, 0.5, "No Lean method columns detected.", ha="center", va="center")
        axes[1].axis("off")
    else:
        sns.heatmap(lean_avg.T, cmap="Oranges", annot=True, fmt=".2f", linewidths=0.8, ax=axes[1])
        axes[1].set_title("Average Lean method usage by cluster")

    if tech_avg.empty:
        axes[2].text(0.5, 0.5, "No Industry 4.0 technology columns detected.", ha="center", va="center")
        axes[2].axis("off")
    else:
        sns.heatmap(tech_avg.T, cmap="PuRd", annot=True, fmt=".2f", linewidths=0.8, ax=axes[2])
        axes[2].set_title("Average Industry 4.0 technology usage by cluster")

    plt.tight_layout()
    st.pyplot(fig)


def render_decision_tree_tab(clf: DecisionTreeClassifier, X: pd.DataFrame) -> None:
    render_section_intro(
        "Interpretability",
        "Decision tree and key drivers",
        "Use the decision tree to explain which Lean methods and technology adoption patterns are most influential in the maturity classification logic.",
    )
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_importances = importances[importances > 0].sort_values(ascending=False).head(12)
    st.markdown("### Feature importances")
    if top_importances.empty:
        st.info("No non-zero feature importances were found for the current decision-tree configuration.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_importances.sort_values().plot(kind="barh", ax=ax, color="#1d4ed8")
        ax.set_title("Top decision-tree drivers", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Decision tree visualization")
    st.caption("Color-coded business rules rendered left-to-right for easier executive reading.")
    pretty_labels = [pretty_tree_label(col) for col in X.columns]
    dot_data = export_graphviz(
        clf,
        out_file=None,
        feature_names=pretty_labels,
        class_names=[str(c) for c in clf.classes_],
        filled=True,
        rounded=True,
        proportion=True,
        precision=2,
        impurity=False,
        rotate=True,
        special_characters=True,
    )
    dot_data = dot_data.replace(
        "digraph Tree {",
        'digraph Tree { graph [rankdir=LR, bgcolor="transparent", pad="0.35", nodesep="0.45", ranksep="1.1"]; node [shape=box, style="rounded,filled", color="#cbd5e1", fontname="Helvetica", fontsize=12, margin="0.22,0.12"]; edge [color="#64748b", penwidth=1.2];',
    )
    try:
        st.graphviz_chart(dot_data, use_container_width=True)
    except Exception:
        fig, ax = plt.subplots(figsize=(36, 18))
        plot_tree(
            clf,
            feature_names=pretty_labels,
            class_names=[str(c) for c in clf.classes_],
            filled=True,
            rounded=True,
            fontsize=10,
            impurity=False,
            proportion=True,
            precision=2,
            ax=ax,
        )
        ax.set_title("Decision tree classification", fontsize=18, fontweight="bold")
        st.pyplot(fig, use_container_width=True)

def render_application_tab(
    df_clustered: pd.DataFrame,
    scaler: StandardScaler,
    kmeans: KMeans,
    selected_features: List[str],
    cluster_label_map: Dict[int, str],
    clf: DecisionTreeClassifier,
    X: pd.DataFrame,
    selected_company,
) -> None:
    render_section_intro(
        "Application",
        "Company diagnosis workflow",
        "Run a guided diagnosis for an existing company from the dataset or a newly assessed company, then generate the target-cluster comparison and recommended roadmap.",
    )
    mode = st.radio("Application mode", ["Existing company from dataset", "New company input"], horizontal=True)

    if mode == "Existing company from dataset":
        selected_company = st.selectbox(
            "Choose the company to diagnose",
            df_clustered.index.tolist(),
            index=df_clustered.index.tolist().index(selected_company) if selected_company in df_clustered.index else 0,
        )
        entreprise = df_clustered.loc[selected_company]
        company_label = f"Dataset company #{selected_company}"
        st.markdown("### Company profile")
        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Company index", selected_company)
        info_2.metric("Sector", entreprise.get("Secteur industriel", "N/A"))
        info_3.metric("Size", entreprise.get("Taille entreprise ", "N/A"))
    else:
        entreprise = build_manual_company_input(df_clustered)
        if entreprise.empty:
            return
        company_label = entreprise.get("Nom entreprise", "Nouvelle entreprise")
        st.markdown("### New company profile")
        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Company", company_label)
        info_2.metric("Sector", entreprise.get("Secteur industriel", "N/A"))
        info_3.metric("Size", entreprise.get("Taille entreprise ", "N/A"))

    st.markdown("### Maturity scores by sub-dimension")
    st.dataframe(company_dimension_table(entreprise, selected_features), use_container_width=True)

    lean_cols = [col for col in df_clustered.columns if col.startswith("Lean_")]
    tech_cols = [col for col in df_clustered.columns if col.startswith("Tech_")]
    lean_adopted = [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_cols if entreprise.get(col, 0) == 1]
    tech_adopted = [col.replace("Tech_", "") for col in tech_cols if entreprise.get(col, 0) == 1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Lean methods already adopted")
        st.dataframe(pd.DataFrame({"Lean method": lean_adopted or ["No Lean method detected"]}), use_container_width=True)
    with col2:
        st.markdown("### Industry 4.0 technologies already adopted")
        st.dataframe(pd.DataFrame({"Technology 4.0": tech_adopted or ["No technology detected"]}), use_container_width=True)

    entreprise_scaled = scaler.transform(entreprise[selected_features].values.reshape(1, -1))
    predicted_cluster = int(kmeans.predict(entreprise_scaled)[0])
    predicted_cluster_label = cluster_label_map.get(predicted_cluster, "Inconnu")

    features_dt_new = pd.DataFrame([entreprise]).reindex(columns=X.columns, fill_value=0)
    predicted_dt = clf.predict(features_dt_new)[0]

    cluster_col, tree_col = st.columns(2)
    cluster_col.metric("Organizational maturity", predicted_cluster_label)
    tree_col.metric("Technological maturity", predicted_dt)
    st.caption(f"Analysis target: {company_label}")

    scenario_key = determine_scenario(predicted_cluster_label, predicted_dt)
    scenario = SCENARIO_TEXT[scenario_key]
    st.markdown(
        f"""
        <div class="hero-box" style="padding:1.1rem 1.2rem; margin-top:0.8rem;">
            <div class="hero-title" style="font-size:1.35rem;">{scenario['icon']} {scenario['title']}</div>
            <p class="hero-copy">{scenario['body']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("## ðŸ§­ Guide dâ€™utilisation personnalisÃ©")
    st.markdown(
        """
        1. Identification du scÃ©nario dâ€™adoption de lâ€™entreprise.
        2. Lecture des Ã©carts par rapport au cluster cible via les radars et les tableaux dâ€™Ã©cart.
        3. GÃ©nÃ©ration de deux feuilles de route complÃ©mentaires :
        - feuille de route organisationnelle Lean 4.0
        - feuille de route technologique et mÃ©thodes Lean
        4. Priorisation et exÃ©cution progressive selon le scÃ©nario dÃ©tectÃ©.
        """
    )
    st.markdown("## ðŸ” Analyse comparative et recommandations")
    for idx, recommendation in enumerate(scenario["recommendations"], start=1):
        st.write(f"{idx}. {recommendation}")

    application_features = df_clustered.filter(items=selected_features)
    selected_features = application_features.columns.tolist()
    if not selected_features:
        st.error("No valid sub-dimension columns are available for application analysis.")
        st.stop()

    cluster_means = (
        pd.concat([df_clustered[["cluster"]], application_features], axis=1)
        .groupby("cluster")
        .mean(numeric_only=True)
    )
    cluster_rank = cluster_means.mean(axis=1).sort_values().index.tolist()
    current_position = cluster_rank.index(predicted_cluster)
    next_cluster = cluster_rank[min(current_position + 1, len(cluster_rank) - 1)]

    st.markdown("### Company vs target cluster radar")
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=entreprise[selected_features].values.tolist(),
            theta=selected_features,
            fill="toself",
            name="Entreprise",
            line=dict(color="rgba(255, 0, 0, 1)", width=3),
            fillcolor="rgba(255, 0, 0, 0.25)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=cluster_means.loc[next_cluster][selected_features].values.tolist(),
            theta=selected_features,
            fill="toself",
            name="Cluster cible",
            line=dict(color="rgba(0, 0, 139, 1)", width=3),
            fillcolor="rgba(0, 0, 139, 0.2)",
        )
    )
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=650)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Company vs target cluster radar by dimension")
    company_dim_scores, target_dim_scores = build_dimension_comparison(
        entreprise,
        cluster_means.loc[next_cluster],
        selected_features,
    )
    if company_dim_scores and target_dim_scores:
        fig_dim = go.Figure()
        fig_dim.add_trace(
            go.Scatterpolar(
                r=list(company_dim_scores.values()),
                theta=list(company_dim_scores.keys()),
                fill="toself",
                name="Entreprise",
                line=dict(color="rgba(255, 0, 0, 1)", width=3),
                fillcolor="rgba(255, 0, 0, 0.25)",
            )
        )
        fig_dim.add_trace(
            go.Scatterpolar(
                r=list(target_dim_scores.values()),
                theta=list(target_dim_scores.keys()),
                fill="toself",
                name="Cluster cible",
                line=dict(color="rgba(0, 0, 139, 1)", width=3),
                fillcolor="rgba(0, 0, 139, 0.2)",
            )
        )
        fig_dim.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=560)
        st.plotly_chart(fig_dim, use_container_width=True)

    entreprise_scores = pd.to_numeric(entreprise[selected_features], errors="coerce")
    target_scores = pd.to_numeric(cluster_means.loc[next_cluster][selected_features], errors="coerce")
    gaps = (entreprise_scores - target_scores).dropna().sort_values()
    negative_gaps = gaps[gaps < 0]
    gap_df = pd.DataFrame(
        {
            "Sous-dimension": negative_gaps.index,
            "Gap": negative_gaps.round(2).values,
            "Priority": [priority_from_gap(x) for x in negative_gaps.values],
            "Graduation": [roadmap_level_from_gap(x) for x in negative_gaps.values],
        }
    )

    st.markdown("### Feuille de route personnalisee")
    st.markdown(
        """
        <div class="roadmap-card">
            <div class="roadmap-title">Transformation roadmap</div>
            <p class="roadmap-copy">This roadmap is split into two synchronized streams: organizational maturity upgrades and technology adoption priorities. Colors indicate urgency and graduation for faster executive reading.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not gap_df.empty:
        critical_count = int((gap_df["Graduation"] == "Critical").sum())
        priority_1_count = int((gap_df["Graduation"] == "Priority 1").sum())
        priority_2_count = int((gap_df["Graduation"] == "Priority 2").sum())
    else:
        critical_count = 0
        priority_1_count = 0
        priority_2_count = 0

    stat_1, stat_2, stat_3 = st.columns(3)
    stat_1.metric("Critical organizational gaps", critical_count)
    stat_2.metric("Priority 1 gaps", priority_1_count)
    stat_3.metric("Priority 2 gaps", priority_2_count)

    st.markdown("#### Organizational roadmap: negative gaps to close")
    if gap_df.empty:
        st.success("No negative maturity gap was detected against the target cluster.")
    else:
        st.dataframe(
            build_roadmap_styler(gap_df, "Gap", "YlOrRd_r", "Graduation"),
            use_container_width=True,
        )

    lean_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, lean_cols].mean()
    tech_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, tech_cols].mean()
    lean_to_adopt = lean_cluster_mean[(lean_cluster_mean > 0) & (entreprise[lean_cluster_mean.index] == 0)].sort_values(ascending=False)
    tech_to_adopt = tech_cluster_mean[(tech_cluster_mean > 0) & (entreprise[tech_cluster_mean.index] == 0)].sort_values(ascending=False)

    roadmap_col, tech_col = st.columns(2)
    with roadmap_col:
        st.markdown("#### Technological roadmap: Lean methods to adopt")
        lean_df = pd.DataFrame(
            {
                "Lean method": [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_to_adopt.index],
                "Technologies support": [LEAN_SUPPORT.get(LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")), "") for col in lean_to_adopt.index],
                "Adoption rate in target cluster": lean_to_adopt.round(2).values,
                "Priority": [priority_from_adoption(v) for v in lean_to_adopt.values],
                "Graduation": [roadmap_level_from_adoption(v) for v in lean_to_adopt.values],
            }
        )
        if lean_df.empty:
            st.info("No priority Lean method to adopt.")
        else:
            st.dataframe(
                build_roadmap_styler(lean_df, "Adoption rate in target cluster", "Blues", "Graduation"),
                use_container_width=True,
            )

    with tech_col:
        st.markdown("#### Technological roadmap: Industry 4.0 technologies to adopt")
        tech_df = pd.DataFrame(
            {
                "Technology": [col.replace("Tech_", "") for col in tech_to_adopt.index],
                "Adoption rate in target cluster": tech_to_adopt.round(2).values,
                "Priority": [priority_from_adoption(v) for v in tech_to_adopt.values],
                "Graduation": [roadmap_level_from_adoption(v) for v in tech_to_adopt.values],
            }
        )
        if tech_df.empty:
            st.info("No priority technology to adopt.")
        else:
            st.dataframe(
                build_roadmap_styler(tech_df, "Adoption rate in target cluster", "PuBu", "Graduation"),
                use_container_width=True,
            )

    executive_lines = []
    if not gap_df.empty:
        executive_lines.append(f"Top organizational priorities: {', '.join(gap_df['Sous-dimension'].head(3).tolist())}.")
    if not lean_df.empty:
        executive_lines.append(f"Top Lean adoption priorities: {', '.join(lean_df['Lean method'].head(3).tolist())}.")
    if not tech_df.empty:
        executive_lines.append(f"Top technology adoption priorities: {', '.join(tech_df['Technology'].head(3).tolist())}.")
    if executive_lines:
        st.markdown("#### Executive summary")
        for line in executive_lines:
            st.write(f"- {line}")
def main() -> None:
    render_hero()
    st.sidebar.title("Control Studio")
    st.sidebar.caption("Configure the scope of the diagnosis. The application module will handle company-level assessment separately.")
    dataset_mode = st.sidebar.radio(
        "Dataset source",
        ["Repository dataset", "Upload custom Excel"],
        index=0,
    )
    uploaded_file = None
    if dataset_mode == "Upload custom Excel":
        uploaded_file = st.sidebar.file_uploader("Upload Excel dataset", type=["xlsx"])
    df_raw = load_dataset(uploaded_file)
    sidebar = build_sidebar(df_raw)

    selected_features = sidebar["selected_features"]
    if not selected_features:
        st.warning("Select at least one sub-dimension to continue.")
        st.stop()

    radar_features = [col for col in sidebar["radar_features"] if col in df_raw.columns]
    if not radar_features:
        radar_features = [col for col in selected_features if col in df_raw.columns]

    feature_frame, scaled_features = prepare_cluster_inputs(df_raw, selected_features)
    selected_features = feature_frame.columns.tolist()
    radar_features = [col for col in radar_features if col in selected_features]
    if not radar_features:
        radar_features = selected_features.copy()
    aligned_df = df_raw.loc[feature_frame.index].copy()
    selected_company = sidebar["default_company"]
    if selected_company not in aligned_df.index:
        selected_company = aligned_df.index[0]
        st.sidebar.warning("The selected company had missing values for the active sub-dimensions. The first valid company was selected instead.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_frame)
    ks, inertia, silhouettes = compute_cluster_metrics(scaled_features, sidebar["k_range"])
    kmeans = KMeans(n_clusters=sidebar["final_k"], random_state=42, n_init=10)
    aligned_df["cluster"] = kmeans.fit_predict(scaled_features)

    selected_features = [col for col in selected_features if col in aligned_df.columns]
    if not selected_features:
        st.error("No valid sub-dimension columns are available in the active dataset after alignment.")
        st.stop()
    radar_features = [col for col in radar_features if col in selected_features]
    if not radar_features:
        radar_features = selected_features.copy()

    cluster_feature_frame = aligned_df.filter(items=selected_features)
    selected_features = cluster_feature_frame.columns.tolist()
    if not selected_features:
        st.error("No valid sub-dimension columns are available for cluster aggregation.")
        st.stop()
    cluster_means = pd.concat([aligned_df[["cluster"]], cluster_feature_frame], axis=1).groupby("cluster").mean(numeric_only=True)
    cluster_label_map = rank_cluster_labels(cluster_means)
    aligned_df["Niveau de maturitÃ© Lean 4.0"] = aligned_df["cluster"].map(cluster_label_map)

    normalized_df = normalize_columns(aligned_df.copy())
    target_col = "Niveau MaturitÃ©" if "Niveau MaturitÃ©" in normalized_df.columns else "Niveau de maturitÃ© Lean 4.0"
    clf, X = train_decision_tree(normalized_df, target_col)

    render_overview(aligned_df, selected_features, cluster_label_map)

    tabs = st.tabs(
        [
            "01 Clustering",
            "02 PCA",
            "03 Radar",
            "04 Heatmaps",
            "05 Decision Tree",
            "06 Application",
        ]
    )

    with tabs[0]:
        render_clustering_tab(ks, inertia, silhouettes, aligned_df, cluster_label_map)
    with tabs[1]:
        render_pca_tab(aligned_df, scaled_features)
    with tabs[2]:
        render_radar_tab(aligned_df, radar_features)
    with tabs[3]:
        render_heatmaps_tab(aligned_df, selected_features)
    with tabs[4]:
        render_decision_tree_tab(clf, X)
    with tabs[5]:
        render_application_tab(
            normalized_df,
            scaler,
            kmeans,
            selected_features,
            cluster_label_map,
            clf,
            X,
            selected_company,
        )


if __name__ == "__main__":
    main()



