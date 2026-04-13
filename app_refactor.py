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
        "Leadership - Stratégie ",
        "Leadership - Communication",
    ],
    "Supply Chain": [
        "Supply Chain - Collaboration inter-organisationnelle",
        "Supply Chain - Traçabilité",
        "Supply Chain - Impact sur les employées",
    ],
    "Opérations": [
        "Opérations - Standardisation des processus",
        "Opérations - Juste-à-temps (JAT)",
        "Opérations - Gestion des résistances",
    ],
    "Technologies": [
        "Technologies - Connectivité et gestion des données",
        "Technologies - Automatisation",
        "Technologies - Pilotage du changement",
    ],
    "Organisation Apprenante": [
        "Organisation apprenante  - Formation et développement des compétences",
        "Organisation apprenante  - Collaboration et Partage des Connaissances",
        "Organisation apprenante  - Flexibilité organisationnelle",
    ],
}

CRONBACH_DATA = {
    "Leadership": {
        "alpha": 0.931,
        "items": {
            "Leadership - Communication": 0.992,
            "Leadership - Engagement Lean": 0.926,
            "Leadership - Stratégie": 0.901,
            "Leadership - Engagement DT": 0.868,
        },
    },
    "Supply Chain": {
        "alpha": 0.863,
        "items": {
            "Supply Chain - Impact sur les employées": 0.925,
            "Supply Chain - Traçabilité": 0.826,
            "Supply Chain - Collaboration inter-organisationnelle": 0.722,
        },
    },
    "Operations": {
        "alpha": 0.867,
        "items": {
            "Opérations - Juste-à-temps (JAT)": 0.931,
            "Opérations - Standardisation des processus": 0.831,
            "Opérations - Gestion des résistances": 0.754,
        },
    },
    "Technologies": {
        "alpha": 0.888,
        "items": {
            "Technologies - Connectivité et gestion des données": 0.904,
            "Technologies - Automatisation": 0.881,
            "Technologies - Pilotage du changement": 0.781,
        },
    },
    "Organisation Apprenante": {
        "alpha": 0.854,
        "items": {
            "Organisation apprenante  - Formation et développement des compétences": 0.876,
            "Organisation apprenante  - Collaboration et Partage des Connaissances": 0.799,
            "Organisation apprenante  - Flexibilité organisationnelle": 0.763,
        },
    },
}

LEAN_SUPPORT = {
    "Juste à temps (JAT)": "Robots autonomes, WMS, RFID",
    "Takt Time": "Big Data & Analytics, Systèmes cyber-physiques, ERP, WMS",
    "Heijunka": "WMS, MES",
    "Méthode TPM / TRS": "MES, RFID",
    "Poka Yoke": "Simulation, Robots autonomes, ERP",
    "Kaizen": "MES, RFID, Big Data & Analytics, Fabrication additive (Impression 3D)",
    "Kanban": "Fabrication additive (Impression 3D)",
    "Value Stream Mapping (VSM)": "Systèmes cyber-physiques, RFID, WMS",
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
    "Lean_MÃ©thode TPM / TRS": "Méthode TPM / TRS",
    "Lean_Kata": "Kata",
    "Lean_Just in time": "Juste à temps (JAT)",
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
        "title": "Scénario 1 : Retard technologique",
        "icon": "🔧",
        "body": "Votre entreprise présente une maturité organisationnelle plus avancée que son niveau d’adoption technologique. Il faut prioriser les outils Lean et les technologies Industrie 4.0 déjà utilisés dans le cluster cible.",
        "recommendations": [
            "Prioriser les technologies et méthodes Lean déjà présentes dans le cluster cible mais encore absentes de l’entreprise.",
            "Utiliser les leviers majeurs de l’arbre de décision comme premiers axes d’implémentation.",
            "Commencer par la feuille de route technologique puis consolider la feuille de route organisationnelle.",
        ],
    },
    "org_lag": {
        "title": "Scénario 2 : Retard organisationnel",
        "icon": "⚡",
        "body": "L’adoption technologique est en avance sur la préparation organisationnelle. Il faut d’abord renforcer le pilotage, les processus et l’apprentissage collectif avant de déployer davantage d’outils.",
        "recommendations": [
            "Prioriser les sous-dimensions Lean 4.0 ayant les plus grands écarts négatifs face au cluster cible.",
            "Renforcer d’abord le leadership, les routines opérationnelles et l’organisation apprenante.",
            "Commencer par la feuille de route organisationnelle puis réintégrer progressivement la feuille de route technologique.",
        ],
    },
    "aligned": {
        "title": "Scénario 3 : Alignement stratégique",
        "icon": "🚀",
        "body": "La maturité organisationnelle et l’adoption technologique sont alignées. L’enjeu est d’accélérer de manière équilibrée en ciblant les leviers les plus impactants.",
        "recommendations": [
            "Maintenir un rythme équilibré entre progression des dimensions de maturité et adoption technologique.",
            "Utiliser les variables les plus influentes de l’arbre de décision pour identifier les prochains accélérateurs.",
            "Faire progresser de façon coordonnée la feuille de route technologique et la feuille de route organisationnelle.",
        ],
    },
}

SIZE_OPTIONS = [
    "TPE / Small company",
    "PME / Medium company",
    "Grande entreprise / Large company",
]

QUESTION_MAP = {
    "Leadership - Engagement Lean ": "Dans quelle mesure la direction est-elle réellement engagée dans la démarche Lean au sein de votre entreprise ?",
    "Leadership - Engagement DT": "Dans quelle mesure la direction est-elle engagée dans la transformation digitale et l’Industrie 4.0 ?",
    "Leadership - Stratégie ": "Dans quelle mesure la stratégie de l’entreprise intègre-t-elle clairement la transformation Lean 4.0 ?",
    "Leadership - Communication": "Dans quelle mesure la communication autour de la transformation Lean 4.0 est-elle claire, régulière et partagée ?",
    "Supply Chain - Collaboration inter-organisationnelle": "Dans quelle mesure votre entreprise collabore-t-elle efficacement avec ses partenaires de la chaîne logistique ?",
    "Supply Chain - Traçabilité": "Dans quelle mesure les flux, produits et informations sont-ils traçables tout au long de la supply chain ?",
    "Supply Chain - Impact sur les employées": "Dans quelle mesure les transformations de la supply chain prennent-elles en compte l’impact sur les employés ?",
    "Opérations - Standardisation des processus": "Dans quelle mesure les processus opérationnels sont-ils standardisés et maîtrisés ?",
    "Opérations - Juste-à-temps (JAT)": "Dans quelle mesure les principes du juste-à-temps sont-ils appliqués dans vos opérations ?",
    "Opérations - Gestion des résistances": "Dans quelle mesure votre entreprise gère-t-elle efficacement les résistances au changement ?",
    "Technologies - Connectivité et gestion des données": "Dans quelle mesure les systèmes de votre entreprise sont-ils connectés et les données exploitées efficacement ?",
    "Technologies - Automatisation": "Dans quelle mesure les opérations et processus sont-ils automatisés ?",
    "Technologies - Pilotage du changement": "Dans quelle mesure le déploiement technologique est-il piloté et accompagné de manière structurée ?",
    "Organisation apprenante  - Formation et développement des compétences": "Dans quelle mesure l’entreprise développe-t-elle les compétences nécessaires à la transformation Lean 4.0 ?",
    "Organisation apprenante  - Collaboration et Partage des Connaissances": "Dans quelle mesure la collaboration interne et le partage des connaissances sont-ils encouragés ?",
    "Organisation apprenante  - Flexibilité organisationnelle": "Dans quelle mesure l’organisation est-elle flexible et capable de s’adapter rapidement ?",
}

DIMENSION_WEIGHTS = {
    "Leadership": 0.2057,
    "Supply Chain": 0.1996,
    "Opérations": 0.1990,
    "Technologies": 0.1904,
    "Organisation Apprenante": 0.2053,
}

TOPSIS_REFERENCE_SCORES = {
    1: 73.64, 2: 51.26, 3: 75.92, 4: 42.47, 5: 68.60, 6: 42.38, 7: 71.15, 8: 39.47, 9: 31.46, 10: 72.06,
    11: 76.09, 12: 5.97, 13: 65.55, 14: 65.85, 15: 83.74, 16: 71.24, 17: 58.22, 18: 55.05, 19: 64.45, 20: 72.98,
    21: 51.77, 22: 22.58, 23: 69.25, 24: 49.68, 25: 65.20, 26: 51.01, 27: 56.12, 28: 55.90, 29: 75.91, 30: 59.98,
    31: 50.23, 32: 71.86, 33: 27.88, 34: 72.44, 35: 44.49, 36: 93.56, 37: 64.23, 38: 59.53, 39: 46.58, 40: 44.22,
    41: 25.00, 42: 34.81, 43: 62.19, 44: 55.89, 45: 39.08, 46: 67.34, 47: 65.07, 48: 73.80, 49: 56.97, 50: 49.64,
    51: 27.92, 52: 53.42, 53: 62.37, 54: 58.93, 55: 51.36, 56: 83.09, 57: 30.23, 58: 49.61, 59: 58.68,
}


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
                Optimisez votre transformation Lean 4.0 grâce à l’intelligence issue du terrain.
                Cette plateforme s’appuie sur des données réelles d’entreprises pour proposer un diagnostic structuré,
                des benchmarks visuels et une feuille de route personnalisée, réaliste et actionnable.
            </p>
            <div class="workflow-grid">
                <div class="workflow-step"><strong>01. Cadrer</strong><span>Charger la base et définir le périmètre d’analyse.</span></div>
                <div class="workflow-step"><strong>02. Segmenter</strong><span>Identifier les groupes de maturité et leur structure.</span></div>
                <div class="workflow-step"><strong>03. Lire</strong><span>Analyser PCA, radars et heatmaps.</span></div>
                <div class="workflow-step"><strong>04. Expliquer</strong><span>Interpréter les facteurs clés via l’arbre de décision.</span></div>
                <div class="workflow-step"><strong>05. Diagnostiquer</strong><span>Évaluer une entreprise existante ou un nouveau questionnaire.</span></div>
                <div class="workflow-step"><strong>06. Décider</strong><span>Construire la feuille de route et les priorités.</span></div>
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
    df = pd.read_excel(uploaded_file) if uploaded_file is not None else pd.read_excel(DEFAULT_DATASET)
    df = df.reset_index(drop=True).copy()
    if "Num" in df.columns:
        df = df.drop(columns=["Num"])
    df.insert(0, "Num", np.arange(1, len(df) + 1))
    return df


@st.cache_data(show_spinner=False)
def load_image() -> Image.Image:
    return Image.open(MATURITY_IMAGE)


def normalize_company_number(value: object) -> object:
    if pd.isna(value):
        return value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def build_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <strong>Configuration Studio</strong>
            <span>Conservez ici uniquement le cadrage du diagnostic. L’évaluation détaillée d’une entreprise se fait dans le module Application.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    selected_features: List[str] = []
    dimension_scope = st.sidebar.multiselect(
        "1. Dimensions à inclure",
        list(DIMENSION_MAP.keys()),
        default=list(DIMENSION_MAP.keys()),
        help="Choisissez les dimensions du modèle de maturité à intégrer dans l’analyse.",
    )
    fine_tune = st.sidebar.checkbox("Ajuster les sous-dimensions", value=False, help="Activez cette option uniquement si vous souhaitez retirer manuellement certaines questions.")

    for dimension in dimension_scope:
        sub_dims = DIMENSION_MAP[dimension]
        if fine_tune:
            with st.sidebar.expander(f"{dimension} questions", expanded=False):
                selected = st.multiselect(
                    f"Sous-dimensions actives pour {dimension}",
                    sub_dims,
                    default=sub_dims,
                    key=f"features_{dimension}",
                )
        else:
            selected = sub_dims
        selected_features.extend(selected)

    radar_dimensions = st.sidebar.multiselect(
        "### 2. Dimensions du radar",
        dimension_scope or list(DIMENSION_MAP.keys()),
        default=dimension_scope or list(DIMENSION_MAP.keys()),
    )
    radar_features: List[str] = []
    for dimension in radar_dimensions:
        radar_features.extend(DIMENSION_MAP[dimension])

    st.sidebar.markdown("### 3. Paramètres du clustering")
    k_range = st.sidebar.slider("Plage de recherche de K", 2, 10, (2, 6), help="L’application calcule les courbes Elbow et Silhouette sur cette plage avant de retenir le nombre de clusters opérationnel.")
    k_values = list(range(k_range[0], k_range[1] + 1))
    default_k = 3 if 3 in k_values else k_values[0]
    final_k = st.sidebar.select_slider("Nombre de clusters retenu", options=k_values, value=default_k, help="Choisissez la structure de clusters à utiliser dans toute l’application.")

    if "Num" in df.columns:
        company_options = [normalize_company_number(val) for val in df["Num"].dropna().tolist()]
    else:
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
    labels = ["Niveau Initial", "Niveau Intégré", "Niveau Avancé"]
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

    jat_column = "Opérations - Juste-à-temps (JAT)"
    if jat_column in normalized.columns:
        normalized["Lean_Juste à temps"] = normalized[jat_column].apply(lambda x: 1 if x in [4, 5] else 0)
        if "Lean_Just in time" in normalized.columns:
            normalized["Lean_Just in time"] = (
                normalized["Lean_Juste à temps"].fillna(0).astype(int)
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
        "Niveau de maturité Lean 4.0",
        "Taille entreprise ",
        "Secteur industriel",
        "Méthodes Lean ",
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
        "Capture d’un nouveau questionnaire",
        "Renseignez directement les réponses de l’entreprise sur la plateforme. Chaque score correspond à une question liée à une sous-dimension du modèle Lean 4.0.",
    )

    link_col_1, link_col_2 = st.columns(2)
    with link_col_1:
        st.link_button("Ouvrir l’éditeur du questionnaire", "https://docs.google.com/forms/d/18q1_-kOGChcj4DbGp7onkGYWFRK9_EW382yCCRSH4U8/edit", use_container_width=True)
    with link_col_2:
        st.link_button("Ouvrir le formulaire répondant", "https://forms.gle/Uc7689Y6Y45qpiTo7", use_container_width=True)

    with st.form("manual_company_form", clear_on_submit=False):
        meta_1, meta_2, meta_3 = st.columns(3)
        company_name = meta_1.text_input("Nom de l’entreprise", value="Nouvelle entreprise cliente")
        sector_candidates = sorted([str(val) for val in df_reference.get("Secteur industriel", pd.Series(dtype=object)).dropna().unique().tolist()])
        if "Autre" not in sector_candidates:
            sector_candidates.append("Autre")
        company_sector_choice = meta_2.selectbox("Secteur industriel", sector_candidates, index=0 if sector_candidates else None)
        company_size = meta_3.selectbox("Taille de l’entreprise", SIZE_OPTIONS, index=1)
        custom_sector = st.text_input("Secteur personnalisé", value="", placeholder="À remplir seulement si vous choisissez Autre")

        manual_scores: Dict[str, float] = {}
        st.markdown("### Questionnaire de maturité Lean 4.0")
        st.caption("Attribuez une note entière de 1 à 5 pour chaque question.")
        for dimension, sub_dims in DIMENSION_MAP.items():
            with st.expander(f"🧩 {dimension}", expanded=True):
                for sub_dim in sub_dims:
                    question_text = QUESTION_MAP.get(sub_dim, sub_dim.strip())
                    st.markdown(f"**{question_text}**")
                    st.caption(f"Sous-dimension : {sub_dim.strip()}")
                    manual_scores[sub_dim] = st.slider(
                        "Score",
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
            "Méthodes Lean déjà adoptées",
            options=sorted(lean_options.keys()),
        )
        selected_tech = st.multiselect(
            "Technologies Industrie 4.0 déjà adoptées",
            options=sorted(tech_options.keys()),
        )

        submitted = st.form_submit_button("Analyser la nouvelle entreprise", use_container_width=True)

    if not submitted:
        st.info("Complétez les questions puis cliquez sur « Analyser la nouvelle entreprise » pour générer le diagnostic complet.")
        return pd.Series(dtype=object)

    manual_company = pd.Series(0, index=df_reference.columns, dtype=object)
    manual_company["Nom entreprise"] = company_name
    manual_company["Secteur industriel"] = custom_sector if company_sector_choice == "Autre" and custom_sector.strip() else company_sector_choice
    manual_company["Taille entreprise "] = company_size

    for col, value in manual_scores.items():
        manual_company[col] = value

    for label in selected_lean:
        manual_company[lean_options[label]] = 1
    for label in selected_tech:
        manual_company[tech_options[label]] = 1

    return manual_company


def determine_scenario(cluster_label: str, predicted_dt: str) -> str:
    order = {"Niveau Initial": 1, "Niveau Intégré": 2, "Niveau Avancé": 3}
    cluster_rank = order.get(cluster_label, 0)
    dt_rank = order.get(predicted_dt, 0)
    if dt_rank < cluster_rank:
        return "tech_lag"
    if dt_rank > cluster_rank:
        return "org_lag"
    return "aligned"


def priority_from_gap(value: float) -> str:
    if value <= -1.0:
        return "Élevé"
    if value <= -0.5:
        return "Moyen"
    return "Faible"


def priority_from_adoption(value: float) -> str:
    if value >= 0.7:
        return "Élevé"
    if value >= 0.4:
        return "Moyen"
    return "Faible"


def style_priority_cell(value: object) -> str:
    palette = {
        "Élevé": "background-color:#fee2e2;color:#991b1b;font-weight:800;border-radius:8px;",
        "Moyen": "background-color:#fef3c7;color:#92400e;font-weight:700;border-radius:8px;",
        "Faible": "background-color:#dcfce7;color:#166534;font-weight:700;border-radius:8px;",
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
        .replace("Opérations - ", "Ops - ")
        .replace("OpÃƒÂ©rations - ", "Ops - ")
        .replace("OpÃ©rations - ", "Ops - ")
    )
    return "\\n".join(textwrap.wrap(clean, width=22)) or clean


def build_dimension_comparison(entreprise: pd.Series, cluster_target: pd.Series, selected_features: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    dimension_groups = {
        "Leadership": [col for col in selected_features if "Leadership" in col],
        "Opérations": [col for col in selected_features if "Opérations" in col or "OpÃ©rations" in col or "Operations" in col],
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


def compute_weighted_topsis_score(entreprise: pd.Series, selected_features: List[str]) -> Tuple[float, pd.DataFrame]:
    rows = []
    for dimension, cols in DIMENSION_MAP.items():
        valid_cols = [col for col in cols if col in selected_features and col in entreprise.index]
        if not valid_cols:
            continue
        dimension_weight = DIMENSION_WEIGHTS.get(dimension, 0.0)
        sub_weight = dimension_weight / len(valid_cols)
        for col in valid_cols:
            value = float(pd.to_numeric(entreprise.get(col, np.nan), errors="coerce"))
            if not np.isnan(value):
                rows.append(
                    {
                        "dimension": dimension,
                        "sub_dimension": col,
                        "score": value,
                        "weight": sub_weight,
                    }
                )

    score_df = pd.DataFrame(rows)
    if score_df.empty:
        return 0.0, pd.DataFrame()

    total_weight = score_df["weight"].sum()
    if total_weight <= 0:
        return 0.0, pd.DataFrame()
    score_df["weight"] = score_df["weight"] / total_weight
    score_df["weighted_score"] = score_df["score"] * score_df["weight"]
    score_df["ideal"] = 5.0 * score_df["weight"]
    score_df["anti_ideal"] = 1.0 * score_df["weight"]

    d_plus = float(np.sqrt(((score_df["weighted_score"] - score_df["ideal"]) ** 2).sum()))
    d_minus = float(np.sqrt(((score_df["weighted_score"] - score_df["anti_ideal"]) ** 2).sum()))
    score = 0.0 if (d_plus + d_minus) == 0 else (d_minus / (d_plus + d_minus)) * 100

    dimension_scores = score_df.groupby("dimension")[["score"]].mean()
    return round(score, 2), dimension_scores


def render_final_maturity_result(scenario_key: str, organizational_score: float, company_reference: str | None = None) -> None:
    prefix_map = {"tech_lag": "RT", "org_lag": "RO", "aligned": "AL"}
    active_prefix = prefix_map.get(scenario_key, "AL")
    circle_classes = {
        "RO": "background:#ef4444;color:#ffffff;",
        "AL": "background:#84cc16;color:#ffffff;",
        "RT": "background:#ef4444;color:#ffffff;",
    }
    inactive_style = "background:#f3f4f6;color:#94a3b8;border:2px solid #cbd5e1;"

    def style_for(label: str) -> str:
        return circle_classes[label] if label == active_prefix else inactive_style

    st.markdown(
        f"""
        <div class="roadmap-card" style="padding:1.2rem 1.3rem;">
            <div class="roadmap-title">Résultat final de l’évaluation Lean 4.0</div>
            <p class="roadmap-copy">Le score de maturité organisationnelle est enrichi par un indice d’alignement technologique pour produire une notation synthétique de type RT-75, RO-68 ou AL-89.</p>
            {"<p class='roadmap-copy' style='margin-top:0.35rem;'><strong>Entreprise analysée :</strong> " + company_reference + "</p>" if company_reference else ""}
            <div style="display:flex;align-items:center;gap:1.2rem;margin-top:1rem;">
                <div style="display:flex;flex-direction:column;gap:0.8rem;align-items:center;min-width:88px;">
                    <div style="width:64px;height:64px;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;font-weight:800;{style_for('RO')}">RO</div>
                    <div style="width:64px;height:64px;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;font-weight:800;{style_for('AL')}">AL</div>
                    <div style="width:64px;height:64px;border-radius:999px;display:flex;align-items:center;justify-content:center;font-size:1.45rem;font-weight:800;{style_for('RT')}">RT</div>
                </div>
                <div style="flex:1;display:flex;justify-content:center;">
                    <div style="min-width:180px;border-radius:18px;border:1px solid #cbd5e1;background:linear-gradient(180deg,#ffffff,#f8fafc);padding:1rem 1.4rem;box-shadow:0 12px 24px rgba(15,23,42,0.08);text-align:center;">
                        <div style="font-size:0.85rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;">Notation synthétique</div>
                        <div style="font-size:2rem;font-weight:900;color:#0f172a;line-height:1.2;margin-top:0.3rem;">{active_prefix}-{int(round(organizational_score))}</div>
                        <div style="font-size:1rem;color:#334155;margin-top:0.15rem;">Score TOPSIS : {organizational_score:.2f}/100</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        st.image(load_image(), caption="Modèle de Maturité Lean 4.0", use_container_width=True)


def render_clustering_tab(ks, inertia, silhouettes, df_clustered, cluster_label_map) -> None:
    render_section_intro(
        "Analytics",
        "Clustering et fiabilité",
        "Validez la structure du modèle de maturité, analysez les courbes Elbow et Silhouette puis retenez la segmentation utilisée dans toute l’application.",
    )
    st.subheader("Analyse de fiabilité")
    st.success("Cronbach's Alpha global pour les colonnes de maturité sélectionnées : 0.934")
    for group, values in CRONBACH_DATA.items():
        with st.expander(group):
            st.write(f"Alpha : {values['alpha']:.3f}")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Sous-dimension": list(values["items"].keys()),
                        "Alpha si supprimée": list(values["items"].values()),
                    }
                ),
                use_container_width=True,
            )

    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.plot(ks, inertia, marker="o")
        ax.set_title("Méthode Elbow")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertie")
        st.pyplot(fig)
    with right:
        fig, ax = plt.subplots()
        ax.plot(ks, silhouettes, marker="o")
        ax.set_title("Score de silhouette")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        st.pyplot(fig)

    summary = df_clustered["cluster"].value_counts().sort_index()
    summary_df = pd.DataFrame(
        {
            "Cluster": summary.index,
            "Nombre d'entreprises": summary.values,
            "Niveau de maturité Lean 4.0": summary.index.map(cluster_label_map),
        }
    )
    st.markdown("### Synthèse des clusters")
    st.dataframe(summary_df, use_container_width=True)


def render_pca_tab(df_clustered: pd.DataFrame, scaled_features: np.ndarray) -> None:
    render_section_intro(
        "Exploration",
        "Visualisation PCA",
        "Utilisez la PCA pour observer le degré de séparation entre les groupes de maturité et les zones de proximité entre profils d’entreprises.",
    )
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_pca = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    maturity_col = "Niveau de maturité Lean 4.0" if "Niveau de maturité Lean 4.0" in df_clustered.columns else "Niveau de maturitÃ© Lean 4.0"
    df_pca["label"] = df_clustered.loc[df_pca.index, maturity_col].values
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="label", palette="Set2", ax=ax)
    ax.set_title("PCA des clusters")
    st.pyplot(fig)


def render_radar_tab(df_clustered: pd.DataFrame, radar_features: List[str]) -> None:
    render_section_intro(
        "Benchmarking",
        "Comparaison radar",
        "Comprenez les différences entre groupes de maturité au niveau des sous-dimensions et des dimensions pour soutenir une lecture claire côté management.",
    )
    maturity_col = "Niveau de maturité Lean 4.0" if "Niveau de maturité Lean 4.0" in df_clustered.columns else "Niveau de maturitÃ© Lean 4.0"
    cluster_avg = df_clustered.groupby(maturity_col)[radar_features].mean().dropna(axis=1, how="any")
    if cluster_avg.empty:
        st.warning("Pas de données disponibles pour le radar avec la sélection actuelle.")
        return

    palette = {
        "Niveau Initial": ("rgba(0, 0, 139, 1)", "rgba(0, 0, 139, 0.35)"),
        "Niveau Intégré": ("rgba(255, 99, 71, 1)", "rgba(255, 99, 71, 0.25)"),
        "Niveau Avancé": ("rgba(60, 179, 113, 1)", "rgba(60, 179, 113, 0.25)"),
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

    st.markdown("### Radar par dimension")
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
        "Comparez les scores de maturité, l’adoption des méthodes Lean et l’usage des technologies Industrie 4.0 selon les clusters.",
    )
    heatmap_features = df_clustered.filter(items=selected_features)
    if heatmap_features.empty:
        st.warning("Aucune colonne de maturité valide n’est disponible pour la heatmap.")
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
    axes[0].set_title("Scores moyens du questionnaire par cluster")

    if lean_avg.empty:
        axes[1].text(0.5, 0.5, "Aucune colonne de méthode Lean détectée.", ha="center", va="center")
        axes[1].axis("off")
    else:
        sns.heatmap(lean_avg.T, cmap="Oranges", annot=True, fmt=".2f", linewidths=0.8, ax=axes[1])
        axes[1].set_title("Adoption moyenne des méthodes Lean par cluster")

    if tech_avg.empty:
        axes[2].text(0.5, 0.5, "Aucune colonne de technologie Industrie 4.0 détectée.", ha="center", va="center")
        axes[2].axis("off")
    else:
        sns.heatmap(tech_avg.T, cmap="PuRd", annot=True, fmt=".2f", linewidths=0.8, ax=axes[2])
        axes[2].set_title("Adoption moyenne des technologies Industrie 4.0 par cluster")

    plt.tight_layout()
    st.pyplot(fig)


def render_decision_tree_tab(clf: DecisionTreeClassifier, X: pd.DataFrame) -> None:
    render_section_intro(
        "Interpretability",
        "Arbre de décision et facteurs clés",
        "Expliquez quelles méthodes Lean et quels leviers technologiques influencent le plus la classification de maturité.",
    )
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_importances = importances[importances > 0].sort_values(ascending=False).head(12)
    st.markdown("### Facteurs les plus influents")
    if top_importances.empty:
        st.info("Aucune importance non nulle n’a été détectée pour la configuration actuelle de l’arbre.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        top_importances.sort_values().plot(kind="barh", ax=ax, color="#1d4ed8")
        ax.set_title("Variables les plus influentes", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")
        st.pyplot(fig, use_container_width=True)

    st.markdown("### Visualisation de l’arbre de décision")
    st.caption("Arbre coloré orienté de gauche à droite pour une lecture plus claire par les décideurs.")
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
        ax.set_title("Arbre de décision", fontsize=18, fontweight="bold")
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
        "Évaluation et feuille de route personnalisée",
        "Évaluez une entreprise existante ou un nouveau questionnaire puis générez une comparaison au cluster cible et une feuille de route personnalisée.",
    )
    mode = st.radio("Mode d’application", ["Entreprise existante de la base", "Nouvelle entreprise"], horizontal=True)
    company_identifier = None

    if mode == "Entreprise existante de la base":
        if "Num" in df_clustered.columns:
            company_options = [normalize_company_number(val) for val in df_clustered["Num"].dropna().tolist()]
            selected_company = st.selectbox(
                "Choisissez l’entreprise à diagnostiquer (Num)",
                company_options,
                index=company_options.index(selected_company) if selected_company in company_options else 0,
            )
            entreprise_matches = df_clustered.loc[df_clustered["Num"].apply(normalize_company_number) == selected_company]
            if entreprise_matches.empty:
                st.error("Impossible de retrouver l’entreprise sélectionnée dans la base alignée.")
                st.stop()
            entreprise = entreprise_matches.iloc[0]
            company_identifier = normalize_company_number(selected_company)
        else:
            selected_company = st.selectbox(
                "Choisissez l’entreprise à diagnostiquer",
                df_clustered.index.tolist(),
                index=df_clustered.index.tolist().index(selected_company) if selected_company in df_clustered.index else 0,
            )
            entreprise = df_clustered.loc[selected_company]
            company_identifier = selected_company
        company_label = f"Entreprise #{company_identifier}"
        st.markdown("### Profil de l'entreprise")
        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Num", company_identifier)
        info_2.metric("Secteur", entreprise.get("Secteur industriel", "N/A"))
        info_3.metric("Taille", entreprise.get("Taille entreprise ", "N/A"))
    else:
        entreprise = build_manual_company_input(df_clustered)
        if entreprise.empty:
            return
        company_label = entreprise.get("Nom entreprise", "Nouvelle entreprise")
        st.markdown("### Profil de la nouvelle entreprise")
        info_1, info_2, info_3 = st.columns(3)
        info_1.metric("Entreprise", company_label)
        info_2.metric("Secteur", entreprise.get("Secteur industriel", "N/A"))
        info_3.metric("Taille", entreprise.get("Taille entreprise ", "N/A"))

    st.markdown("### Scores de maturité par sous-dimension")
    st.dataframe(company_dimension_table(entreprise, selected_features), use_container_width=True)

    lean_cols = [col for col in df_clustered.columns if col.startswith("Lean_")]
    tech_cols = [col for col in df_clustered.columns if col.startswith("Tech_")]
    lean_adopted = [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_cols if entreprise.get(col, 0) == 1]
    tech_adopted = [col.replace("Tech_", "") for col in tech_cols if entreprise.get(col, 0) == 1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Méthodes Lean déjà adoptées")
        st.dataframe(pd.DataFrame({"Méthode Lean": lean_adopted or ["Aucune méthode Lean détectée"]}), use_container_width=True)
    with col2:
        st.markdown("### Technologies Industrie 4.0 déjà adoptées")
        st.dataframe(pd.DataFrame({"Technologie 4.0": tech_adopted or ["Aucune technologie détectée"]}), use_container_width=True)

    entreprise_scaled = scaler.transform(entreprise[selected_features].values.reshape(1, -1))
    predicted_cluster = int(kmeans.predict(entreprise_scaled)[0])
    predicted_cluster_label = cluster_label_map.get(predicted_cluster, "Inconnu")
    actual_organizational_label = predicted_cluster_label
    if mode == "Entreprise existante de la base":
        dataset_maturity = entreprise.get("Niveau Maturité")
        if pd.notna(dataset_maturity):
            actual_organizational_label = str(dataset_maturity)

    features_dt_new = pd.DataFrame([entreprise]).reindex(columns=X.columns, fill_value=0)
    predicted_dt = clf.predict(features_dt_new)[0]

    if mode == "Entreprise existante de la base" and company_identifier in TOPSIS_REFERENCE_SCORES:
        organizational_score = TOPSIS_REFERENCE_SCORES[company_identifier]
    else:
        organizational_score, _ = compute_weighted_topsis_score(entreprise, selected_features)

    cluster_col, score_col, tree_col = st.columns(3)
    cluster_col.metric("Maturité organisationnelle", actual_organizational_label)
    score_col.metric("Score de maturité organisationnelle", f"{organizational_score:.2f}/100")
    tree_col.metric("Maturité technologique", predicted_dt)
    st.caption(f"Entreprise analysée : {company_label}")
    if mode == "Entreprise existante de la base":
        comparison_df = pd.DataFrame(
            {
                "Num": [company_identifier],
                "Niveau organisationnel réel (colonne Niveau Maturité)": [actual_organizational_label],
                "Niveau technologique prédit": [predicted_dt],
            }
        )
        st.markdown("### Référence utilisée pour le scénario final")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    scenario_key = determine_scenario(actual_organizational_label, predicted_dt)
    scenario = SCENARIO_TEXT[scenario_key]
    render_final_maturity_result(scenario_key, organizational_score, company_label)
    st.markdown(
        f"""
        <div class="hero-box" style="padding:1.1rem 1.2rem; margin-top:0.8rem;">
            <div class="hero-title" style="font-size:1.35rem;">{scenario['icon']} {scenario['title']}</div>
            <p class="hero-copy">{scenario['body']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Méthode de calcul du score global Lean 4.0", expanded=False):
        st.markdown(
            """
            Le résultat final combine :
            1. Un score continu de maturité organisationnelle calculé par la logique TOPSIS pondérée.
            2. Un indice qualitatif d’alignement entre maturité organisationnelle et maturité technologique :
               `RT` pour retard technologique, `RO` pour retard organisationnel et `AL` pour alignement.

            La note finale affichée prend donc la forme :
            `RT-75`, `RO-68` ou `AL-89`.
            """
        )
    st.markdown("## 🧭 Guide d’utilisation personnalisé")
    st.markdown(
        """
        1. Identification du scénario d’adoption de l’entreprise.
        2. Lecture des écarts par rapport au cluster cible via les radars et les tableaux d’écart.
        3. Génération de deux feuilles de route complémentaires :
        - feuille de route organisationnelle Lean 4.0
        - feuille de route technologique et méthodes Lean
        4. Priorisation et exécution progressive selon le scénario détecté.
        """
    )
    st.markdown("## 🔍 Analyse comparative et recommandations")
    for idx, recommendation in enumerate(scenario["recommendations"], start=1):
        st.write(f"{idx}. {recommendation}")

    application_features = df_clustered.filter(items=selected_features)
    selected_features = application_features.columns.tolist()
    if not selected_features:
        st.error("Aucune sous-dimension valide n’est disponible pour l’analyse entreprise.")
        st.stop()

    cluster_means = (
        pd.concat([df_clustered[["cluster"]], application_features], axis=1)
        .groupby("cluster")
        .mean(numeric_only=True)
    )
    cluster_rank = cluster_means.mean(axis=1).sort_values().index.tolist()
    current_position = cluster_rank.index(predicted_cluster)
    next_cluster = cluster_rank[min(current_position + 1, len(cluster_rank) - 1)]

    st.markdown("### Radar : entreprise vs cluster cible")
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

    st.markdown("### Radar par dimension : entreprise vs cluster cible")
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
            "Écart": negative_gaps.round(2).values,
            "Priorité": [priority_from_gap(x) for x in negative_gaps.values],
        }
    )

    st.markdown("### 🗺️ Feuille de route personnalisée")
    st.markdown(
        """
        <div class="roadmap-card">
            <div class="roadmap-title">Transformation roadmap</div>
            <p class="roadmap-copy">Cette feuille de route est structurée en deux volets synchronisés : progression de la maturité organisationnelle et adoption prioritaire des méthodes Lean et technologies Industrie 4.0.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not gap_df.empty:
        high_count = int((gap_df["Priorité"] == "Élevé").sum())
        medium_count = int((gap_df["Priorité"] == "Moyen").sum())
        low_count = int((gap_df["Priorité"] == "Faible").sum())
    else:
        high_count = 0
        medium_count = 0
        low_count = 0

    stat_1, stat_2, stat_3 = st.columns(3)
    stat_1.metric("Priorité élevée", high_count)
    stat_2.metric("Priorité moyenne", medium_count)
    stat_3.metric("Priorité faible", low_count)

    st.markdown("#### Feuille de route organisationnelle : écarts à résorber")
    if gap_df.empty:
        st.success("Aucun écart négatif détecté par rapport au cluster cible.")
    else:
        st.dataframe(
            build_roadmap_styler(gap_df, "Écart", "YlOrRd_r", "Priorité"),
            use_container_width=True,
        )

    lean_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, lean_cols].mean()
    tech_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, tech_cols].mean()
    lean_to_adopt = lean_cluster_mean[(lean_cluster_mean > 0) & (entreprise[lean_cluster_mean.index] == 0)].sort_values(ascending=False)
    tech_to_adopt = tech_cluster_mean[(tech_cluster_mean > 0) & (entreprise[tech_cluster_mean.index] == 0)].sort_values(ascending=False)

    roadmap_col, tech_col = st.columns(2)
    with roadmap_col:
        st.markdown("#### Feuille de route technologique : méthodes Lean à adopter")
        lean_df = pd.DataFrame(
            {
                "Méthode Lean": [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_to_adopt.index],
                "Taux d'adoption dans le cluster cible": lean_to_adopt.round(2).values,
                "Priorité": [priority_from_adoption(v) for v in lean_to_adopt.values],
            }
        )
        if lean_df.empty:
            st.info("Aucune méthode Lean prioritaire à adopter.")
        else:
            st.dataframe(
                build_roadmap_styler(lean_df, "Taux d'adoption dans le cluster cible", "Blues", "Priorité"),
                use_container_width=True,
            )

    with tech_col:
        st.markdown("#### Feuille de route technologique : technologies Industrie 4.0 à adopter")
        tech_df = pd.DataFrame(
            {
                "Technologie": [col.replace("Tech_", "") for col in tech_to_adopt.index],
                "Taux d'adoption dans le cluster cible": tech_to_adopt.round(2).values,
                "Priorité": [priority_from_adoption(v) for v in tech_to_adopt.values],
            }
        )
        if tech_df.empty:
            st.info("Aucune technologie prioritaire à adopter.")
        else:
            st.dataframe(
                build_roadmap_styler(tech_df, "Taux d'adoption dans le cluster cible", "PuBu", "Priorité"),
                use_container_width=True,
            )

    executive_lines = []
    if not gap_df.empty:
        executive_lines.append(f"Top priorités organisationnelles : {', '.join(gap_df['Sous-dimension'].head(3).tolist())}.")
    if not lean_df.empty:
        executive_lines.append(f"Top priorités Lean : {', '.join(lean_df['Méthode Lean'].head(3).tolist())}.")
    if not tech_df.empty:
        executive_lines.append(f"Top priorités technologiques : {', '.join(tech_df['Technologie'].head(3).tolist())}.")
    if executive_lines:
        st.markdown("#### Synthèse exécutive")
        for line in executive_lines:
            st.write(f"- {line}")
def main() -> None:
    render_hero()
    st.sidebar.title("Control Studio")
    st.sidebar.caption("Configurez le périmètre de l’analyse. Le module Application gère l’évaluation détaillée des entreprises.")
    dataset_mode = st.sidebar.radio(
        "Source des données",
        ["Base du dépôt", "Upload custom Excel"],
        index=0,
    )
    uploaded_file = None
    if dataset_mode == "Upload custom Excel":
        uploaded_file = st.sidebar.file_uploader("Uploader un fichier Excel", type=["xlsx"])
    df_raw = load_dataset(uploaded_file)
    sidebar = build_sidebar(df_raw)

    selected_features = sidebar["selected_features"]
    if not selected_features:
        st.warning("Veuillez sélectionner au moins une sous-dimension pour continuer.")
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
    if "Num" in aligned_df.columns:
        valid_company_numbers = [normalize_company_number(val) for val in aligned_df["Num"].dropna().tolist()]
        if selected_company not in valid_company_numbers:
            selected_company = valid_company_numbers[0]
            st.sidebar.warning("L’entreprise sélectionnée contenait des valeurs manquantes sur les sous-dimensions actives. La première entreprise valide a été choisie à la place.")
    else:
        if selected_company not in aligned_df.index:
            selected_company = aligned_df.index[0]
            st.sidebar.warning("L’entreprise sélectionnée contenait des valeurs manquantes sur les sous-dimensions actives. La première entreprise valide a été choisie à la place.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_frame)
    ks, inertia, silhouettes = compute_cluster_metrics(scaled_features, sidebar["k_range"])
    kmeans = KMeans(n_clusters=sidebar["final_k"], random_state=42, n_init=10)
    aligned_df["cluster"] = kmeans.fit_predict(scaled_features)

    selected_features = [col for col in selected_features if col in aligned_df.columns]
    if not selected_features:
        st.error("Aucune sous-dimension valide n’est disponible dans le dataset actif après alignement.")
        st.stop()
    radar_features = [col for col in radar_features if col in selected_features]
    if not radar_features:
        radar_features = selected_features.copy()

    cluster_feature_frame = aligned_df.filter(items=selected_features)
    selected_features = cluster_feature_frame.columns.tolist()
    if not selected_features:
        st.error("Aucune sous-dimension valide n’est disponible pour l’agrégation des clusters.")
        st.stop()
    cluster_means = pd.concat([aligned_df[["cluster"]], cluster_feature_frame], axis=1).groupby("cluster").mean(numeric_only=True)
    cluster_label_map = rank_cluster_labels(cluster_means)
    aligned_df["Niveau de maturité Lean 4.0"] = aligned_df["cluster"].map(cluster_label_map)

    normalized_df = normalize_columns(aligned_df.copy())
    target_col = "Niveau Maturité" if "Niveau Maturité" in normalized_df.columns else "Niveau de maturité Lean 4.0"
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

