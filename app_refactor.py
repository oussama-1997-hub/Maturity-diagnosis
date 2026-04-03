from __future__ import annotations

from pathlib import Path
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
from sklearn.tree import DecisionTreeClassifier, export_graphviz


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
    "Lean_Maki-Gami/Hoshin…etc": "Maki-Gami / Hoshin",
    "Lean_Value Stream Mapping (VSM)": "Value Stream Mapping (VSM)",
    "Lean_Kaizen": "Kaizen",
    "Lean_DDMRP": "DDMRP",
    "Lean_Méthode TPM / TRS": "Méthode TPM / TRS",
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
        "title": "Scenario 1: Technology lag",
        "icon": "🔧",
        "body": "The organizational maturity is stronger than the technology adoption. Prioritize the missing Lean tools and Industry 4.0 enablers used by the next maturity cluster.",
    },
    "org_lag": {
        "title": "Scenario 2: Organizational lag",
        "icon": "⚡",
        "body": "Technology adoption is ahead of organizational readiness. Focus on process discipline, leadership, and learning-system gaps before scaling more tools.",
    },
    "aligned": {
        "title": "Scenario 3: Strategic alignment",
        "icon": "🚀",
        "body": "Organizational maturity and technology adoption are aligned. Continue with balanced improvements and target the highest-impact decision-tree drivers.",
    },
}


def render_hero() -> None:
    st.markdown(
        """
        <style>
        .hero-box {
            padding: 1.6rem 1.8rem;
            border-radius: 24px;
            background: linear-gradient(135deg, #f7fbff 0%, #eef9f4 100%);
            border: 1px solid rgba(16, 110, 95, 0.12);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            color: #12332b;
            margin-bottom: 0.4rem;
        }
        .hero-copy {
            color: #48615b;
            font-size: 1rem;
            line-height: 1.7;
            margin-bottom: 0;
        }
        .info-card {
            border: 1px solid rgba(18, 51, 43, 0.08);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            background: #ffffff;
            min-height: 118px;
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
        </style>
        <div class="hero-box">
            <div class="hero-title">Lean 4.0 Studio</div>
            <p class="hero-copy">
                Full-featured Lean 4.0 analytics with a more structured workflow: data intake, clustering, PCA, radar analysis,
                heatmaps, decision tree insights, and a guided application module for company-level recommendations.
            </p>
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
    st.sidebar.title("Workflow")
    st.sidebar.markdown("Use the controls below to configure the complete analysis pipeline.")

    selected_features: List[str] = []
    st.sidebar.markdown("### 1. Select maturity dimensions")
    for dimension, sub_dims in DIMENSION_MAP.items():
        with st.sidebar.expander(dimension, expanded=False):
            selected = st.multiselect(
                f"Sub-dimensions for {dimension}",
                sub_dims,
                default=sub_dims,
                key=f"features_{dimension}",
            )
            selected_features.extend(selected)

    radar_dimensions = st.sidebar.multiselect(
        "### 2. Radar dimensions",
        list(DIMENSION_MAP.keys()),
        default=list(DIMENSION_MAP.keys()),
    )
    radar_features: List[str] = []
    for dimension in radar_dimensions:
        radar_features.extend(DIMENSION_MAP[dimension])

    st.sidebar.markdown("### 3. Clustering settings")
    k_range = st.sidebar.slider("K range", 2, 10, (2, 6))
    k_values = list(range(k_range[0], k_range[1] + 1))
    default_k = 3 if 3 in k_values else k_values[0]
    final_k = st.sidebar.selectbox("Final number of clusters", k_values, index=k_values.index(default_k))

    company_options = df.index.tolist()
    default_company = 4 if len(company_options) > 4 else 0
    selected_company = st.sidebar.selectbox("### 4. Company for application module", company_options, index=default_company)

    return {
        "selected_features": selected_features,
        "radar_features": radar_features,
        "k_range": k_range,
        "final_k": final_k,
        "selected_company": selected_company,
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
        ("Lean_Méthode TPM / TRS", "Lean_TPM / TRS method"),
        ("Tech_Réalité augmentée", "Tech_Augmented reality"),
        ("Tech_Systèmes cyber physiques", "Tech_Cyber ​​physical systems"),
        ("Tech_Intelligence artificielle", "Tech_Artificial intelligence"),
        ("Tech_Robots autonomes", "Tech_Autonomous robots"),
    ]

    for primary, alias in merge_pairs:
        if primary in normalized.columns and alias in normalized.columns:
            normalized[primary] = normalized[primary].fillna(0).astype(int) | normalized[alias].fillna(0).astype(int)
            normalized.drop(columns=[alias], inplace=True)

    if {"Lean_DDMRP/ hoshin kanri", "Lean_DDMRP", "Lean_Maki-Gami/Hoshin…etc"}.issubset(normalized.columns):
        normalized["Lean_DDMRP/ hoshin kanri"] = (
            normalized["Lean_DDMRP/ hoshin kanri"].fillna(0).astype(int)
            | normalized["Lean_DDMRP"].fillna(0).astype(int)
            | normalized["Lean_Maki-Gami/Hoshin…etc"].fillna(0).astype(int)
        )
        normalized.drop(columns=["Lean_DDMRP", "Lean_Maki-Gami/Hoshin…etc"], inplace=True)

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


def determine_scenario(cluster_label: str, predicted_dt: str) -> str:
    order = {"Niveau Initial": 1, "Niveau Intégré": 2, "Niveau Avancé": 3}
    cluster_rank = order.get(cluster_label, 0)
    dt_rank = order.get(predicted_dt, 0)
    if dt_rank < cluster_rank:
        return "tech_lag"
    if dt_rank > cluster_rank:
        return "org_lag"
    return "aligned"


def render_overview(df: pd.DataFrame, selected_features: List[str], cluster_labels: Dict[int, str]) -> None:
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
        st.write(pd.DataFrame({"Cluster": list(cluster_labels.keys()), "Label": list(cluster_labels.values())}))
    with image_col:
        st.image(load_image(), caption="Modèle de Maturité Lean 4.0", use_container_width=True)


def render_clustering_tab(ks, inertia, silhouettes, df_clustered, cluster_label_map) -> None:
    st.subheader("Reliability snapshot")
    st.success("Cronbach's Alpha global for the selected maturity columns: 0.934")
    for group, values in CRONBACH_DATA.items():
        with st.expander(group):
            st.write(f"Alpha: {values['alpha']:.3f}")
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
            "Niveau de maturité Lean 4.0": summary.index.map(cluster_label_map),
        }
    )
    st.markdown("### Cluster summary")
    st.dataframe(summary_df, use_container_width=True)


def render_pca_tab(df_clustered: pd.DataFrame, scaled_features: np.ndarray) -> None:
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_features)
    df_pca = pd.DataFrame(pca_result, columns=["PCA1", "PCA2"])
    df_pca["label"] = df_clustered.loc[df_pca.index, "Niveau de maturité Lean 4.0"].values
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_pca, x="PCA1", y="PCA2", hue="label", palette="Set2", ax=ax)
    ax.set_title("PCA of clusters")
    st.pyplot(fig)


def render_radar_tab(df_clustered: pd.DataFrame, radar_features: List[str]) -> None:
    cluster_avg = df_clustered.groupby("Niveau de maturité Lean 4.0")[radar_features].mean().dropna(axis=1, how="any")
    if cluster_avg.empty:
        st.warning("No radar data available for the selected dimensions.")
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
    avg_scores = df_clustered.groupby("cluster")[selected_features].mean()
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
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_importances = importances[importances > 0].sort_values(ascending=False).head(20)

    left, right = st.columns([0.9, 1.1])
    with left:
        st.markdown("### Feature importances")
        if top_importances.empty:
            st.info("No non-zero feature importances were found for the current decision-tree configuration.")
        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            top_importances.sort_values().plot(kind="barh", ax=ax, color="steelblue")
            ax.set_title("Top decision-tree drivers")
            st.pyplot(fig)
    with right:
        st.markdown("### Decision tree")
        dot_data = export_graphviz(
            clf,
            out_file=None,
            feature_names=X.columns,
            class_names=[str(c) for c in clf.classes_],
            filled=True,
            rounded=True,
            special_characters=True,
        )
        st.graphviz_chart(dot_data)


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
    entreprise = df_clustered.loc[selected_company]
    st.markdown("### Company profile")
    info_1, info_2, info_3 = st.columns(3)
    info_1.metric("Company index", selected_company)
    info_2.metric("Secteur", entreprise.get("Secteur industriel", "N/A"))
    info_3.metric("Taille", entreprise.get("Taille entreprise ", "N/A"))

    st.markdown("### Maturity scores by sub-dimension")
    st.dataframe(company_dimension_table(entreprise, selected_features), use_container_width=True)

    lean_cols = [col for col in df_clustered.columns if col.startswith("Lean_")]
    tech_cols = [col for col in df_clustered.columns if col.startswith("Tech_")]
    lean_adopted = [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_cols if entreprise.get(col, 0) == 1]
    tech_adopted = [col.replace("Tech_", "") for col in tech_cols if entreprise.get(col, 0) == 1]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Lean methods already adopted")
        st.dataframe(pd.DataFrame({"Méthode Lean": lean_adopted or ["Aucune méthode détectée"]}), use_container_width=True)
    with col2:
        st.markdown("### Industry 4.0 technologies already adopted")
        st.dataframe(pd.DataFrame({"Technologie 4.0": tech_adopted or ["Aucune technologie détectée"]}), use_container_width=True)

    entreprise_scaled = scaler.transform(entreprise[selected_features].values.reshape(1, -1))
    predicted_cluster = int(kmeans.predict(entreprise_scaled)[0])
    predicted_cluster_label = cluster_label_map.get(predicted_cluster, "Inconnu")

    features_dt_new = pd.DataFrame([entreprise]).reindex(columns=X.columns, fill_value=0)
    predicted_dt = clf.predict(features_dt_new)[0]

    cluster_col, tree_col = st.columns(2)
    cluster_col.metric("Organizational maturity (KMeans)", predicted_cluster_label)
    tree_col.metric("Technological maturity (Decision Tree)", predicted_dt)

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

    cluster_means = df_clustered.groupby("cluster")[selected_features].mean()
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

    gaps = (entreprise[selected_features] - cluster_means.loc[next_cluster]).sort_values()
    negative_gaps = gaps[gaps < 0]
    gap_df = pd.DataFrame(
        {
            "Sous-dimension": negative_gaps.index,
            "Écart": np.round(negative_gaps.values, 2),
            "Priorité": ["Élevée" if x <= -1 else "Moyenne" if x <= -0.5 else "Faible" for x in negative_gaps.values],
        }
    )
    st.markdown("### Improvement gaps")
    st.dataframe(gap_df, use_container_width=True)

    lean_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, lean_cols].mean()
    tech_cluster_mean = df_clustered.loc[df_clustered["cluster"] == next_cluster, tech_cols].mean()
    lean_to_adopt = lean_cluster_mean[(lean_cluster_mean > 0) & (entreprise[lean_cluster_mean.index] == 0)].sort_values(ascending=False)
    tech_to_adopt = tech_cluster_mean[(tech_cluster_mean > 0) & (entreprise[tech_cluster_mean.index] == 0)].sort_values(ascending=False)

    roadmap_col, tech_col = st.columns(2)
    with roadmap_col:
        st.markdown("### Lean methods to adopt")
        lean_df = pd.DataFrame(
            {
                "Méthode Lean": [LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")) for col in lean_to_adopt.index],
                "Technologies support": [LEAN_SUPPORT.get(LEAN_DISPLAY_NAMES.get(col, col.replace("Lean_", "")), "") for col in lean_to_adopt.index],
                "Adoption rate in target cluster": lean_to_adopt.round(2).values,
            }
        )
        st.dataframe(lean_df if not lean_df.empty else pd.DataFrame({"Info": ["Aucune méthode prioritaire à adopter."]}), use_container_width=True)

    with tech_col:
        st.markdown("### Industry 4.0 technologies to adopt")
        tech_df = pd.DataFrame(
            {
                "Technologie": [col.replace("Tech_", "") for col in tech_to_adopt.index],
                "Adoption rate in target cluster": tech_to_adopt.round(2).values,
            }
        )
        st.dataframe(tech_df if not tech_df.empty else pd.DataFrame({"Info": ["Aucune technologie prioritaire à adopter."]}), use_container_width=True)


def main() -> None:
    render_hero()
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
    aligned_df = df_raw.loc[feature_frame.index].copy()
    selected_company = sidebar["selected_company"]
    if selected_company not in aligned_df.index:
        selected_company = aligned_df.index[0]
        st.sidebar.warning("The selected company had missing values for the active sub-dimensions. The first valid company was selected instead.")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_frame)
    ks, inertia, silhouettes = compute_cluster_metrics(scaled_features, sidebar["k_range"])
    kmeans = KMeans(n_clusters=sidebar["final_k"], random_state=42, n_init=10)
    aligned_df["cluster"] = kmeans.fit_predict(scaled_features)

    cluster_means = aligned_df.groupby("cluster")[selected_features].mean()
    cluster_label_map = rank_cluster_labels(cluster_means)
    aligned_df["Niveau de maturité Lean 4.0"] = aligned_df["cluster"].map(cluster_label_map)

    normalized_df = normalize_columns(aligned_df.copy())
    target_col = "Niveau Maturité" if "Niveau Maturité" in normalized_df.columns else "Niveau de maturité Lean 4.0"
    clf, X = train_decision_tree(normalized_df, target_col)

    render_overview(aligned_df, selected_features, cluster_label_map)

    tabs = st.tabs(
        [
            "1. Clustering",
            "2. PCA",
            "3. Radar",
            "4. Heatmaps",
            "5. Decision Tree",
            "6. Application",
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
