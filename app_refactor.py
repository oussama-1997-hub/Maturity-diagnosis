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
