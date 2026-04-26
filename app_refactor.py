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

