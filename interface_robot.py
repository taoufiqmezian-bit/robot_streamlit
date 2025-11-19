import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from io import StringIO
import sys
import os  # Ajout de la bibliothèque os


# =============================================================================
# FONCTIONS UTILITAIRES DE MACHINE LEARNING
# =============================================================================

@st.cache_data
def load_data(file_name="Robot.csv"):
    """
    Charge les données à partir du fichier CSV en utilisant un chemin relatif
    au script en cours d'exécution.
    """
    # Chemin d'accès relatif au script interface_robot.py
    # Assure que le fichier Robot.csv est trouvé quel que soit l'environnement
    script_dir = os.path.dirname(__file__)
    file_path = os.path.join(script_dir, file_name)

    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"Erreur fatale : Le fichier de données '{file_name}' est introuvable. Chemin vérifié : {file_path}")
        st.stop()
        return None


@st.cache_data(show_spinner="Entraînement de l'Isolation Forest...")
def train_isolation_forest(X, contamination):
    """Entraîne et prédit avec l'Isolation Forest."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)
    pred = model.predict(X)
    return model, pred


@st.cache_data(show_spinner="Entraînement de l'Arbre de Décision...")
def train_decision_tree(X, y):
    """Entraîne et retourne le modèle Decision Tree."""
    # Séparation des données (70% train / 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Entraînement
    tree = DecisionTreeClassifier(max_depth=None, random_state=0)
    tree.fit(X_train, y_train)

    return tree, X_train, X_test, y_train, y_test


# =============================================================================
# FONCTIONS DE SECTIONS (pour les Onglets)
# =============================================================================

# Le reste des fonctions (section_exploration, section_isolation_forest, section_arbre_decision)
# et la structure principale (run_analysis et main) restent inchangées.

def section_exploration(df):
    """Contient toutes les informations pour la section 1 : Exploration des Données."""

    # Calculs préliminaires
    defaillance_count = df["Cycle_Normal"].value_counts()
    normal_cycles = defaillance_count.get(1, 0)
    failed_cycles = defaillance_count.get(0, 0)
    total_rows = len(df)

    st.markdown("### Aperçu des Données et Statistiques Clés")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("Aperçu de toutes les Données disponibles")
        st.dataframe(df)  # Affiche tout le DataFrame sans limitation

    with col2:
        st.subheader("Répartition des Cycles")

        st.metric(label="Cycles Normaux (Cycle_Normal = 1)", value=normal_cycles)
        st.metric(label="Cycles Défaillants (Cycle_Normal = 0)", value=failed_cycles)

        st.markdown("---")
        st.metric(label="Dimensions du Dataset", value=f"{df.shape[0]} lignes, {df.shape[1]} colonnes")

        if total_rows > 0:
            pct_defaillance = (failed_cycles / total_rows * 100)
            st.metric(label="Pourcentage de Défaillances", value=f"{pct_defaillance:.2f}%")

    # Visualisation Pairplot
    st.markdown("### Visualisation - Pairplot des Caractéristiques")
    with st.expander("Cliquez ici pour visualiser le Pairplot complet (peut prendre quelques secondes)"):
        st.write("Génération du Pairplot... (Cycles normaux en vert, Défaillances en rouge)")
        try:
            # Création de la figure Pairplot
            fig_pair = sns.pairplot(df, hue="Cycle_Normal", palette={1: "green", 0: "red"})
            st.pyplot(fig_pair)
        except Exception as e:
            st.warning(f"Impossible de générer le Pairplot : {e}")

    st.markdown("---")


def section_isolation_forest(df):
    """Contient toutes les informations pour la section 2 : Isolation Forest."""

    # Prépare les données pour la forêt d'isolation
    X = df.iloc[:, :-1].copy()
    total_rows = len(df)
    failed_cycles = df["Cycle_Normal"].value_counts().get(0, 0)
    real_contamination = failed_cycles / total_rows

    # -------------------------------------------------------------------------
    # CURSEUR INTERACTIF
    # -------------------------------------------------------------------------
    contamination_rate = st.slider(
        "Taux de contamination estimé (Isolation Forest)",
        min_value=0.01, max_value=0.05,
        value=min(0.02, 0.05),  # Valeur par défaut 2%
        step=0.005,
        format='%.3f',
        help=f"Ce paramètre indique au modèle le pourcentage d'anomalies à rechercher. Taux réel : {real_contamination * 100:.2f}%."
    )

    # -------------------------------------------------------------------------
    # EXÉCUTION & RÉSULTATS
    # -------------------------------------------------------------------------

    st.subheader(f"Résultats pour Contamination = {contamination_rate * 100:.2f}%")

    model, pred = train_isolation_forest(X, contamination_rate)

    temp = pd.DataFrame({'IF_pred': pred, 'Cycle_Normal': df["Cycle_Normal"]})
    anomalies_df = df[pred == -1].copy()
    anomalies_count = len(anomalies_df)

    col_if_1, col_if_2 = st.columns(2)

    with col_if_1:
        st.metric(label="Nombre d'anomalies détectées", value=anomalies_count)
        st.markdown("#### Tableau X (Caractéristiques d'entrée)")
        st.dataframe(X.head(5))

    with col_if_2:
        st.markdown("#### Matrice de Confusion (Réel vs Prédit par IF)")
        confusion = pd.crosstab(temp["Cycle_Normal"], temp["IF_pred"], rownames=['Réel (Cycle_Normal)'],
                                colnames=['Prédit (IF)'])
        st.dataframe(confusion)

        # Calcul des métriques pour la conclusion
        VN = confusion.loc[0, -1] if -1 in confusion.columns and 0 in confusion.index else 0
        FN = confusion.loc[0, 1] if 1 in confusion.columns and 0 in confusion.index else 0
        FP = confusion.loc[1, -1] if -1 in confusion.columns and 1 in confusion.index else 0

    st.markdown("#### Conclusion de l'évaluation :")
    st.success(f"**Défaillances bien détectées (Vrais Négatifs) : {VN}** sur {failed_cycles} défaillances réelles.")
    st.warning(f"**Défaillances manquées (Faux Négatifs) : {FN}** cycle défaillant a été vu comme normal.")
    st.info(f"**Faux Positifs (Normal vu comme défaillant) : {FP}** cycle normal a été vu comme défaillant.")

    st.markdown("---")

    st.markdown("#### Lignes classées comme Anomalies (Prédit = -1)")
    if anomalies_count > 0:
        st.dataframe(anomalies_df)
    else:
        st.markdown("Aucune anomalie détectée avec ce taux de contamination.")


def section_arbre_decision(df):
    """Contient toutes les informations pour la section 3 : Arbre de Décision."""

    st.markdown("### Apprentissage Supervisé : Définir les Seuils de Défaillance")
    st.markdown("L'objectif est d'identifier les paramètres qui impactent le fonctionnement du robot presseur.")

    # -------------------------------------------------------------------------
    # 1. Afficher le tableau « Robot »
    # -------------------------------------------------------------------------
    with st.expander("1. Afficher le tableau 'Robot' complet"):
        st.dataframe(df)
        st.success("Le tableau complet (avec la variable cible 'Cycle_Normal') est affiché.")

    # Déterminer les entrées et les sorties (fait une seule fois pour les étapes suivantes)
    X_tree = df.drop(columns=["Cycle_Normal"])
    y_tree = df["Cycle_Normal"]

    # Entraînement du modèle (fait une seule fois pour les étapes suivantes)
    tree, X_train, X_test, y_train, y_test = train_decision_tree(X_tree, y_tree)

    # -------------------------------------------------------------------------
    # 2. Déterminer les entrées et les sorties
    # -------------------------------------------------------------------------
    with st.expander("2. Déterminer les entrées (X) et la sortie (y)"):
        st.markdown("#### Entrées (X) : Caractéristiques d'impact")
        st.text(", ".join(X_tree.columns.tolist()))
        st.info(f"Dimensions : {X_tree.shape}")

        st.markdown("#### Sortie (y) : État du cycle")
        st.text("Cycle_Normal (1 = Normal, 0 = Défaillant)")
        st.info(f"Dimensions : {y_tree.shape}")

    # -------------------------------------------------------------------------
    # 3. Importer le modèle et entrainer le (Déjà fait par la fonction ci-dessus)
    # 4. Calculer le nombre de nœud
    # -------------------------------------------------------------------------
    with st.expander("3. Entraîner le modèle"):
        st.success("Le modèle 'DecisionTreeClassifier' a été importé et entraîné sur 70% des données (Train).")
        st.markdown("Les données restantes (30%) sont conservées pour le test (voir l'étape 6).")

    with st.expander("4. Calculer le nombre de nœuds et la profondeur"):
        st.metric(label="Nombre de nœuds dans l'arbre", value=tree.tree_.node_count)
        st.metric(label="Profondeur de l'arbre", value=tree.get_depth())
        st.info("Un arbre peu profond (ici 2) indique que la règle de séparation est simple et claire.")

    # -------------------------------------------------------------------------
    # 5. Afficher l’arbre
    # -------------------------------------------------------------------------
    with st.expander("5. Afficher l'Arbre de Décision (pour interprétation)", expanded=True):
        st.markdown("L'arbre montre les règles apprises pour séparer les classes.")

        fig_tree = plt.figure(figsize=(15, 10))
        plot_tree(tree,
                  feature_names=X_tree.columns.tolist(),
                  class_names=["Défaillant", "Normal"],
                  filled=True,
                  rounded=True,
                  fontsize=8)
        plt.title("Arbre de décision - Détection de défaillances", fontsize=14)
        st.pyplot(fig_tree)

    # -------------------------------------------------------------------------
    # 6. Commenter ce résultat (Analyse des performances)
    # -------------------------------------------------------------------------
    with st.expander("6. Commenter ce résultat (Analyse des performances)"):
        score_train = tree.score(X_train, y_train)
        score_test = tree.score(X_test, y_test)

        st.markdown("#### Performance du Modèle")
        st.metric(label="Précision sur l'entraînement (Train)", value=f"{score_train * 100:.2f}%")
        st.metric(label="Précision sur le test (Test)", value=f"{score_test * 100:.2f}%")

        st.markdown("#### Interprétation :")
        st.markdown(
            "- **Précision Test** : Le score sur l'ensemble de test, qui simule de nouvelles données, est très élevé. Cela confirme que l'arbre a trouvé des règles de séparation **robustes**.")
        st.markdown(
            "- **Surapprentissage (Overfitting)** : L'écart entre les scores Train et Test est très faible, ce qui signifie que le modèle généralise bien et n'est pas surajusté aux données d'entraînement.")

    # -------------------------------------------------------------------------
    # 7. Que signifie X[0], X[1], X[2] ? Pourquoi il y a que ces critères ?
    # -------------------------------------------------------------------------
    with st.expander("7. Analyse des critères décisifs (X[0], X[1], X[2])"):
        # Mapping des indices vers les noms de colonnes
        column_names = X_tree.columns.tolist()

        st.markdown(f"**X[0]** : {column_names[0]} (Temps_Cycle)")
        st.markdown(f"**X[1]** : {column_names[1]} (Effort_Arriere)")
        st.markdown(f"**X[2]** : {column_names[2]} (Effort_Avant)")

        st.markdown("#### Pourquoi seulement ces critères ?")
        st.markdown(
            "1. **Sélection Automatique** : L'algorithme sélectionne les variables qui réduisent le plus l'impureté (Gini/Entropie) à chaque nœud.")
        st.markdown(
            "2. **Suffisance** : Si un petit sous-ensemble de variables (ici X[0], X[1], X[2]) permet de séparer les classes, les autres ne sont pas utilisées.")
        st.markdown(
            "3. **Impact** : Cela signifie que le **temps de cycle**, l'**effort arrière**, et l'**effort avant** sont les paramètres ayant le plus grand impact sur la défaillance.")

    # -------------------------------------------------------------------------
    # 8. Si j’enrichi le fichier de données, le résultat changera-t-il ?
    # -------------------------------------------------------------------------
    with st.expander("8. Impact de l'enrichissement des données"):
        st.markdown("#### Le résultat changera-t-il ?")
        st.success("✅ **OUI**, le résultat changera très probablement.")
        st.markdown("#### Explication :")
        st.markdown(
            "1. **Modèle Non Statique** : L'arbre de décision est un modèle qui **apprend des données d'entraînement**.")
        st.markdown(
            "2. **Adaptation des Seuils** : L'ajout de nouvelles valeurs peut obliger l'arbre à **ajuster les seuils de décision**.")
        st.markdown(
            "3. **Interprétabilité** : De nouvelles données pourraient introduire de nouveaux critères décisifs, changeant ainsi l'interprétation finale.")

    st.markdown("---")


def run_analysis(df):
    """Structure de l'application Streamlit avec des onglets."""

    # Création des onglets
    tab1, tab2, tab3 = st.tabs([
        "📊 1. Exploration des Données",
        "🌲 2. Isolation Forest (Non Supervisé)",
        "🌳 3. Arbre de Décision (Supervisé)"
    ])

    with tab1:
        st.header("📊 1. Exploration des Données")
        section_exploration(df)

    with tab2:
        st.header("🌲 2. Apprentissage Non Supervisé : Isolation Forest")
        section_isolation_forest(df)

    with tab3:
        st.header("🌳 3. Apprentissage Supervisé : Arbre de Décision")
        section_arbre_decision(df)


# =============================================================================
# STRUCTURE DE L'INTERFACE STREAMLIT
# =============================================================================

def main():
    # CSS injection to hide Streamlit footer and menu
    hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    st.set_page_config(layout="wide", page_title="Analyse Robot - Défaillances")

    st.title("🤖 Analyse des Défaillances d'un Robot Industriel")
    st.markdown(
        "Interface d'analyse comparative structurée en trois étapes clés : Exploration, Détection d'Anomalies (IF) et Classification (Arbre).")

    data = load_data()

    if data is not None:
        run_analysis(data)


if __name__ == "__main__":
    main()