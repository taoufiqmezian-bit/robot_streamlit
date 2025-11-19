import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# =============================================================================
# CHARGEMENT ET EXPLORATION DES DONNÉES
# =============================================================================

print("="*70)
print("EXERCICE 3 - DÉTECTION DE DÉFAILLANCES D'UN ROBOT")
print("="*70)

# 1. Charger le fichier Robot.csv
# 🚨 Correction : Utilisation du chargement standard pour les environnements locaux
try:
    df = pd.read_csv("Robot.csv")
except FileNotFoundError:
    print("Erreur fatale : Le fichier 'Robot.csv' est introuvable.")
    print("Assurez-vous que le fichier est dans le même répertoire que le script.")
    import sys
    sys.exit() # Arrête le script si le fichier n'est pas trouvé

print("\n📊 Aperçu des données:")
print(df.head())
print(f"\nDimensions: {df.shape}")
print(f"Colonnes: {list(df.columns)}")

# 2. Afficher le nombre de cycles normaux et défaillants
print("\n📈 Répartition des cycles:")
print(df["Cycle_Normal"].value_counts())
print(f"\nPourcentage de défaillances: {(df['Cycle_Normal']==0).sum()/len(df)*100:.2f}%")

# 3. Pairplot avec cycles défaillants en rouge
print("\n🎨 Génération du pairplot...")
sns.pairplot(df, hue="Cycle_Normal", palette={1:"green", 0:"red"})
plt.suptitle("Pairplot - Cycles normaux (vert) vs défaillants (rouge)", y=1.01)
plt.show()

# =============================================================================
# PARTIE 1 - APPRENTISSAGE NON SUPERVISÉ : ISOLATION FOREST
# =============================================================================

print("\n" + "="*70)
print("PARTIE 1 - ISOLATION FOREST (Non supervisé)")
print("="*70)

# A. Copier toutes les colonnes sauf la dernière dans X
X = df.iloc[:, :-1]
print("\n📦 Caractéristiques (X):")
print(X.info())

# B. Isolation Forest avec 2% de contamination
print("\n🌲 Entraînement de l'Isolation Forest...")
model = IsolationForest(contamination=0.02, random_state=42)
model.fit(X)

pred = model.predict(X)
print(f"Prédictions: {len(pred[pred==-1])} anomalies détectées")

# C. Tableau croisé pour évaluer les performances
temp = pd.DataFrame()
temp["IF_pred"] = pred
temp["Cycle_Normal"] = df["Cycle_Normal"]

print("\n📊 Matrice de confusion (Isolation Forest):")
confusion = pd.crosstab(temp["Cycle_Normal"], temp["IF_pred"],
                        rownames=['Réel'], colnames=['Prédit'])
print(confusion)

# Calcul des métriques (Sécurisé)
vrais_negatifs = confusion.loc[0, -1] if -1 in confusion.columns and 0 in confusion.index else 0
faux_positifs = confusion.loc[1, -1] if -1 in confusion.columns and 1 in confusion.index else 0
faux_negatifs = confusion.loc[0, 1] if 1 in confusion.columns and 0 in confusion.index else 0
vrais_positifs = confusion.loc[1, 1] if 1 in confusion.columns and 1 in confusion.index else 0

print(f"\n✅ Vrais positifs (normaux détectés): {vrais_positifs}")
print(f"✅ Vrais négatifs (défauts détectés): {vrais_negatifs}")
print(f"❌ Faux positifs (normaux vus comme défauts): {faux_positifs}")
print(f"❌ Faux négatifs (défauts non détectés): {faux_negatifs}")

# D. Ajouter la colonne des anomalies
tempDF = X.copy()
tempDF["Anomalie"] = pred
print("\n📋 Dataframe avec anomalies:")
print(tempDF.head())

# E. Afficher les lignes contenant des anomalies
anomalies = tempDF[tempDF["Anomalie"] == -1]
print(f"\n🔍 {len(anomalies)} anomalies détectées:")
print(anomalies)

print("\n💡 Conclusion Isolation Forest:")
print("   - Méthode non supervisée : n'utilise pas l'étiquette 'Cycle_Normal'")

# =============================================================================
# PARTIE 2 - APPRENTISSAGE SUPERVISÉ : ARBRE DE DÉCISION (AVEC SPLIT TRAIN/TEST)
# =============================================================================

print("\n" + "="*70)
print("PARTIE 2 - ARBRE DE DÉCISION (Supervisé avec Train/Test Split)")
print("="*70)

# B. Déterminer les entrées (X) et la sortie (y)
X_tree = df.drop(columns=["Cycle_Normal"])
y_tree = df["Cycle_Normal"]

# 📢 Séparation des données en ensembles d'entraînement (Train) et de test (Test)
X_train, X_test, y_train, y_test = train_test_split(
    X_tree, y_tree, test_size=0.3, random_state=42, stratify=y_tree
)

print(f"\n📥 Entrées (X_train): {X_train.shape[0]} exemples d'entraînement")
print(f"📥 Entrées (X_test): {X_test.shape[0]} exemples de test")
print(f"📤 Sortie (y): Classes: {y_tree.unique()} (0=Défaillant, 1=Normal)")


# C. Entraîner l'arbre de décision
print("\n🌳 Entraînement de l'arbre de décision sur l'ensemble d'entraînement...")
tree = DecisionTreeClassifier(max_depth=None, random_state=0)
tree.fit(X_train, y_train)

# D. Nombre de nœuds et Scores
print(f"📊 Nombre de nœuds dans l'arbre: {tree.tree_.node_count}")
print(f"📏 Profondeur de l'arbre: {tree.get_depth()}")

# Score de précision (Évalué sur les deux ensembles)
score_train = tree.score(X_train, y_train)
score_test = tree.score(X_test, y_test)

print(f"\n🎯 Précision sur les données d'entraînement (Train): {score_train*100:.2f}%")
print(f"🎯 Précision sur les données de test (Test): {score_test*100:.2f}%")
print("\n👉 Si le score Train est beaucoup plus élevé que le score Test, il y a surapprentissage (overfitting).")

# E. Afficher l'arbre
print("\n🎨 Génération de la visualisation de l'arbre...")
plt.figure(figsize=(20, 12))
plot_tree(tree,
          feature_names=X_tree.columns,
          class_names=["Défaillant", "Normal"],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("Arbre de décision - Détection de défaillances (Entraîné sur 70% des données)", fontsize=16, pad=20)
plt.tight_layout()
plt.show()

# =============================================================================
# ANALYSE ET COMMENTAIRES
# =============================================================================

print("\n" + "="*70)
print("ANALYSE ET COMMENTAIRES")
print("="*70)

print("\n❓ Pourquoi l'arbre utilise X[0], X[1], X[2] ?")
print("   - X[0], X[1], X[2] sont les variables sélectionnées car elles sont les plus discriminantes.")

print("\n❓ Si j'ajoute d'autres valeurs dans Robot.csv, le résultat change-t-il ?")
print("   ✅ OUI, le modèle s'adapte aux nouvelles données d'entraînement.")

print("\n🎓 Différences clés entre les deux méthodes:")
print("   - ISOLATION FOREST : Non supervisé, cherche l'isolement.")
print("   - ARBRE DE DÉCISION : Supervisé, utilise les étiquettes 'Cycle_Normal' pour maximiser la précision.")

print("\n" + "="*70)
print("FIN DE L'ANALYSE")
print("="*70)