

## Nom complet :Hassi Asmae
## Classe : S8 CAC G2
## Apogée : 24010417

# Modifications techniques apportées

## 1. Correction du ConvergenceWarning dans la Régression Logistique

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modele = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

modele.fit(X_train, y_train)
````

Problème :
Un avertissement de type `ConvergenceWarning` apparaissait lors de l'entraînement du modèle de régression logistique.

Solution :

* Augmentation du nombre maximal d'itérations avec le paramètre `max_iter=1000`.
* Intégration de `StandardScaler` dans un `Pipeline` afin de normaliser les données avant l'entraînement du modèle.

---

# Correction des avertissements Seaborn

## 2. Correction du FutureWarning dans le Countplot

```python
sns.countplot(
    x=df["target"],
    hue=df["target"],
    palette="coolwarm",
    legend=False
)
```

Problème :
L'utilisation du paramètre `palette` sans spécifier `hue` est désormais obsolète dans les versions récentes de Seaborn (v0.14).

Solution :

* Ajout du paramètre `hue=df["target"]`.
* Désactivation de la légende avec `legend=False`.

---

## 3. Correction du FutureWarning dans le Barplot

```python
sns.barplot(
    x="Accuracy",
    y="Modèle",
    hue="Modèle",
    data=df_results,
    palette="coolwarm",
    legend=False
)
```

Problème :
Un avertissement similaire apparaissait lors de la création d'un barplot horizontal utilisant `palette` sans `hue`.

Solution :

* Ajout du paramètre `hue="Modèle"`.
* Suppression de la légende afin d'éviter une redondance visuelle.

---

# Technologies et bibliothèques utilisées

* Python 3.12
* Jupyter Notebook ou Google Colab
* Scikit-learn :

  * LogisticRegression
  * DecisionTreeClassifier
  * KNeighborsClassifier
* Pandas et NumPy pour la manipulation et l'analyse des données
* Seaborn (v0.13+) et Matplotlib pour la visualisation
* Pipeline et StandardScaler pour l'automatisation du prétraitement

---

# Structure du Notebook

```
Formation_ML_&_DL:S1_Apprentissage_supervisé
(Hajar_HAMINE_S8_CAC_G2_22001267).ipynb
│
├── 01. Chargement et exploration des données
├── 02. Prétraitement des données (division 80/20)
├── 03. Entraînement des modèles
│     ├── Régression logistique
│     ├── Arbre de décision
│     └── K plus proches voisins (KNN)
├── 04. Évaluation des modèles et calcul des métriques
├── 05. Visualisation des résultats (warnings corrigés)
└── 06. Comparaison des performances des modèles
```

---

# Objectif du projet

Ce projet vise à appliquer et comparer plusieurs algorithmes d’apprentissage supervisé afin de :

* Comprendre le fonctionnement de différents modèles de Machine Learning.
* Évaluer leurs performances à l’aide de métriques comme l’accuracy.
* Visualiser les résultats pour faciliter l’interprétation.
* Mettre en place un pipeline de prétraitement automatisé.


