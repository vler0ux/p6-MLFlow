#  Projet 7 - Modèle de Scoring Crédit avec MLOps

## Description

Ce projet consiste à construire un **modèle de scoring** pour prédire la probabilité de faillite d'un client, dans le cadre d'une démarche **MLOps de bout en bout**. Il s'appuie sur les données de la compétition Kaggle [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk).

---

## Objectifs

1. **Modèle de scoring** : prédire la probabilité de défaut de remboursement d'un client
2. **Interprétabilité** : analyser les features importantes globalement (feature importance) et localement (SHAP)
3. **MLOps** : tracking des expérimentations, model registry et serving avec MLflow

---

## Structure du projet

```
├── 01_nettoyage_dataset.ipynb         # Nettoyage, imputation, encodage, scaling
├── 02_feature_engineering.ipynb       # Construction des features métier, feature importance
├── 02_modelisation_mlflow.ipynb       # Modélisation, tracking MLflow, optimisation
├── data/                              # Données brutes (non versionnées)
├── models/                            # Modèles sauvegardés
├── mlruns/                            # Runs MLflow (non versionnés)
├── .gitignore
└── README.md
```

---

## Pipeline MLOps

### Étape 1 — Nettoyage & Préparation des données
- Exploration des données brutes (300 000 clients, ~120 features)
- Gestion des valeurs manquantes (`SimpleImputer`, stratégie médiane)
- Encodage des variables catégorielles (`LabelEncoder`)
- Normalisation (`MinMaxScaler`)
- Analyse du déséquilibre des classes (cible : ~8% de défauts)

### Étape 2 — Feature Engineering
- Construction de features métier (ratios financiers, ancienneté emploi, etc.)
- Analyse des corrélations avec la cible
- Visualisation des feature importances (top 15)

### Étape 3 — Modélisation & Expérimentations
| Modèle | AUC-ROC |
|--------|---------|
| Régression Logistique (baseline) | 0.7144 |
| Random Forest sans features métier | 0.6335 |
| Random Forest avec features métier | 0.7134 |
| LightGBM (en cours) | - |

- Validation croisée stratifiée (`StratifiedKFold`)
- Gestion du déséquilibre des classes (`class_weight='balanced'`)
- Tracking des expériences avec **MLflow**

### Étape 4 — Optimisation
- Optimisation des hyperparamètres (`GridSearchCV` / `Optuna`)
- **Coût métier personnalisé** : un faux négatif (FN) coûte 10× plus qu'un faux positif (FP)
- Optimisation du seuil de décision (≠ 0.5 par défaut)

---

## Métriques

- **AUC-ROC** : métrique principale de comparaison des modèles
- **Score métier** : minimisation du coût pondéré FN/FP
- **Recall classe minoritaire** : priorité sur la détection des mauvais payeurs

---

## Stack technique

| Outil | Usage |
|-------|-------|
| Python 3.11 | Langage principal |
| Pandas / NumPy | Manipulation des données |
| Scikit-learn | Preprocessing & modèles |
| LightGBM / XGBoost | Modèles de boosting |
| MLflow | Tracking & model registry |
| Matplotlib / Seaborn | Visualisation |
| SHAP | Interprétabilité locale |

---

## Installation

```bash
# Cloner le repo
git clone <url_du_repo>
cd <nom_du_repo>

# instller uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Créer un environnement virtuel
uv venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
uv pip install -r requirements.txt
ou
uv pip install mlflow scikit-learn pandas numpy matplotlib seaborn jupyterlab xgboost optuna

#installer jupyter lab
pip install jupyterlab
uv pip install jupyterlab
```


---

## Lancer MLflow UI
ouvrir un nouveau terminal

```bash
source venv/bin/activate  # Linux/Mac
```

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Ouvrir http://localhost:5000
```

---

## Points de vigilance

- Le dataset est **déséquilibré** (~8% de défauts) → utiliser `class_weight='balanced'` ou SMOTE
- Le **seuil de décision** est optimisé selon le coût métier et non fixé à 0.5
- Un AUC > 0.82 doit alerter sur un possible **overfitting**
- `SK_ID_CURR` est exclu des features (identifiant sans valeur prédictive)

---

## Auteur

Projet réalisé dans le cadre de la formation **Data Scientist - OpenClassrooms**