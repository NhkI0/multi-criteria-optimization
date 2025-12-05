# Méthodologie d'Optimisation Multi-Critère de Portefeuille

## Table des Matières
1. [Introduction](#introduction)
2. [Formulation Mathématique](#formulation-mathématique)
3. [Objectifs d'Optimisation](#objectifs-doptimisation)
4. [Contraintes](#contraintes)
5. [Algorithme d'Optimisation](#algorithme-doptimisation)
6. [Frontière Efficiente](#frontière-efficiente)
7. [Métriques de Performance](#métriques-de-performance)
8. [Choix d'Implémentation](#choix-dimplémentation)
9. [Interprétation des Résultats](#interprétation-des-résultats)

---

## Introduction

### Problématique

L'optimisation de portefeuille est un problème classique en finance quantitative qui consiste à déterminer la meilleure allocation de capital entre différents actifs financiers. Le défi principal est de trouver un équilibre optimal entre deux objectifs contradictoires :

- **Maximiser le rendement espéré** : Augmenter les gains potentiels
- **Minimiser le risque** : Réduire la volatilité et les pertes potentielles

Ce problème est dit "multi-critère" car il n'existe pas une solution unique, mais un ensemble de solutions optimales (Pareto-optimales) qui représentent différents compromis risque-rendement.

### Contexte du Projet

Ce projet utilise des données réelles du marché américain couvrant **11 secteurs économiques** et **192 actifs** sur une période de **2015 à 2025**, permettant une diversification sectorielle complète.

---

## Formulation Mathématique

### Variables de Décision

Soit **w** = (w₁, w₂, ..., wₙ) le vecteur des poids du portefeuille, où :
- **wᵢ** représente la proportion du capital investi dans l'actif i
- **n** = 192 (nombre total d'actifs)

### Rendements et Risque

#### Rendement du Portefeuille

Le rendement espéré du portefeuille est calculé comme :

```
R_p = Σ wᵢ × μᵢ = w^T × μ
```

où :
- **μᵢ** = rendement annuel moyen de l'actif i
- **μ** = vecteur des rendements moyens de tous les actifs

**Pourquoi cette formule ?**
Le rendement d'un portefeuille est simplement la somme pondérée des rendements individuels. C'est une propriété linéaire fondamentale en finance.

#### Risque du Portefeuille (Volatilité)

Le risque est mesuré par la variance (ou son écart-type) :

```
σ_p² = w^T × Σ × w
σ_p = √(w^T × Σ × w)
```

où **Σ** est la matrice de covariance (192×192) des rendements des actifs.

**Pourquoi la covariance ?**
La variance d'un portefeuille ne dépend pas seulement des volatilités individuelles, mais aussi des corrélations entre actifs. La matrice de covariance capture ces interactions :

```
σ_p² = Σᵢ Σⱼ wᵢ wⱼ σᵢⱼ
```

où σᵢⱼ est la covariance entre les actifs i et j.

**Bénéfice de la diversification** : Si deux actifs ont une corrélation < 1, leur combinaison réduit le risque total du portefeuille (principe de diversification de Markowitz).

### Calcul des Statistiques

#### À partir des données historiques

1. **Rendements quotidiens** :
   ```
   r_t = (P_t - P_{t-1}) / P_{t-1}
   ```

2. **Rendement annuel moyen** (annualisé avec 252 jours de trading) :
   ```
   μ = moyenne(r_t) × 252
   ```

3. **Matrice de covariance annualisée** :
   ```
   Σ = cov(r_t) × 252
   ```

**Pourquoi annualiser ?**
Les investisseurs pensent en termes annuels. L'annualisation permet de comparer des actifs avec différentes périodes de détention et facilite l'interprétation.

---

## Objectifs d'Optimisation

### 1. Maximisation du Ratio de Sharpe

#### Formulation

```
maximize: (R_p - r_f) / σ_p
```

où **r_f** est le taux sans risque (par défaut 2% annuel).

#### Pourquoi le Ratio de Sharpe ?

Le ratio de Sharpe mesure le **rendement excédentaire par unité de risque**. C'est la métrique de référence en finance pour comparer des investissements de risques différents.

**Interprétation** :
- Sharpe > 1 : Bon investissement (rendement excédentaire > risque)
- Sharpe > 2 : Excellent investissement
- Sharpe < 0 : Performance inférieure au taux sans risque

**Avantages** :
- Prend en compte à la fois rendement ET risque
- Permet de comparer des portefeuilles de volatilités différentes
- Aligné avec la théorie moderne du portefeuille

**Limites** :
- Suppose que le risque est bien capturé par la volatilité
- Suppose une distribution normale des rendements (peut être problématique pour des actifs avec queues épaisses)

#### Implémentation

Pour maximiser le ratio de Sharpe, on minimise son inverse :

```python
def negative_sharpe(weights):
    returns, risk = portfolio_performance(weights)
    sharpe = (returns - risk_free_rate) / risk
    return -sharpe
```

**Pourquoi minimiser l'inverse ?**
Les algorithmes d'optimisation (comme SLSQP) sont conçus pour la minimisation. Minimiser -Sharpe équivaut à maximiser Sharpe.

### 2. Minimisation de la Variance

#### Formulation

```
minimize: σ_p² = w^T × Σ × w
```

#### Pourquoi Minimiser la Variance ?

Cette approche est idéale pour les investisseurs **averses au risque** qui privilégient la stabilité à la performance.

**Avantages** :
- Problème quadratique convexe (garantit un optimum global)
- Plus stable numériquement
- Résout rapidement même avec beaucoup d'actifs

**Profil d'investisseur** :
- Capital de préservation
- Horizon court terme
- Faible tolérance aux pertes

#### Implémentation

```python
def portfolio_variance(weights):
    return np.dot(weights.T, np.dot(cov_matrix, weights))
```

**Pourquoi minimiser la variance plutôt que l'écart-type ?**
Mathématiquement équivalent (σ minimal ⟺ σ² minimal), mais la variance est plus simple à optimiser (forme quadratique pure).

---

## Contraintes

### 1. Contrainte de Budget (Égalité)

```
Σ wᵢ = 1
```

**Signification** : 100% du capital doit être investi (pas de cash résiduel).

**Implémentation** :
```python
{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
```

**Pourquoi cette contrainte ?**
- Garantit l'allocation complète du capital
- Standard en théorie moderne du portefeuille
- Permet l'interprétation directe des poids comme pourcentages

### 2. Contrainte de Diversification (Bornes)

```
0 ≤ wᵢ ≤ 0.10  (par défaut)
```

**Signification** :
- **borne inférieure (0)** : Pas de ventes à découvert
- **borne supérieure (0.10)** : Maximum 10% par actif

**Pourquoi limiter à 10% ?**

**Avantages** :
- **Réduit le risque de concentration** : Un seul actif en échec ne peut détruire plus de 10% du portefeuille
- **Améliore la liquidité** : Facilite l'entrée/sortie de positions
- **Conformité réglementaire** : Certains fonds ont des limites légales
- **Robustesse** : Réduit la sensibilité aux erreurs d'estimation des rendements

**Compromis** :
- Peut légèrement réduire le rendement théorique maximal
- Force une diversification qui peut diluer la performance des meilleurs actifs

**Paramètre ajustable** :
Dans l'application Streamlit, l'utilisateur peut modifier cette limite (1-100%) selon son profil de risque.

### 3. Contrainte de Non-Négativité

```
wᵢ ≥ 0  ∀i
```

**Signification** : Positions longues uniquement (pas de short-selling).

**Pourquoi interdire les ventes à découvert ?**

**Avantages** :
- **Simplicité** : Plus facile à gérer pour les investisseurs individuels
- **Moins de risque** : Les pertes sur positions longues sont limitées à 100%, les pertes sur shorts sont théoriquement illimitées
- **Pas de coût d'emprunt** : Évite les frais de prêt de titres
- **Réaliste** : Beaucoup d'investisseurs n'ont pas accès aux ventes à découvert

**Cas où autoriser les shorts** :
- Hedge funds sophistiqués
- Stratégies market-neutral
- Couverture de risques spécifiques

### 4. Contrainte de Rendement Cible (Frontière Efficiente uniquement)

```
w^T × μ = R_target
```

**Signification** : Le portefeuille doit atteindre un rendement spécifique.

**Utilisée pour** : Générer la frontière efficiente en balayant différents niveaux de rendement cibles.

**Implémentation** :
```python
{'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) - target_return}
```

---

## Algorithme d'Optimisation

### Méthode : SLSQP (Sequential Least Squares Programming)

#### Qu'est-ce que SLSQP ?

SLSQP est un algorithme d'optimisation non-linéaire avec contraintes qui :
1. Approxime le problème par des moindres carrés séquentiels
2. Utilise des gradients pour converger vers l'optimum
3. Gère les contraintes d'égalité et d'inégalité

#### Pourquoi SLSQP ?

**Avantages** :
- **Robuste** : Fonctionne bien sur des problèmes quadratiques (notre cas)
- **Flexible** : Accepte contraintes d'égalité et d'inégalité
- **Efficace** : Converge rapidement pour 192 actifs
- **Intégré** : Disponible dans SciPy (scipy.optimize.minimize)
- **Éprouvé** : Largement utilisé en finance quantitative

**Alternatives considérées** :
| Algorithme | Avantages | Inconvénients | Verdict |
|------------|-----------|---------------|---------|
| **Quadratic Programming** | Optimal pour variance | Pas pour Sharpe (non-linéaire) | Partiel |
| **Interior Point** | Très précis | Plus lent | Overkill |
| **Genetic Algorithms** | Global optimum | Très lent, non-déterministe | Non adapté |
| **SLSQP** | Rapide, flexible, robuste | Optimum local possible | ✅ Choisi |

#### Configuration de l'Optimisation

```python
result = minimize(
    objective_function,      # negative_sharpe ou portfolio_variance
    initial_weights,         # Point de départ : poids égaux
    method='SLSQP',         # Algorithme choisi
    bounds=bounds,          # (0, 0.10) pour chaque actif
    constraints=constraints, # Budget = 1
    options={
        'maxiter': 1000,    # Maximum 1000 itérations
        'disp': False       # Pas d'affichage verbeux
    }
)
```

#### Point de Départ (Initial Weights)

```python
initial_weights = np.array([1.0 / n_assets] * n_assets)
```

**Pourquoi des poids égaux ?**
- **Neutre** : Aucun biais vers certains actifs
- **Faisable** : Respecte toutes les contraintes (somme = 1, tous positifs)
- **Central** : Au milieu de l'espace de recherche

**Alternative** : Utiliser des poids aléatoires multiples et garder le meilleur résultat (robustesse accrue mais plus lent).

#### Convergence

L'optimisation s'arrête quand :
1. **Gradient proche de zéro** : Plus d'amélioration possible
2. **maxiter atteint** : 1000 itérations dépassées
3. **Contraintes satisfaites** : À une tolérance près (10⁻⁶)

**Vérification du succès** :
```python
if result.success:
    optimal_weights = result.x
else:
    print("Optimisation échouée!")
```

---

## Frontière Efficiente

### Concept

La **frontière efficiente** est l'ensemble de tous les portefeuilles optimaux possibles. Chaque point sur cette courbe représente le portefeuille avec :
- Le **risque minimal** pour un rendement donné, OU
- Le **rendement maximal** pour un risque donné

**Théorème** (Markowitz, 1952) : Tout investisseur rationnel devrait choisir un portefeuille sur la frontière efficiente.

### Génération de la Frontière

#### Algorithme

1. **Trouver les bornes** :
   - Rendement minimal : Portefeuille de variance minimale
   - Rendement maximal : Portefeuille de Sharpe maximal

2. **Discrétisation** :
   ```python
   target_returns = np.linspace(min_return, max_return, n_portfolios)
   ```

3. **Pour chaque rendement cible** :
   - Minimiser la variance
   - Avec contrainte : rendement = cible
   - Stocker les poids optimaux

4. **Résultat** : Collection de portefeuilles optimaux

#### Pourquoi 50 Portefeuilles par Défaut ?

**Compromis précision/vitesse** :
- < 20 : Courbe trop anguleuse, manque de détails
- 50 : Courbe lisse, temps raisonnable (~5-10 secondes)
- > 100 : Amélioration marginale, beaucoup plus lent

**Paramètre ajustable** dans l'app Streamlit (20-100).

### Points Remarquables

#### 1. Portefeuille de Variance Minimale (MVP)

- **Position** : Point le plus à gauche (risque minimal)
- **Caractéristique** : Portefeuille le plus stable
- **Investisseur type** : Très averse au risque

#### 2. Portefeuille Tangent (Max Sharpe)

- **Position** : Point de tangence avec la ligne passant par le taux sans risque
- **Caractéristique** : Meilleur ratio risque/rendement
- **Investisseur type** : Équilibre optimal

**Théorie** : Si on peut emprunter/prêter au taux sans risque, tous les investisseurs devraient détenir ce portefeuille (théorème de séparation en deux fonds).

### Visualisation Interactive

L'application utilise **Plotly** pour une visualisation interactive permettant :
- **Hover** : Voir rendement/risque exact au survol
- **Zoom** : Explorer des régions spécifiques
- **Pan** : Déplacer la vue
- **Export** : Sauvegarder en PNG

---

## Métriques de Performance

### 1. Rendement Annuel Espéré

```
R_p = w^T × μ
```

**Interprétation** : Gain annuel moyen attendu en % du capital investi.

**Exemple** : R_p = 32.44% signifie qu'on s'attend à gagner 32.44% par an en moyenne.

**⚠️ Important** :
- Basé sur des données historiques (performances passées ≠ futures)
- Moyenne de long terme (volatilité court terme peut être forte)

### 2. Volatilité (Risque)

```
σ_p = √(w^T × Σ × w)
```

**Interprétation** : Écart-type annualisé des rendements.

**Exemple** : σ_p = 19.35% signifie :
- Environ 68% du temps, le rendement sera dans [R_p - σ_p, R_p + σ_p]
- Environ 95% du temps, le rendement sera dans [R_p - 2σ_p, R_p + 2σ_p]

**Pour notre portefeuille Max Sharpe** (R = 32.44%, σ = 19.35%) :
- 68% de chance : rendement entre 13.09% et 51.79%
- 95% de chance : rendement entre -6.26% et 71.14%

### 3. Ratio de Sharpe

```
Sharpe = (R_p - r_f) / σ_p
```

**Interprétation** : Rendement excédentaire par unité de risque.

**Échelle d'évaluation** :
| Ratio | Qualité | Signification |
|-------|---------|---------------|
| < 0 | Mauvais | Sous-performe le sans risque |
| 0-0.5 | Faible | Rendement ne compense pas le risque |
| 0.5-1 | Acceptable | Rendement compense modérément le risque |
| 1-2 | Bon | Bon équilibre risque/rendement |
| > 2 | Excellent | Très bon rendement pour le risque pris |

**Notre résultat** : Sharpe = 1.574 → **Bon** investissement

### 4. Rendement Espéré en Dollars

```
Gain_espéré = Capital × R_p
```

**Exemple** : Avec 1M$ et R_p = 32.44%
```
Gain = 1,000,000 × 0.3244 = 324,400 $/an
```

**Utilité** : Concrétise le rendement abstrait (% → $).

### Comparaison des Deux Stratégies

#### Portefeuille Max Sharpe
- **Profil** : Agressif
- **Rendement** : ⬆️ 32.44%
- **Risque** : ⬆️ 19.35%
- **Sharpe** : 1.574
- **Pour qui ?** : Investisseurs acceptant la volatilité pour de meilleurs rendements

#### Portefeuille Min Variance
- **Profil** : Conservateur
- **Rendement** : ⬇️ 12.45%
- **Risque** : ⬇️ 12.91%
- **Sharpe** : 0.810
- **Pour qui ?** : Investisseurs privilégiant la stabilité

---

## Choix d'Implémentation

### 1. Pourquoi Python ?

**Écosystème scientifique mature** :
- **NumPy** : Algèbre linéaire rapide (matrices, produits)
- **Pandas** : Manipulation de séries temporelles
- **SciPy** : Algorithmes d'optimisation
- **Matplotlib/Plotly** : Visualisations
- **Streamlit** : Interface web rapide

**Alternatives** :
- **R** : Bon pour stats, moins polyvalent pour web
- **MATLAB** : Excellent mais payant
- **Julia** : Plus rapide mais écosystème moins mature

### 2. Architecture de Code

#### Classe `PortfolioOptimizer`

**Principe** : Encapsulation orientée objet

```python
class PortfolioOptimizer:
    def __init__(self, data_dir, initial_capital, risk_free_rate)
    def load_data(self)
    def calculate_returns(self)
    def optimize(self, objective, max_weight)
    def generate_efficient_frontier(self, n_portfolios)
```

**Avantages** :
- **Réutilisable** : Une instance, multiples optimisations
- **État partagé** : Données chargées une fois (cov_matrix, mean_returns)
- **Paramétrable** : Configuration flexible
- **Testable** : Facile à unit-tester

### 3. Gestion des Données

#### Structure de Fichiers

```
data/
├── Communication_Services.csv
├── Consumer_Discretionary.csv
├── Consumer_Staples.csv
├── Energy.csv
├── Financials.csv
├── Health_Care.csv
├── Industrials.csv
├── Information_Technology.csv
├── Materials.csv
├── Real_Estate.csv
└── Utilities.csv
```

**Format** : CSV avec dates en index, tickers en colonnes

**Pourquoi séparer par secteur ?**
- **Organisation** : Plus facile à maintenir
- **Traçabilité** : Facile d'identifier la source d'un actif
- **Évolutif** : Ajouter/supprimer des secteurs facilement

#### Nettoyage des Données

```python
# Supprime les colonnes avec > 20% de données manquantes
df = df.dropna(axis=1, thresh=len(df) * 0.8)
```

**Pourquoi 80% de seuil ?**
- Trop strict (ex: 95%) : Perd trop d'actifs
- Trop laxiste (ex: 50%) : Garde des actifs avec peu de données fiables
- 80% : Bon compromis empirique

### 4. Optimisation des Performances

#### Caching dans Streamlit

```python
@st.cache_resource
def load_optimizer():
    optimizer = PortfolioOptimizer()
    optimizer.load_data()
    optimizer.calculate_returns()
    return optimizer
```

**Effet** :
- 1ère exécution : ~3-5 secondes (chargement + calculs)
- Exécutions suivantes : <100ms (données en cache)

**Amélioration UX** : Interface réactive malgré 192 actifs et 2500+ jours de données.

#### Vectorisation NumPy

Au lieu de boucles Python :
```python
# ❌ Lent (boucles Python)
for i in range(n):
    for j in range(n):
        variance += w[i] * w[j] * cov[i][j]

# ✅ Rapide (vectorisé NumPy)
variance = w.T @ cov @ w
```

**Gain** : 50-100x plus rapide grâce aux opérations BLAS optimisées.

---

## Interprétation des Résultats

### Résultats Obtenus

#### Portefeuille Max Sharpe

**Top Holdings** :
1. NVDA (10%) - Information Technology
2. AEP (10%) - Utilities
3. LLY (10%) - Health Care
4. WMT (10%) - Consumer Staples

**Observations** :
- **Diversification sectorielle forte** : 4 secteurs différents dans le top 4
- **Mix tech/défensif** : NVDA (croissance) + WMT (stabilité)
- **Hits la limite** : 4 actifs à 10% (contrainte active)

**Pourquoi ces actifs ?**
- **NVDA** : Rendement historique exceptionnel (AI boom)
- **LLY** : Pharma stable avec croissance (Ozempic, etc.)
- **WMT** : Valeur défensive, dividendes réguliers
- **AEP** : Utilités = revenus prévisibles, faible corrélation avec tech

#### Allocation Sectorielle Max Sharpe

- Technology : 29.76% ⬆️ (momentum fort)
- Consumer Staples : 18.25% (stabilité)
- Health Care : 15.50% (croissance défensive)
- Industrials : 11.23%

**Interprétation** :
- **Biais croissance** : Tech + Healthcare = 45%
- **Équilibre** : Staples + Utilities pour atténuer la volatilité
- **Sous-pondération** : Energy, Materials (plus volatils)

#### Portefeuille Min Variance

**Top Holdings** :
1. VZ (10%) - Communication Services
2. JNJ (9%) - Health Care
3. WMT (8%) - Consumer Staples
4. KR (6%) - Consumer Staples

**Observations** :
- **Dominance défensive** : Consumer Staples = 36.48%
- **Actifs matures** : Grandes cap, dividendes stables
- **Faible exposition tech** : 6.09% seulement

**Pourquoi ces actifs ?**
- **VZ** : Télécoms = demande inélastique, cash-flows prévisibles
- **JNJ** : Conglomérat pharma/santé, très stable
- **WMT/KR** : Alimentaire = nécessité, récession-résilient

#### Allocation Sectorielle Min Variance

- Consumer Staples : 36.48% ⬆️ (très faible volatilité)
- Health Care : 18.78% (défensif)
- Communication : 14.95% (stabilité)
- Technology : 6.09% ⬇️ (trop volatil)

**Interprétation** :
- **Stratégie défensive pure** : 70% dans 3 secteurs stables
- **Évite la volatilité** : Sous-pondère Tech, Energy
- **Corrélations faibles** : Mix secteurs peu corrélés

### Analyse Comparative

| Métrique | Max Sharpe | Min Variance | Différence |
|----------|------------|--------------|------------|
| Rendement | 32.44% | 12.45% | **+20pp** |
| Risque | 19.35% | 12.91% | **+6.4pp** |
| Sharpe | 1.574 | 0.810 | **+0.76** |
| Gain/$1M | $324k | $125k | **+$199k** |

**Trade-off** :
- Max Sharpe : +160% de rendement mais +50% de risque
- Ratio risque/rendement favorable (Sharpe supérieur)

### Frontière Efficiente

**Forme de la courbe** :
- **Convexe** : Bénéfice décroissant de la prise de risque
- **Asymétrie** : Plus facile de réduire risque que d'augmenter rendement

**Points d'inflexion** :
- **Min Variance (gauche)** : Meilleure efficacité de diversification
- **Max Sharpe (haut-droite)** : Point optimal théorique
- **Au-delà** : Rendement croît moins vite que le risque

**Utilisation pratique** :
Un investisseur peut choisir n'importe quel point sur la courbe selon sa tolérance au risque :
- Conservateur → gauche (Min Var)
- Modéré → milieu
- Agressif → droite (Max Sharpe)

---

## Limites et Extensions Possibles

### Limites Actuelles

#### 1. Hypothèses Simplificatrices

**Distributions normales** : On suppose que les rendements suivent une loi normale.
- **Réalité** : Queues épaisses (black swans), asymétrie
- **Impact** : Sous-estime les risques extrêmes

**Stabilité temporelle** : Les paramètres (μ, Σ) sont constants.
- **Réalité** : Les corrélations changent (surtout en crise)
- **Impact** : Le portefeuille optimal peut devenir sous-optimal

#### 2. Limitations des Données

**Biais de survie** : Les actifs analysés ont survécu jusqu'en 2025.
- **Impact** : Surestime légèrement les rendements historiques

**Période historique** : 2015-2025 inclut COVID, tech boom.
- **Impact** : Peut ne pas représenter le futur

#### 3. Contraintes Simplifiées

**Pas de coûts de transaction** : Ignore les frais de trading.
- **Impact réel** : Réduit légèrement le rendement net

**Pas de contraintes fiscales** : Ignore les impôts sur plus-values.
- **Impact** : Le rendement net sera inférieur

### Extensions Possibles

#### 1. Modèles de Risque Avancés

**VaR/CVaR (Value-at-Risk)** :
```python
# Minimiser la perte maximale avec 95% de confiance
def cvar_objective(weights, returns, alpha=0.95):
    portfolio_returns = returns @ weights
    var = np.percentile(portfolio_returns, (1-alpha)*100)
    cvar = portfolio_returns[portfolio_returns <= var].mean()
    return -cvar
```

**Avantage** : Capture mieux les risques de queue.

#### 2. Contraintes Sectorielles

```python
# Limiter l'exposition par secteur (ex: max 30% en Tech)
for sector in sectors:
    constraints.append({
        'type': 'ineq',
        'fun': lambda w: 0.30 - sum(w[i] for i in sector_indices[sector])
    })
```

**Avantage** : Force une diversification sectorielle.

#### 3. Black-Litterman Model

**Idée** : Combiner les vues du marché (CAPM) avec vos propres prédictions.

```python
# Mélange rendements d'équilibre + vues personnelles
mu_bl = tau * Sigma @ inv(tau*Sigma + Omega) @ (P.T @ views) +
        inv(tau*Sigma + Omega) @ (tau*Sigma @ pi)
```

**Avantage** : Portefeuilles plus stables, moins sensibles aux erreurs d'estimation.

#### 4. Ré-équilibrage Dynamique

```python
# Ré-optimiser tous les trimestres
for quarter in quarters:
    weights = optimizer.optimize(data=data[:quarter])
    trades = rebalance(current_weights, weights)
    current_weights = weights
```

**Avantage** : S'adapte aux changements de marché.

#### 5. Optimisation Robuste

```python
# Pire cas dans un ensemble d'incertitude
def robust_objective(weights, mu_nominal, Sigma_nominal, uncertainty):
    mu_worst = mu_nominal - uncertainty
    return -portfolio_performance(weights, mu_worst, Sigma_nominal)[0]
```

**Avantage** : Protège contre l'incertitude des estimations.

#### 6. Contraintes ESG

```python
# Minimiser l'empreinte carbone
constraints.append({
    'type': 'ineq',
    'fun': lambda w: carbon_budget - w @ carbon_emissions
})
```

**Avantage** : Investissement responsable.

---

## Conclusion

### Points Clés

1. **Multi-critère** : Deux objectifs antagonistes (rendement vs risque) nécessitent des compromis
2. **Frontière efficiente** : Ensemble des portefeuilles Pareto-optimaux
3. **Diversification** : Clé de la réduction du risque (covariance < 1)
4. **Ratio de Sharpe** : Métrique universelle pour comparer des investissements
5. **Algorithme SLSQP** : Efficace pour résoudre des problèmes quadratiques avec contraintes

### Résultats Pratiques

- **Max Sharpe (1.574)** : Excellent pour investisseurs modérés à agressifs
- **Min Variance** : Idéal pour préservation du capital
- **Diversification sectorielle** : Crucial (répartition intelligente sur 11 secteurs)
- **Contrainte 10%** : Force la diversification, réduit le risque de concentration

### Applicabilité

Ce framework est applicable à :
- **Gestion individuelle** : Choix d'allocation pour un particulier
- **Gestion institutionnelle** : Base pour fonds d'investissement
- **Recherche académique** : Illustration de la théorie moderne du portefeuille
- **Outils pédagogiques** : Comprendre les trade-offs risque/rendement

### Recommandations d'Utilisation

1. **Ré-optimiser régulièrement** : Au moins trimestriellement
2. **Backtesting** : Tester sur des données out-of-sample
3. **Analyse de sensibilité** : Vérifier la robustesse aux paramètres
4. **Diversification géographique** : Étendre au-delà du marché US
5. **Conseil professionnel** : Consulter un conseiller financier pour décisions réelles

---

## Références

### Théorie

- **Markowitz, H. (1952)** : "Portfolio Selection", Journal of Finance
  - Base de la théorie moderne du portefeuille

- **Sharpe, W. (1966)** : "Mutual Fund Performance", Journal of Business
  - Introduction du ratio de Sharpe

- **Merton, R. (1972)** : "An Analytic Derivation of the Efficient Portfolio Frontier"
  - Résolution analytique pour portefeuilles

### Implémentation

- **SciPy Documentation** : scipy.optimize.minimize
- **NumPy Linear Algebra** : np.linalg
- **Pandas Time Series** : Manipulation de données financières

### Données

- **Yahoo Finance** : Source des données de marché
- **GICS Sectors** : Classification standard des secteurs

---

**Auteur** : Projet M1 Data - Mathématiques
**Date** : 2025
**Version** : 1.0
