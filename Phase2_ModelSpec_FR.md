# Phase 2 — Spécification du Modèle
**Projet :** Agrégation des préférences utilisateurs en garantissant l'Équité, la Diversité et l'Inclusion par la Résumé de Graphe  
**Auteure :** Adji Marieme Sita Cissé  
**Directeur :** Prof. Malek Mouhoub — Université de Regina  
**Date :** Juin 2026

---

## 1. Modèle de Graphe de Préférences

### 1.1 Définition Formelle

Les données de préférences sont représentées sous forme de **graphe biparti pondéré et attribué** :

> **G = (U ∪ I, E, w, s)**

| Symbole | Définition |
|---|---|
| U = {u₁, u₂, …, u_m} | Ensemble des nœuds utilisateurs |
| I = {i₁, i₂, …, i_n} | Ensemble des nœuds items (ex. films dans MovieLens) |
| E ⊆ U × I | Ensemble des arêtes — (u, i) ∈ E si et seulement si u a noté i |
| w : E → [0,5 ; 5] | Fonction de poids — w(u, i) = note attribuée par u à i |
| s : U → {F, H} | Attribut sensible — genre de chaque utilisateur (Femme / Homme) |

**Propriété clé (biparti) :** Aucune arête n'existe entre deux utilisateurs ou entre deux items. Seules les arêtes utilisateur–item existent.

### 1.2 Choix de l'Attribut Sensible

**Attribut retenu : le genre (côté utilisateur)**

Justification :
- Directement lié à la question de recherche : le classement collectif représente-t-il équitablement les utilisatrices et les utilisateurs ?
- Le genre est un champ binaire propre dans MovieLens (100k, 1M), sans pré-traitement nécessaire
- Toutes les métriques de référence (Yao & Huang 2017, Leonhardt et al. 2018) utilisent le genre côté utilisateur → comparabilité directe avec la littérature
- **Limite à mentionner dans le mémoire :** l'encodage binaire F/H ne reflète pas la diversité de genre réelle ; c'est un point de départ méthodologique à élargir

### 1.3 Exemple Jouet

4 utilisateurs, 3 items, attribut de genre :

| Utilisateur | Genre | i₁ | i₂ | i₃ |
|---|---|---|---|---|
| u₁ | F | 5 | 2 | — |
| u₂ | F | 4 | — | 1 |
| u₃ | H | — | 5 | 4 |
| u₄ | H | 1 | — | 5 |

Groupes : G_F = {u₁, u₂}, G_H = {u₃, u₄}

---

## 2. Du Graphe au Classement Collectif

### 2.1 Paradigme Retenu : Notes pour les métriques EDI, Rangs pour les baselines

**Les métriques EDI sont ancrées dans les notes (valeurs brutes)** — justification :
- MovieLens fournit des notes directement ; aucune conversion nécessaire
- Préserve l'intensité des préférences (« j'adore » ≠ « bof »)
- Correspond exactement aux métriques de Yao & Huang disponibles dans la bibliographie

**Les baselines utilisent les rangs** — justification :
- Borda et Condorcet sont des méthodes à rangs par définition
- Permet de comparer « agrégation par notes » vs « agrégation par rangs » — résultat expérimental en soi

### 2.2 Méthodes d'Agrégation de Référence (baselines)

| Méthode | Formule | Problème EDI connu |
|---|---|---|
| Score moyen | score(i) = (1/\|U_i\|) Σ_{u:(u,i)∈E} w(u,i) | Favorise les items populaires auprès du groupe majoritaire |
| Borda | score_B(i) = Σ_u rang_u(i) | Biaisé si les groupes sont de tailles inégales |
| Condorcet | i ≻ j si \|{u : w(u,i)>w(u,j)}\| > \|U\|/2 | La majorité peut systématiquement écraser la minorité |

Aucune de ces méthodes ne contrôle explicitement l'EDI pendant l'agrégation — ce manque est la motivation du modèle proposé.

### 2.3 Exemple Jouet — Agrégation

**Score moyen :**
- i₁ : (5+4+1)/3 = 3,33
- i₂ : (2+5)/2 = 3,50
- i₃ : (1+4+5)/3 = 3,33

Classement collectif : i₂ > i₁ ≈ i₃

**Classements par groupe :**
- G_F préfère : i₁ > i₂ > i₃
- G_H préfère : i₃ > i₂ (i₁ mal noté)
- Global : i₂ en tête — semble neutre, mais **i₁ (adoré des femmes) est rétrogradé**

→ Le classement global n'est pas équitable : les préférences des femmes sont sous-représentées.

---

## 3. Métriques EDI Formelles

Soit R = (i₁, …, i_k) le top-k collectif produit. Soient G_F, G_H les deux groupes d'utilisateurs.

### 3.1 E — Équité (entre groupes d'utilisateurs)

```
utilité_g(R) = (1/|G_g|) × Σ_{u∈G_g} Σ_{i∈R} w(u,i) / k

ΔE = |utilité_F(R) − utilité_H(R)|
```

- **Interprétation :** Satisfaction moyenne de chaque groupe vis-à-vis du top-k collectif
- **Cible :** ΔE ≈ 0 (les deux groupes sont également satisfaits)
- **Source :** Yao & Huang (2017) — métrique de value unfairness

### 3.2 D — Diversité (variété des items recommandés)

```
ILD(R) = (2 / k(k−1)) × Σ_{i≠j ∈ R} dist(i, j)

dist(i, j) = 1 − similarité_cosinus(v_i, v_j)
```

où v_i ∈ ℝ^m est le vecteur de notes de l'item i sur l'ensemble des utilisateurs (0 si non noté).

- **Interprétation :** Dissimilarité pairwise moyenne au sein du top-k
- **Cible :** ILD élevé (recommandations variées, pas tous le même genre)
- **Paramètre à justifier :** similarité cosinus sur vecteurs de notes — alternative : distance basée sur les genres de films

### 3.3 I — Inclusion (représentation de la minorité dans les résultats)

```
inclusion_g(R) = (1/|G_g|) × Σ_{u∈G_g} |{i ∈ R : w(u,i) ≥ θ}| / k

θ = note minimale considérée comme "pertinente" (ex. θ = 4,0)
α_g = |G_g| / |U|   (part de la population du groupe g)
```

- **Interprétation :** Part des items du top-k jugés pertinents par les membres du groupe g
- **Cible (équité proportionnelle — Kellerhals & Peters 2024) :**
  Chaque groupe doit recevoir au moins sa part proportionnelle d'items pertinents :
  `inclusion_g(R) ≥ α_g`
  Si les femmes représentent 30 % des utilisateurs, au moins 30 % des items du top-k doivent leur convenir.
  Ce critère s'appuie sur la notion de « core » de l'équité proportionnelle : un groupe de taille α_g mérite au moins α_g de l'outcome collectif.
- **Paramètre à justifier :** choix de θ (à valider expérimentalement ; θ ∈ {3,5 ; 4,0} seront comparés)

---

## 4. Contribution Proposée : Résumé de Graphe sous Contrainte EDI

### 4.1 Principe

Lors du coarsening du graphe (Phase 3), au lieu de fusionner les nœuds librement pour minimiser l'erreur structurelle, on ajoute une **contrainte de préservation de l'EDI** :

> Ne pas fusionner deux supernœuds s_a, s_b si leur fusion provoque :
> - une augmentation de ΔE au-delà de ε_E, OU
> - une diminution de ILD(R) au-delà de ε_D, OU
> - une diminution de inclusion_F(R) au-delà de ε_I

### 4.2 Formulation de l'Objectif (à affiner en Phase 3)

```
min  perte_structurelle(G, G')       [objectif standard du coarsening]
s.c. ΔE(G') ≤ ΔE(G) + ε_E
     ILD(G') ≥ ILD(G) − ε_D
     inclusion_F(G') ≥ inclusion_F(G) − ε_I
```

où G' est le graphe résumé et ε_E, ε_D, ε_I sont les tolérances de dégradation EDI.

### 4.3 Question de Recherche (formulation candidate)

> Étant donné un graphe biparti pondéré de préférences G = (U ∪ I, E, w, s), comment construire un résumé G' (via détection de communautés et coarsening) qui réduit la taille tout en bornant la dégradation des métriques d'équité (ΔE), de diversité (ILD) et d'inclusion, comparé aux règles d'agrégation classiques (Borda, Condorcet, score moyen) ?

---

## 5. Prochaines Étapes

| Phase | Action | Livrable |
|---|---|---|
| Phase 2 (Juin) | Valider le modèle sur l'exemple jouet en Python/NetworkX | Modèle vérifié + code des métriques |
| Phase 3 (Juillet) | Concevoir l'algorithme de coarsening sous contrainte EDI | Description de l'algorithme + complexité |
| Phase 4 (Août) | Implémenter sur MovieLens 100k | Prototype fonctionnel |
| Phase 5 (Sept.) | Comparer ΔE, ILD, inclusion vs baselines | Résultats expérimentaux |

---

## 6. Paramètres Ouverts (à documenter dans le mémoire)

| Paramètre | Rôle | Justification nécessaire |
|---|---|---|
| θ | Seuil de pertinence pour la métrique d'inclusion | Comparer θ ∈ {3,5 ; 4,0} expérimentalement |
| k | Taille du top-k | Valeurs standard : k ∈ {5, 10, 20} |
| ε_E, ε_D, ε_I | Tolérances de dégradation EDI | Fixer via mesure baseline sur MovieLens |
| dist(i,j) | Distance entre items pour l'ILD | Cosinus sur vecteurs de notes vs distance basée sur les genres |

---

*Document à réviser avec Prof. Mouhoub — Juin 2026*
