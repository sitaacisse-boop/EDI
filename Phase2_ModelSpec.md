# Phase 2 — Model Design Specification
**Project:** Aggregating User Preferences while Ensuring Equity, Diversity, and Inclusion using Graph Summarization  
**Author:** Adji Marieme Sita Cissé  
**Supervisor:** Prof. Malek Mouhoub — University of Regina  
**Date:** June 2026

---

## 1. Graph-Based Preference Model

### 1.1 Formal Definition

The preference data is represented as a **weighted attributed bipartite graph**:

> **G = (U ∪ I, E, w, s)**

| Symbol | Definition |
|---|---|
| U = {u₁, u₂, …, u_m} | Set of user nodes |
| I = {i₁, i₂, …, i_n} | Set of item nodes (e.g., movies in MovieLens) |
| E ⊆ U × I | Set of edges — an edge (u, i) ∈ E iff user u rated item i |
| w : E → [0.5, 5] | Weight function — w(u, i) = rating given by u to i |
| s : U → {F, H} | Sensitive attribute — gender of each user (Female / Male) |

**Key property (bipartite):** No edges exist between two users or two items. Only user–item edges exist.

### 1.2 Sensitive Attribute Choice

**Chosen attribute: gender (côté utilisateur)**

Rationale:
- Directly tied to the research question: does the collective ranking fairly represent female and male users?
- Gender is a clean binary field in MovieLens (100k, 1M), requiring no pre-processing
- All reference metrics (Yao & Huang 2017, Leonhardt et al. 2018) use gender on users → direct comparability with literature
- Limitations to acknowledge in the paper: the binary F/H encoding does not reflect gender diversity in reality; this is a methodological starting point to be extended

### 1.3 Toy Example

4 users, 3 items, genre attribute:

| User | Genre | i₁ | i₂ | i₃ |
|---|---|---|---|---|
| u₁ | F | 5 | 2 | — |
| u₂ | F | 4 | — | 1 |
| u₃ | H | — | 5 | 4 |
| u₄ | H | 1 | — | 5 |

Groups: G_F = {u₁, u₂}, G_H = {u₃, u₄}

---

## 2. From Graph to Collective Ranking

### 2.1 Paradigm Choice: Notes (ratings) for EDI metrics, Ranks for baselines

**EDI metrics are anchored in notes (rating values)** — rationale:
- MovieLens provides raw ratings; no conversion needed
- Preserves preference intensity ("love" ≠ "like")
- Directly matches Yao & Huang's metrics

**Baselines use ranks** — rationale:
- Borda and Condorcet are rank-based by design
- Allows comparison between "rating aggregation" vs "rank aggregation" — itself an interesting experimental result

### 2.2 Baseline Aggregation Methods (comparison targets)

| Method | Formula | Known EDI problem |
|---|---|---|
| Average score | score(i) = (1/\|U_i\|) Σ_{u:(u,i)∈E} w(u,i) | Favors items popular within the majority group |
| Borda | score_B(i) = Σ_u rank_u(i) | Biased when group sizes are unequal |
| Condorcet | i ≻ j if \|{u: w(u,i)>w(u,j)}\| > \|U\|/2 | Majority can systematically override minority |

None of these methods explicitly controls EDI during aggregation — this gap is the motivation for the proposed approach.

### 2.3 Toy Example — Aggregation

**Average score:**
- i₁: (5+4+1)/3 = 3.33
- i₂: (2+5)/2 = 3.50
- i₃: (1+4+5)/3 = 3.33

Collective ranking: i₂ > i₁ ≈ i₃

**Group rankings:**
- G_F prefers: i₁ > i₂ > i₃
- G_H prefers: i₃ > i₂ (i₁ rated low)
- Global: i₂ first — appears neutral, but **i₁ (strongly preferred by women) is demoted**

→ The global ranking is not equitable: women's preferences are underrepresented.

---

## 3. Formal EDI Metrics

Let R = (i₁, …, i_k) be the collective top-k ranking. Let G_F, G_H be the two user groups.

### 3.1 E — Equity (fairness between user groups)

```
utility_g(R) = (1/|G_g|) × Σ_{u∈G_g} Σ_{i∈R} w(u,i) / k

ΔE = |utility_F(R) − utility_H(R)|
```

- **Interpretation:** Average satisfaction of each group with the collective top-k
- **Target:** ΔE ≈ 0 (both groups equally satisfied)
- **Source:** Yao & Huang (2017) — value unfairness metric

### 3.2 D — Diversity (variety of recommended items)

```
ILD(R) = (2 / k(k−1)) × Σ_{i≠j ∈ R} dist(i, j)

dist(i, j) = 1 − cosine_sim(v_i, v_j)
```

where v_i ∈ ℝ^m is the rating vector of item i across all users (0 if unrated).

- **Interpretation:** Average pairwise dissimilarity within the top-k list
- **Target:** ILD high (varied recommendations, not all same genre)
- **Parameter to justify:** cosine similarity on rating vectors — alternative: genre-based distance

### 3.3 I — Inclusion (minority representation in results)

```
inclusion_g(R) = (1/|G_g|) × Σ_{u∈G_g} |{i ∈ R : w(u,i) ≥ θ}| / k

θ = minimum rating considered "relevant" (e.g., θ = 4.0)
α_g = |G_g| / |U|   (population share of group g)
```

- **Interpretation:** Fraction of top-k items that group g members actually find relevant
- **Target (proportional fairness — Kellerhals & Peters 2024):**
  Each group should receive at least its proportional share of relevant items:
  `inclusion_g(R) ≥ α_g`
  i.e., if women represent 30% of users, at least 30% of top-k items should be relevant to them.
  This is grounded in the "core" concept of proportional fairness: a group of size α_g deserves at least α_g of the collective outcome.
- **Note:** This replaces the incorrect formulation `inclusion_F ≥ inclusion_H × α` which would have allowed the minority to be systematically less well served.
- **Parameter to justify:** choice of θ (to be validated experimentally; 3.5 and 4.0 will be compared)

---

## 4. Proposed Contribution: EDI-Constrained Graph Summarization

### 4.1 Principle

During graph coarsening (Phase 3), instead of merging nodes freely to minimize structural error, add an **EDI preservation constraint**:

> Do not merge two supernodes s_a, s_b if their fusion causes:
> - ΔE to increase beyond ε_E, OR
> - ILD(R) to decrease beyond ε_D, OR
> - inclusion_F(R) to decrease beyond ε_I

### 4.2 Formal Objective (to be refined in Phase 3)

```
min  structural_loss(G, G')          [standard coarsening objective]
s.t. ΔE(G') ≤ ΔE(G) + ε_E
     ILD(G') ≥ ILD(G) − ε_D
     inclusion_F(G') ≥ inclusion_F(G) − ε_I
```

where G' is the summarized graph and ε_E, ε_D, ε_I are degradation tolerances.

### 4.3 Research Question (candidate formulation)

> Given a weighted bipartite preference graph G = (U ∪ I, E, w, s), how can we construct a summary G' (via community detection and coarsening) that reduces size while bounding the degradation of equity (ΔE), diversity (ILD), and inclusion metrics, compared to classical aggregation rules (Borda, Condorcet, average score)?

---

## 5. Next Steps

| Phase | Action | Deliverable |
|---|---|---|
| Phase 2 (June) | Validate model on toy example in Python/NetworkX | Verified model + metrics code |
| Phase 3 (July) | Design EDI-constrained coarsening algorithm | Algorithm description + complexity |
| Phase 4 (August) | Implement on MovieLens 100k | Functional prototype |
| Phase 5 (Sept.) | Compare ΔE, ILD, inclusion vs baselines | Experimental results |

---

## 6. Open Parameters (to document in paper)

| Parameter | Role | Justification needed |
|---|---|---|
| θ | Relevance threshold for inclusion metric | Compare θ ∈ {3.5, 4.0} experimentally |
| k | Top-k size | Standard values: k ∈ {5, 10, 20} |
| ε_E, ε_D, ε_I | EDI degradation tolerances | Set via baseline measurement on MovieLens |
| dist(i,j) | Item distance for ILD | Cosine on rating vectors vs genre-based distance |

---

