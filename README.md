# Projet de Recherche — EDI & Graph Summarization
**Auteure :** Adji Marieme Sita Cissé  
**Directeur :** Prof. Malek Mouhoub — Université Paris-Saclay · University of Regina  
**Période :** Mai – Octobre 2026 | BAURORAe DataIA Mobilité Internationale

---

## Sujet
Agrégation des préférences utilisateurs en garantissant l'Équité, la Diversité et l'Inclusion (EDI) par la Résumé de Graphe.

**4 corpus — 5 jeux de données :**

| Corpus | Attribut sensible | Utilisateurs | Items | Ratings |
|---|---|---|---|---|
| **MovieLens 100k** | genre côté utilisateur (F/M) | 943 (273F/670M) | 1 682 films | 100 000 |
| **MovieLens 1M** | genre côté utilisateur (F/M) | 6 040 (1 709F/4 331M) | 3 706 films | 1 000 209 |
| **libimseti.cz** | genre côté utilisateur (F/M) | réseau de rencontres social | — | scores de séduction |
| **Rate My Professors** | genre côté item (prof. F/M) | étudiants (venues) | 18 000+ professeurs | notes de cours |
| **OpenAlex** | genre côté item (auteur·e F/M) | 99 venues IA/ML/CS | 904 auteur·e·s | 2 124 (2018–2023) |

**Tableau de bord interactif :** https://sitaacisse-boop.github.io/EDI/

---

## Contenu du dossier

### Code principal
| Fichier | Description |
|---|---|
| `edi_coarsening.py` | Algorithme de coarsening EDI-contraint — cœur de la méthode proposée |
| `edi_baselines.py` | 6 baselines : Average Score, Borda, Weighted Borda, Condorcet, Fair Re-rank, AURORA |
| `run_all_experiments.py` | Lance toutes les expériences 100k (6 méthodes × k ∈ {5,10,20} × θ ∈ {3.5, 4.0}) |
| `run_experiments_1m.py` | Lance toutes les expériences MovieLens 1M |
| `run_libimseti.py` | Expériences sur libimseti.cz — équité côté utilisateur |
| `run_rmp.py` | Expériences sur Rate My Professors — équité côté item (frac_F) |
| `run_openalex.py` | Expériences sur OpenAlex — équité côté item, genre des auteur·e·s recommandé·e·s |
| `run_add_wborda.py` | Ajoute Weighted Borda aux JSONs existants sans relancer le coarsening |
| `run_sensitivity.py` | Analyse de sensibilité sur les paramètres ε_E, ε_D, max_merges |
| `run_scalability.py` | Scalabilité 1M : sous-ensembles stratifiés 500→6 040 utilisateurs, k=10, θ=4.0 |
| `run_scalability_fine.py` | Scalabilité fine 100k : 100→943 utilisateurs — courbe granulaire |

### Résultats
| Fichier | Description |
|---|---|
| `experiments_results.json` | Résultats 100k — 6 méthodes × k ∈ {5,10,20} (ΔE, ILD, inclusion) |
| `experiments_results_1m.json` | Résultats 1M — 6 méthodes × k ∈ {5,10,20} |
| `libimseti_results.json` | Résultats libimseti — 6 méthodes × k=10 (ΔE, ILD, inc_F, inc_M) |
| `rmp_results.json` | Résultats Rate My Professors — 6 méthodes × k=10 (ΔE, ILD, frac_F) |
| `openalex_results.json` | Résultats OpenAlex — 6 méthodes × k=10 (ΔE, ILD, frac_F) |
| `openalex_raw_cache.json` | Cache brut OpenAlex API (45 008 papers) — évite re-fetch |
| `scalability_results.json` | Scalabilité MovieLens 1M — 6 méthodes × 5 tailles (500, 1k, 2k, 4k, 6 040) |
| `scalability_100k_results.json` | Scalabilité fine-grained ML-100k — 6 méthodes × 8 tailles (100→943 utilisateurs) |
| `sensitivity_results.json` | Résultats analyse de sensibilité |
| `fig_edi_baselines.png` | Figure comparative : 6 méthodes × k ∈ {5,10,20} (100k) |
| `fig_comparison_100k_1m.png` | Comparaison 100k vs 1M — scalabilité |
| `fig_data_overview.png` | Vue d'ensemble du dataset MovieLens 100k |

### Notebook
| Fichier | Description |
|---|---|
| `edi_baselines.ipynb` | Notebook complet : chargement, baselines, métriques EDI, figures, tables LaTeX — couvre les 4 corpus |

### Documents de référence
| Fichier | Description |
|---|---|
| `Phase2_ModelSpec_FR.pdf` | Spécification du modèle — version française |
| `Phase2_ModelSpec_EN.pdf` | Model specification — English version |
| `rechercheplan_pdf.pdf` | Plan de recherche complet (6 phases, Mai–Octobre 2026) |
| `Bibliographie_annotee_Phase1.pdf` | Bibliographie annotée — 13 références classées en 4 axes |

---

## Métriques EDI

| Métrique | Formule | Cible |
|---|---|---|
| **ΔE** (équité) | `\|utility_F(R) − utility_M(R)\|` | ≈ 0 |
| **ILD** (diversité intra-liste) | dissimilarité cosinus pairwise moyenne | élevé |
| **inclusion_g** | fraction d'items du top-k notés ≥ θ par le groupe g | ≥ α_g |
| **frac_F** | fraction d'items féminins dans le top-k (item-side equity) | élevé |

Paramètres MovieLens : seuil θ = 4.0, α_F = 0.289, α_M = 0.711

---

## Résultats principaux (θ = 4.0, k = 10 sauf indication)

### MovieLens 100k — équité côté utilisateur

| Méthode | k | ΔE | ILD | inc_F | inc_M |
|---|---|---|---|---|---|
| Average Score | 10 | **0.0068** | 1.000 | 0.001 | 0.002 |
| Borda | 10 | 0.4249 | 0.354 | 0.293 | 0.392 |
| Weighted Borda | 10 | 0.4249 | 0.354 | 0.293 | 0.392 |
| Condorcet | 10 | 0.4459 | 0.356 | 0.292 | 0.394 |
| Fair Re-rank | 10 | 0.1623 | 0.430 | 0.271 | 0.313 |
| **AURORA** | 10 | 0.3511 | **0.448** | 0.248 | 0.325 |
| **AURORA** | 20 | 0.3475 | **0.465** | **0.269** | **0.348** |

À k=20 (100k), notre méthode domine sur ILD, inc_F et inc_M.  
Compression : 943 → 272 super-nœuds (−71.2%), ratio F/M rééquilibré 29% → 37%.

### MovieLens 1M — validation scalabilité

| Méthode | k | ΔE | ILD | inc_F | inc_M |
|---|---|---|---|---|---|
| Average Score | 10 | **0.0001** | 1.000 | 0.000 | 0.000 |
| Borda | 10 | 0.538 | 0.402 | 0.302 | 0.417 |
| Weighted Borda | 10 | 0.371 | 0.413 | 0.324 | 0.404 |
| Condorcet | 10 | 0.487 | 0.408 | 0.305 | 0.412 |
| Fair Re-rank | 10 | 0.019 | 0.452 | 0.336 | 0.343 |
| **AURORA** | 5 | **0.064** | **0.578** | 0.277 | 0.295 |
| **AURORA** | 10 | 0.332 | 0.423 | 0.317 | 0.390 |

Résultat marquant : AURORA k=5 sur 1M atteint ΔE=0.064 + ILD=0.578 (meilleure diversité de toutes les méthodes).

### libimseti.cz — équité côté utilisateur

| Méthode | ΔE ↓ | ILD ↑ | inc_F ↑ | inc_M ↑ | Temps |
|---|---|---|---|---|---|
| Average Score | **0.016** | **0.867** | 0.002 | 0.000 | 0.009s |
| Borda | 1.218 | 0.744 | **0.147** | **0.024** | 0.667s |
| Weighted Borda | 1.218 | 0.744 | **0.147** | **0.024** | 0.705s |
| Condorcet | 1.296 | 0.746 | 0.157 | 0.024 | 0.257s |
| Fair Re-rank | 0.144 | 0.719 | 0.039 | 0.024 | 0.807s |
| **AURORA** | 0.876 | 0.816 | 0.112 | 0.024 | 402.5s |

### Rate My Professors — équité côté item (frac_F)

| Méthode | ΔE ↓ | ILD ↑ | frac_F ↑ | Temps |
|---|---|---|---|---|
| Average Score | **0.000** | **0.982** | 0.40 | 0.012s |
| Borda | 0.008 | 0.145 | 0.40 | 1.308s |
| Weighted Borda | 0.008 | 0.145 | 0.40 | 41.3s |
| Condorcet | 0.076 | 0.253 | 0.30 | 0.289s |
| Fair Re-rank | **0.000** | 0.196 | 0.50 | 1.455s |
| **AURORA** | 0.017 | 0.924 | **0.60** | 46.6s |

AURORA est la seule méthode à atteindre ILD ≈ 0.924 (proche d'Average Score) tout en augmentant frac_F à 60%.

### OpenAlex (IA/ML/CS 2018–2023) — équité côté item (frac_F auteur·e·s)

99 venues × 904 auteur·e·s (171F/733M, α_F ≈ 19%) × 2 124 ratings

| Méthode | ΔE ↓ | ILD ↑ | frac_F ↑ | Temps |
|---|---|---|---|---|
| Average Score | 0.071 | 0.356 | 0.20 | 0.029s |
| Borda | 1.233 | 0.000 | 0.00 | 0.010s |
| Weighted Borda | 1.233 | 0.000 | 0.00 | 0.011s |
| Condorcet | 1.233 | 0.000 | 0.00 | 0.008s |
| Fair Re-rank | 0.071 | 0.356 | 0.20 | 0.013s |
| **AURORA** | **0.051** | 0.351 | **0.20** | 0.144s |

AURORA obtient ΔE=0.051 (vs 0.071 pour Average Score) — amélioration de l'équité inter-groupes tout en maintenant la représentation féminine (frac_F=20%). Borda/WBorda/Condorcet ne recommandent aucune auteure (frac_F=0%).

---

## Avancement par phase

| Phase | Mois | Statut | Livrable |
|---|---|---|---|
| Phase 1 — Revue de littérature | Mai | ✅ Terminé | Bibliographie annotée |
| Phase 2 — Conception du modèle | Juin | ✅ Terminé | `Phase2_ModelSpec_FR.pdf` |
| Phase 3 — Développement algorithme | Juillet | ✅ Terminé | `edi_coarsening.py` |
| Phase 4 — Implémentation & baselines | Juillet | ✅ Terminé | `edi_baselines.py`, `run_all_experiments.py` |
| Phase 5 — Évaluation MovieLens | Juillet | ✅ Terminé | `experiments_results.json`, figures, tables, scalabilité |
| Phase 5b — Scalabilité | Juillet | ✅ Terminé | `scalability_results.json` (1M, 500→6 040) + `scalability_100k_results.json` (100k, 100→943) |
| Phase 5c — Corpus multi-domaines | Juillet | ✅ Terminé | `libimseti_results.json`, `rmp_results.json`, `openalex_results.json` |
| Phase 6 — Article de recherche | Juillet–Août | ⏳ En cours | Paper LLNCS (Overleaf) |

---

## Environnement technique

- **Python 3.11** — NumPy, Pandas, NetworkX, Matplotlib
- **pyalex** — client OpenAlex API (pagination, rate-limit)
- **gender-guesser** — inférence du genre des auteur·e·s par prénom
- **Jupyter Notebook** — `edi_baselines.ipynb`
- **LaTeX / Overleaf** — format LLNCS Springer
- **Datasets** — MovieLens 100k (`data/ml-100k/`), MovieLens 1M (`data/ml-1m/`), libimseti, Rate My Professors, OpenAlex API

---

## Question de recherche
> Étant donné un graphe biparti pondéré G = (U ∪ I, E, w, s), comment construire un résumé G' via coarsening EDI-contraint qui réduit la taille tout en bornant la dégradation des métriques d'équité (ΔE), de diversité (ILD) et d'inclusion, comparé aux règles d'agrégation classiques (Average Score, Borda, Condorcet, Fair Re-rank) — et ce sur des domaines variés (films, rencontres, éducation, recherche académique) ?
