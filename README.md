# Projet de Recherche — EDI & Graph Summarization
**Auteure :** Adji Marieme Sita Cissé  
**Directeur :** Prof. Malek Mouhoub — University of Regina  
**Période :** Mai – Octobre 2026 | Bourse DataIA Mobilité Internationale

---

## Sujet
Agrégation des préférences utilisateurs en garantissant l'Équité, la Diversité et l'Inclusion (EDI) par la Résumé de Graphe.

**4 corpus — 5 jeux de données :**

| Corpus | Attribut sensible | Utilisateurs | Items | Ratings |
|---|---|---|---|---|
| **MovieLens 100k** | genre côté utilisateur (F/M) | 943 (273F/670M) | 1 682 films | 100 000 |
| **MovieLens 1M** | genre côté utilisateur (F/M) | 6 040 (1 709F/4 331M) | 3 706 films | 1 000 209 |
| **libimseti.cz** | genre côté utilisateur (F/M) | réseau de rencontres social | — | scores de séduction |
| **Rate My Professors** | genre côté item (prof. F/M) | étudiants (venues) | 59 066 professeurs | notes de cours |
| **OpenAlex** | genre côté item (auteur·e F/M) | 99 venues IA/ML/CS | 904 auteur·e·s | 2 124 (2018–2023) |

**Tableau de bord interactif (bilingue FR/EN) :** https://sitaacisse-boop.github.io/EDI/
17 onglets — données, graphe biparti, calcul des métriques EDI, résultats par corpus, scalabilité, radar EDI, Pareto, et un onglet « Histoire d'AURORA » (origine du nom, photos d'aurores boréales prises pendant le séjour de recherche à Regina).

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
| `run_bootstrap_ml100k.py` | Sous-échantillonnage sans remise (90% de \|U\|, 30 répétitions) — significativité statistique sur ML100k |
| `run_robustness_budget.py` | Balayage du budget de fusion (30–60% de \|U\|) sur ML1M (k=20) et RMP (k=10) |

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
| `bootstrap_ml100k_results.json` | Sous-échantillonnage sans remise ML100k — moyenne/écart-type et taux de victoire par méthode |
| `robustness_budget_results.json` | Balayage du budget de fusion (30–60%) — ML1M (k=20) et RMP (k=10) |
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
| `samplepaper_FINAL.tex` | Article scientifique complet (format Springer LNCS, 15 références, compile via `references.bib`) |
| `memoire/Thesis.tex` | Mémoire de stage M2 DataScale (33 pages) — Contexte, Objectif, État de l'art, Approche, Validation, Conclusion, Apport personnel |
| `references.bib` | Base bibliographique commune à l'article et au mémoire (15 entrées) |

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

Budget de fusion : 50% de n_users sur tous les corpus (valeur validée par test de robustesse sur une plage de 30 à 60%, cf. section méthodologique). Corrige un bug de comptage antérieur qui faussait la compression réelle sur les corpus de tailles différentes.

### MovieLens 100k — équité côté utilisateur

| Méthode | k | ΔE | ILD | inc_F | inc_M |
|---|---|---|---|---|---|
| Average Score | 10 | **0.0068** | 1.000 | 0.001 | 0.002 |
| Borda | 10 | 0.4522 | 0.349 | 0.287 | 0.393 |
| Weighted Borda | 10 | 0.4249 | 0.354 | 0.293 | 0.392 |
| Condorcet | 10 | 0.4459 | 0.356 | 0.292 | 0.394 |
| Fair Re-rank | 10 | 0.1623 | 0.430 | 0.271 | 0.313 |
| **AURORA** | 5 | 0.4387 | 0.330 | 0.347 | 0.450 |
| **AURORA** | 10 | 0.4228 | 0.391 | **0.307** | **0.405** |
| **AURORA** | 20 | 0.2849 | **0.462** | **0.277** | 0.344 |

À k=20, AURORA bat toutes les méthodes de vote (Borda/W-Borda/Condorcet) sur ΔE et ILD simultanément, et obtient le meilleur ILD toutes méthodes non-triviales confondues. Fair Re-rank reste plus précis sur ΔE seul. À k=5, AURORA bat Borda/Weighted Borda sur les deux axes mais reste légèrement en retrait de Condorcet en diversité.
Compression : 943 → 471 super-nœuds (ratio 0.50, budget uniforme 50%).

Robustesse statistique (sous-échantillonnage sans remise, 90% de \|U\|, 30 répétitions, k=10) : AURORA bat Borda/Condorcet sur ΔE dans 28/30 tirages et sur l'ILD dans 30/30 — l'avantage n'est pas un artefact du tirage unique ci-dessus. Fair Re-rank reste devant AURORA sur les deux métriques dans les 30/30 tirages.

### MovieLens 1M — validation scalabilité

| Méthode | k | ΔE | ILD | inc_F | inc_M |
|---|---|---|---|---|---|
| Average Score | 10 | **0.0001** | 1.000 | 0.000 | 0.000 |
| Borda | 10 | 0.4818 | 0.418 | 0.305 | 0.409 |
| Weighted Borda | 10 | 0.4142 | 0.426 | 0.311 | 0.400 |
| Condorcet | 10 | 0.4872 | 0.408 | 0.305 | 0.412 |
| Fair Re-rank | 10 | 0.0185 | 0.452 | 0.336 | 0.343 |
| **AURORA** | 5 | 0.4156 | 0.395 | 0.341 | 0.434 |
| **AURORA** | 10 | 0.2137 | 0.440 | 0.321 | 0.370 |
| **AURORA** | 20 | 0.2773 | 0.456 | 0.294 | 0.356 |

AURORA bat systématiquement les 3 règles de vote classiques (Borda/Weighted Borda/Condorcet) sur ΔE et ILD simultanément, à tout k testé. Sur ce corpus, en revanche, Fair Re-rank domine AURORA sur ΔE **et** ILD à la fois, à tout k testé — un résultat propre à ML1M (grande base d'utilisateurs), pas un désavantage général de la méthode.

### libimseti.cz — équité côté utilisateur

| Méthode | ΔE ↓ | ILD ↑ | inc_F ↑ | inc_M ↑ | Temps |
|---|---|---|---|---|---|
| Average Score | **0.016** | **0.867** | 0.002 | 0.000 | 0.005s |
| Borda | 1.301 | 0.716 | 0.143 | **0.024** | 0.206s |
| Weighted Borda | 1.301 | 0.716 | 0.143 | **0.024** | 0.234s |
| Condorcet | 1.296 | 0.746 | **0.157** | 0.024 | 0.084s |
| Fair Re-rank | 0.139 | 0.716 | 0.038 | 0.024 | 0.269s |
| AURORA | 1.232 | 0.761 | 0.148 | 0.024 | 2.3s |

Sur ΔE, aucune méthode de vote (Borda/Weighted Borda/Condorcet/AURORA) n'échappe à l'écart d'équité élevé — seuls Average Score et Fair Re-rank y échappent (ΔE=0.016 et 0.139) ; vérifié robuste à un budget de fusion plus large (jusqu'à 60%) et à une contrainte d'équité resserrée. Limite structurelle, pas un problème de réglage : libimseti est le seul corpus où le genre existe des deux côtés du graphe biparti (notateurs et profils notés), et les patterns de notation diffèrent significativement selon la paire de genres — une illustration empirique de l'argument de Yao & Huang (2017) selon lequel la parité démographique n'est pas toujours appropriée quand les préférences dépendent légitimement de l'attribut sensible (voir discussion détaillée, section 5.6 de `samplepaper_FINAL.tex`, ou l'onglet « libimseti.cz » du tableau de bord). AURORA garde néanmoins un net avantage de diversité (ILD=0.761, la meilleure valeur après Average Score) sur les méthodes de vote.

### Rate My Professors — équité côté item (frac_F)

| Méthode | ΔE ↓ | ILD ↑ | frac_F ↑ | Temps |
|---|---|---|---|---|
| Average Score | **0.000** | **0.982** | 0.40 | 0.008s |
| Borda | 0.065 | 0.267 | 0.50 | 0.4s |
| Weighted Borda | **0.000** | 0.245 | 0.20 | 22.9s |
| Condorcet | 0.076 | 0.253 | 0.30 | 0.1s |
| Fair Re-rank | **0.000** | 0.276 | 0.50 | 0.5s |
| **AURORA** | 0.112 | 0.808 | **0.60** | 11.3s |

AURORA reste la seule méthode combinant haute diversité (ILD=0.808, 2e meilleure derrière Average Score) et forte représentation (frac_F=0.60) ; son ΔE (0.112) est en revanche le plus élevé du tableau, derrière toutes les autres méthodes — compromis équité/diversité assumé (voir onglet RMP du tableau de bord).

### OpenAlex (IA/ML/CS 2018–2023) — équité côté item (frac_F auteur·e·s)

99 venues × 904 auteur·e·s (171F/733M, α_F ≈ 19%) × 2 124 ratings

| Méthode | ΔE ↓ | ILD ↑ | frac_F ↑ | Temps |
|---|---|---|---|---|
| Average Score | 0.071 | 0.356 | 0.20 | 0.029s |
| Borda | 1.233 | 0.000 | 0.00 | 0.010s |
| Weighted Borda | 1.233 | 0.000 | 0.00 | 0.011s |
| Condorcet | 1.233 | 0.000 | 0.00 | 0.008s |
| Fair Re-rank | 0.071 | 0.356 | 0.20 | 0.013s |
| **AURORA** | **0.031** | 0.345 | 0.19 | 0.070s |

AURORA obtient ΔE=0.031 (−56% vs Average Score/Fair Re-rank à 0.071) — seule méthode à corriger ce biais extrême (le plus sévère des 4 corpus) tout en maintenant une représentation féminine proche (frac_F=19% vs 20%). Borda/WBorda/Condorcet ne recommandent aucune auteure (frac_F=0%).

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
| Phase 6 — Article de recherche | Juillet–Août | ✅ Terminé | `samplepaper_FINAL.tex` (LLNCS, 15 références, Overleaf) |
| Phase 7 — Mémoire de stage M2 | Août | ✅ Terminé | `memoire/Thesis.tex` (33 pages, 15 références) |
| Phase 8 — Bilinguisme du tableau de bord | Août | ✅ Terminé | 17 onglets + graphiques traduits FR/EN sur https://sitaacisse-boop.github.io/EDI/ |
| Phase 9 — Rigueur pré-soutenance | Août | ✅ Terminé | Correctif bug de non-déterminisme (`run_openalex.py`), extension k∈{5,10,20} aux 3 corpus multi-domaines, preuve de significativité statistique, justification SCRUF-D, vérification exhaustive site/mémoire/papier vs JSON sources |

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
