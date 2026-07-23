# Projet de Recherche — EDI & Graph Summarization
**Auteure :** Adji Marieme Sita Cissé  
**Directeur :** Prof. Malek Mouhoub — Université de Regina  
**Période :** Mai – Octobre 2026 | Bourse DataIA Mobilité Internationale

---

## Sujet
Agrégation des préférences utilisateurs en garantissant l'Équité, la Diversité et l'Inclusion (EDI) par la Résumé de Graphe.  
Données : MovieLens 100k — 943 utilisateurs (273 F / 670 M), 1 682 films, 100 000 notes (1–5).

**Tableau de bord interactif :** https://sitaacisse-boop.github.io/EDI/

---

## Contenu du dossier

### Code principal
| Fichier | Description |
|---|---|
| `edi_coarsening.py` | Algorithme de coarsening EDI-contraint — cœur de la méthode proposée |
| `edi_baselines.py` | 4 baselines : Average Score, Borda, Condorcet, Fair Re-rank |
| `run_all_experiments.py` | Lance toutes les expériences (5 méthodes × k ∈ {5,10,20} × θ ∈ {3.5, 4.0}) |
| `run_sensitivity.py` | Analyse de sensibilité sur les paramètres ε_E, ε_D, max_merges |

### Résultats
| Fichier | Description |
|---|---|
| `experiments_results.json` | Résultats complets — source unique de vérité (ΔE, ILD, inclusion) |
| `sensitivity_results.json` | Résultats analyse de sensibilité |
| `fig_edi_baselines.png` | Figure comparative : 5 méthodes × k ∈ {5,10,20} |
| `fig_data_overview.png` | Vue d'ensemble du dataset MovieLens 100k |

### Notebook
| Fichier | Description |
|---|---|
| `edi_baselines.ipynb` | Notebook complet : chargement, baselines, métriques EDI, figures, tables LaTeX |

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

Paramètres : seuil θ = 4.0, α_F = 0.289, α_M = 0.711

---

## Résultats principaux (θ = 4.0)

| Méthode | k | ΔE | ILD | inc_F | inc_M |
|---|---|---|---|---|---|
| Average Score | 10 | **0.0068** | 1.000 | 0.001 | 0.002 |
| Borda | 10 | 0.4249 | 0.354 | 0.293 | 0.392 |
| Condorcet | 10 | 0.4459 | 0.356 | 0.292 | 0.394 |
| Fair Re-rank | 10 | 0.1623 | 0.430 | 0.271 | 0.313 |
| **Ours** | 10 | 0.3511 | **0.448** | 0.248 | 0.325 |
| **Ours** | 20 | 0.3475 | **0.465** | **0.269** | **0.348** |

À k=20, notre méthode est la seule à dominer simultanément sur ILD, inc_F et inc_M.  
Compression : 943 → 272 super-nœuds (−71.2%), ratio F/M rééquilibré 29% → 37%.

---

## Avancement par phase

| Phase | Mois | Statut | Livrable |
|---|---|---|---|
| Phase 1 — Revue de littérature | Mai | ✅ Terminé | Bibliographie annotée |
| Phase 2 — Conception du modèle | Juin | ✅ Terminé | `Phase2_ModelSpec_FR.pdf` |
| Phase 3 — Développement algorithme | Juillet | ✅ Terminé | `edi_coarsening.py` |
| Phase 4 — Implémentation & baselines | Juillet | ✅ Terminé | `edi_baselines.py`, `run_all_experiments.py` |
| Phase 5 — Évaluation | Juillet | ✅ Terminé | `experiments_results.json`, figures, tables |
| Phase 6 — Article de recherche | Juillet–Août | ⏳ En cours | Paper LLNCS (Overleaf) |

---

## Environnement technique

- **Python 3.11** — NumPy, Pandas, NetworkX, Matplotlib
- **Jupyter Notebook** — `edi_baselines.ipynb`
- **LaTeX / Overleaf** — format LLNCS Springer
- **Dataset** — MovieLens 100k (`data/ml-100k/`)

---

## Question de recherche
> Étant donné un graphe biparti pondéré G = (U ∪ I, E, w, s), comment construire un résumé G' via coarsening EDI-contraint qui réduit la taille tout en bornant la dégradation des métriques d'équité (ΔE), de diversité (ILD) et d'inclusion, comparé aux règles d'agrégation classiques (Average Score, Borda, Condorcet, Fair Re-rank) ?
