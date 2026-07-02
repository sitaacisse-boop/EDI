# Projet de Recherche — EDI & Graph Summarization
**Auteure :** Adji Marieme Sita Cissé  
**Directeur :** Prof. Malek Mouhoub — Université de Regina  
**Période :** Mai – Octobre 2026 | Bourse DataIA Mobilité Internationale

---

## Sujet
Agrégation des préférences utilisateurs en garantissant l'Équité, la Diversité et l'Inclusion (EDI) par la Résumé de Graphe.

---

## Contenu du dossier

### Documents de référence
| Fichier | Description |
|---|---|
| `rechercheplan_pdf.pdf` | Plan de recherche complet (6 phases, Mai–Octobre 2026) |
| `Bibliographie_annotee_Phase1.pdf` | Bibliographie annotée — 13 références classées en 4 axes |
| `Logiciels Stage Recherche Cisse V3.pdf` | Liste des logiciels et outils utilisés |

### Spécification du modèle — Phase 2 (Juin 2026)
| Fichier | Description |
|---|---|
| `Phase2_ModelSpec_FR.pdf` | Spécification du modèle — **version française** (à envoyer à Prof. Mouhoub) |
| `Phase2_ModelSpec_EN.pdf` | Model specification — **English version** (for paper/conferences) |
| `Phase2_ModelSpec_FR.md` | Source Markdown — version française |
| `Phase2_ModelSpec.md` | Source Markdown — English version |

### Bibliographie Zotero (fichiers d'import)
| Fichier | Contenu |
|---|---|
| `AXE1-AXE2-Complement.ris` | Arrow 1951 + Awesome Graph Reduction + Coarse Measurements |
| `AXE3-EDI-Fairness.ris` | 4 références Axe 3 (fairness, diversité, inclusion) |
| `AXE4-SocialChoice.ris` | 2 références Axe 4 (SCRUF-D, SCRUF) |

---

## Environnement technique validé
- **Neo4j Desktop 2.1.4** — instance locale `EDI` (bolt://localhost:7687)
- **Python 3.11.9** + **NetworkX 3.6.1**
- **Zotero** (compte : adji-sita) — bibliothèque synchronisée, 13 références importées

---

## Avancement par phase

| Phase | Mois | Statut | Livrable |
|---|---|---|---|
| Phase 1 — Revue de littérature | Mai | ✅ En cours | Bibliographie annotée, setup technique |
| Phase 2 — Conception du modèle | Juin | ✅ En cours | `Phase2_ModelSpec_FR.pdf` |
| Phase 3 — Développement algorithme | Juillet | ⏳ À venir | Description algo + complexité |
| Phase 4 — Implémentation | Août | ⏳ À venir | Prototype Python + baselines |
| Phase 5 — Évaluation | Septembre | ⏳ À venir | Résultats expérimentaux |
| Phase 6 — Documentation | Octobre | ⏳ À venir | Article de recherche |

---

## Question de recherche
> Étant donné un graphe biparti pondéré G = (U ∪ I, E, w, s), comment construire un résumé G' (via détection de communautés et coarsening) qui réduit la taille tout en bornant la dégradation des métriques d'équité (ΔE), de diversité (ILD) et d'inclusion, comparé aux règles d'agrégation classiques ?
