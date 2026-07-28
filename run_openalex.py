"""
run_openalex.py — Dataset 4 : OpenAlex (recommandation d'auteurs académiques)

User   = venue (conférence / journal AI/ML/CS, 2018-2023)
Item   = auteur (chercheur)
Rating = min(nb_papers_auteur_dans_venue, 5)   [échelle 1-5]
Equité = genre des auteurs recommandés (item-side, comme RMP)
θ      = 2.0  (auteur "aimé" si ≥ 2 papers dans cette venue)
k      = [10]
"""

import sys, json, time, math, heapq
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np
from collections import defaultdict
import pyalex
from pyalex import Works
import gender_guesser.detector as gd

# Polite pool → 100 req/s au lieu de 10
pyalex.config.email = "fsamouyakar@ept.sn"

# ── Paramètres ───────────────────────────────────────────────────────────────
N_VENUES        = 100   # top venues par nb d'auteurs uniques
MIN_AUTHORS     = 20    # auteurs uniques minimum par venue
MIN_VENUES_AUTH = 2     # auteur doit publier dans ≥ 2 venues
THETA           = 1.0   # auteur "aimé" dès 1 paper dans cette venue
k_VALUES        = [10]
MAX_WORKS       = 60_000
SEED            = 42

CONCEPT_IDS = {
    "C119857082": "Machine Learning",
    "C154945302": "Artificial Intelligence",
    "C41008148":  "Computer Science",
}
YEAR_RANGE = (2018, 2023)

OUT_FILE   = "openalex_results.json"
CACHE_FILE = "openalex_raw_cache.json"   # évite de re-télécharger après rate-limit

# ── Gender inference ──────────────────────────────────────────────────────────
_det = gd.Detector()

def infer_gender(display_name: str) -> str:
    if not display_name:
        return "unknown"
    parts = str(display_name).strip().split()
    if not parts:
        return "unknown"
    first = parts[0].rstrip('.')
    if len(first) <= 2:          # initiale seule → inconnu
        return "unknown"
    first = first.capitalize()
    g = _det.get_gender(first)
    if g in ("male", "mostly_male"):
        return "M"
    if g in ("female", "mostly_female"):
        return "F"
    return "unknown"

# ── Collecte OpenAlex (avec cache + retry) ────────────────────────────────────
import os

def _fetch_page_with_retry(pager, max_retries=5):
    """Itérateur sur les pages avec backoff exponentiel sur 429."""
    for page in pager:
        yield page
        time.sleep(0.15)   # politesse de base entre chaque page

def collect_data():
    """
    Télécharge les papers via OpenAlex API et met en cache.
    Si CACHE_FILE existe, charge directement depuis le cache.
    """
    import importlib

    # ── Chargement depuis cache si disponible ──────────────────────────────────
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            cached = json.load(f)
        # Ignorer cache vide (run interrompu)
        if cached.get("raw"):
            print(f"Cache trouvé : {CACHE_FILE} — chargement sans API...")
            raw = defaultdict(lambda: defaultdict(int))
            for vid, auths in cached["raw"].items():
                for aname, cnt in auths.items():
                    raw[vid][aname] = cnt
            author_gender = cached["author_gender"]
            venue_names   = cached["venue_names"]
            n_F = sum(1 for g in author_gender.values() if g == "F")
            n_M = sum(1 for g in author_gender.values() if g == "M")
            print(f"  {sum(len(a) for a in raw.values())} (venue,auteur) paires chargées")
            print(f"  Auteurs avec genre : F={n_F}, M={n_M}")
            return raw, venue_names, author_gender
        else:
            print("Cache vide détecté — re-téléchargement...")

    # ── Téléchargement via API ─────────────────────────────────────────────────
    venue_names:  dict[str, str]           = {}
    raw:          dict[str, dict[str, int]]= defaultdict(lambda: defaultdict(int))
    author_gender:dict[str, str]           = {}
    seen_works = set()
    n_total = 0

    for cid, cname in CONCEPT_IDS.items():
        if n_total >= MAX_WORKS:
            break
        print(f"\nConcept : {cname}")

        pager = Works().filter(
            concepts={"id": cid},
            from_publication_date=f"{YEAR_RANGE[0]}-01-01",
            to_publication_date=f"{YEAR_RANGE[1]}-12-31",
            has_doi=True,
        ).select([
            "id", "authorships", "primary_location"
        ]).paginate(per_page=200, n_max=MAX_WORKS - n_total)

        n_concept = 0
        retries = 0
        page_iter = iter(pager)

        while True:
            try:
                page = next(page_iter)
                retries = 0
            except StopIteration:
                break
            except Exception as e:
                retries += 1
                if retries > 6:
                    print(f"  ⚠ Abandon après {retries} tentatives : {e}")
                    break
                wait = 2 ** retries
                print(f"  Rate-limit — attente {wait}s... (tentative {retries})")
                time.sleep(wait)
                continue

            for work in page:
                wid = work.get("id", "")
                if not wid or wid in seen_works:
                    continue
                seen_works.add(wid)

                loc   = work.get("primary_location") or {}
                src   = loc.get("source") or {}
                vid   = src.get("id", "")
                vname = src.get("display_name", "")
                if not vid or not vname:
                    continue

                auths = work.get("authorships") or []
                if not auths:
                    continue
                first_obj = next(
                    (a for a in auths if a.get("author_position") == "first"),
                    auths[0]
                )
                aobj  = first_obj.get("author") or {}
                aname = aobj.get("display_name", "")
                if not aname:
                    continue

                if aname not in author_gender:
                    author_gender[aname] = infer_gender(aname)
                if author_gender[aname] == "unknown":
                    continue

                venue_names[vid]  = vname
                raw[vid][aname]  += 1
                n_concept += 1
                n_total   += 1

                if n_total >= MAX_WORKS:
                    break

            if n_total >= MAX_WORKS:
                break
            time.sleep(0.15)

        print(f"  → {n_concept} papers | total {n_total}")

    print(f"\nTotal papers   : {n_total}")
    print(f"Venues uniques : {len(raw)}")
    print(f"Auteurs (genre connu) : {sum(1 for g in author_gender.values() if g != 'unknown')}")

    # ── Sauvegarde du cache (seulement si données non-vides) ──────────────────
    if raw:
        cache = {
            "raw":          {v: dict(auths) for v, auths in raw.items()},
            "venue_names":  venue_names,
            "author_gender":author_gender,
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        print(f"Cache sauvegardé : {CACHE_FILE}")
    else:
        print("⚠ Aucune donnée collectée — cache non écrasé.")

    return raw, venue_names, author_gender


# ── Construction de la matrice ────────────────────────────────────────────────
def build_matrix(raw, venue_names, author_gender):
    # Top N_VENUES par nb d'auteurs uniques, ≥ MIN_AUTHORS
    venue_uniq = {v: len(auths) for v, auths in raw.items()}
    top_venues = sorted(
        [(v, c) for v, c in venue_uniq.items() if c >= MIN_AUTHORS],
        key=lambda x: -x[1]
    )[:N_VENUES]
    sel_venues = {v for v, _ in top_venues}
    print(f"\nVenues sélectionnées : {len(sel_venues)} (≥{MIN_AUTHORS} auteurs uniques)")

    # Nb de venues par auteur (sur les venues sélectionnées)
    auth_v_cnt: dict[str, int] = defaultdict(int)
    for vid in sel_venues:
        for aname in raw[vid]:
            auth_v_cnt[aname] += 1

    valid_authors = {a for a, c in auth_v_cnt.items() if c >= MIN_VENUES_AUTH}
    print(f"Auteurs valides (≥{MIN_VENUES_AUTH} venues) : {len(valid_authors)}")

    # Matrice ratings : venue_idx → {author → rating}
    venue_list   = [v for v, _ in top_venues]
    venue_to_idx = {v: i for i, v in enumerate(venue_list)}

    ratings:    dict[int, dict[str, float]] = {}
    gender_map: dict[str, str]              = {}

    for vid in sel_venues:
        row: dict[str, float] = {}
        for aname, cnt in raw[vid].items():
            if aname not in valid_authors:
                continue
            gender_map[aname] = author_gender[aname]
            row[aname] = float(min(cnt, 5))
        if len(row) >= 2:
            ratings[venue_to_idx[vid]] = row

    # Stats genre
    g_counts = [gender_map[a] for a in gender_map]
    n_F = g_counts.count("F")
    n_M = g_counts.count("M")
    total = n_F + n_M
    alpha_F = n_F / total if total > 0 else 0.5
    alpha_M = n_M / total if total > 0 else 0.5
    n_ratings = sum(len(r) for r in ratings.values())

    print(f"Matrice finale : {len(ratings)} venues × {len(gender_map)} auteurs")
    print(f"Ratings        : {n_ratings}")
    print(f"Auteurs F={n_F} ({100*alpha_F:.1f}%), M={n_M} ({100*alpha_M:.1f}%)")

    return ratings, gender_map, alpha_F, alpha_M, venue_list, venue_names


# ── Métriques EDI (item-side) ─────────────────────────────────────────────────
METHODS_LIST = ["Average Score","Borda","Weighted Borda","Condorcet","Fair Re-rank","Ours"]

def _avg_scores(ratings: dict[int,dict[str,float]]) -> dict[str,float]:
    acc: dict[str,float] = defaultdict(float)
    cnt: dict[str,int]   = defaultdict(int)
    for row in ratings.values():
        for item, r in row.items():
            acc[item] += r
            cnt[item] += 1
    return {item: acc[item]/cnt[item] for item in acc}

def _borda(ratings: dict[int,dict[str,float]]) -> dict[str,float]:
    scores: dict[str,float] = defaultdict(float)
    for row in ratings.values():
        ranked = sorted(row.items(), key=lambda x: -x[1])
        n = len(ranked)
        for rank, (item, _) in enumerate(ranked):
            scores[item] += n - rank
    return dict(scores)

def _weighted_borda(ratings, alpha_F, alpha_M, gender_map):
    # même chose que borda ici (pas de group_weights sans user gender)
    return _borda(ratings)

def _condorcet(ratings: dict[int,dict[str,float]]) -> dict[str,float]:
    wins: dict[str,int] = defaultdict(int)
    items = list({i for row in ratings.values() for i in row})
    item_set = set(items)
    for row in ratings.values():
        row_items = list(row.keys())
        for i, a in enumerate(row_items):
            for b in row_items[i+1:]:
                if row[a] > row[b]:
                    wins[a] += 1
                elif row[b] > row[a]:
                    wins[b] += 1
    return dict(wins)

def topk(scores: dict[str,float], k: int) -> list[str]:
    if len(scores) <= k:
        return list(scores.keys())
    return [i for _, i in heapq.nlargest(k, ((v, i) for i, v in scores.items()))]

def fair_rerank(topk_list: list[str], k: int, gender_map: dict[str,str],
                alpha_F: float) -> list[str]:
    target_F = round(alpha_F * k)
    F_pool = [i for i in topk_list if gender_map.get(i) == "F"]
    M_pool = [i for i in topk_list if gender_map.get(i) == "M"]
    extra  = [i for i in topk_list if gender_map.get(i) not in ("F","M")]
    result = F_pool[:target_F] + M_pool[:k-target_F]
    while len(result) < k and extra:
        result.append(extra.pop(0))
    return result[:k]

def edi_metrics_item(all_topk: dict[int, list[str]], gender_map: dict[str,str],
                     alpha_F: float, alpha_M: float) -> dict:
    """Item-side EDI : ΔE sur le genre des items recommandés."""
    frac_F_list, frac_M_list = [], []
    ILD_list = []
    for uid, lst in all_topk.items():
        if not lst:
            continue
        genders = [gender_map.get(a, "?") for a in lst]
        k = len(lst)
        fF = genders.count("F") / k
        fM = genders.count("M") / k
        frac_F_list.append(fF)
        frac_M_list.append(fM)
        # ILD : proportion de paires de genres différents
        pairs = k * (k-1) / 2
        if pairs > 0:
            diff = sum(1 for i in range(k) for j in range(i+1,k)
                       if genders[i] != genders[j] and
                          genders[i] in ("F","M") and genders[j] in ("F","M"))
            ILD_list.append(diff / pairs)

    fF_mean = np.mean(frac_F_list) if frac_F_list else 0.0
    fM_mean = np.mean(frac_M_list) if frac_M_list else 0.0
    ILD_mean = np.mean(ILD_list) if ILD_list else 0.0

    # ΔE = |frac_F/alpha_F - frac_M/alpha_M|
    rF = fF_mean / alpha_F if alpha_F > 0 else 0
    rM = fM_mean / alpha_M if alpha_M > 0 else 0
    dE = abs(rF - rM)

    return {"dE": round(dE, 4), "ILD": round(ILD_mean, 4),
            "frac_F": round(fF_mean, 4)}


# ── Coarsening (item-side) ────────────────────────────────────────────────────
def cosine_sim(r1: dict, r2: dict) -> float:
    common = set(r1) & set(r2)
    if not common:
        return 0.0
    dot   = sum(r1[i]*r2[i] for i in common)
    norm1 = math.sqrt(sum(v*v for v in r1.values()))
    norm2 = math.sqrt(sum(v*v for v in r2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def run_coarsening_item(ratings_in, gender_map, alpha_F, alpha_M, k,
                        max_merges=400, eps_E=0.05):
    import time as _time
    t0 = _time.time()

    ratings = {u: dict(row) for u, row in ratings_in.items()}
    users   = list(ratings.keys())
    n       = len(users)
    parent  = {u: u for u in users}
    weights = {u: 1 for u in users}

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u

    def gtk(root):
        cb = {i: s for i, s in ratings[root].items() if s > 0}
        if not cb:
            return []
        return [i for _, i in heapq.nlargest(k, ((s, i) for i, s in cb.items()))]

    def global_dE(topk_map):
        return edi_metrics_item(topk_map, gender_map, alpha_F, alpha_M)["dE"]

    # Top-k initiaux et ΔE de référence
    cur_topk = {u: gtk(u) for u in users}
    dE0 = global_dE(cur_topk)

    # Toutes les paires (n=100 → 4 950 paires)
    print(f"    Paires similaires (n={n})...")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            sv = cosine_sim(ratings[users[i]], ratings[users[j]])
            if sv > 0.0:
                pairs.append((sv, users[i], users[j]))
    pairs.sort(key=lambda x: -x[0])
    print(f"    {len(pairs)} paires | dE0={dE0:.4f}")

    na = 0
    for sv, u0, v0 in pairs:
        if na >= max_merges:
            break
        ru, rv = find(u0), find(v0)
        if ru == rv:
            continue

        wu, wv = weights[ru], weights[rv]

        # Sauvegarde avant fusion (rollback propre)
        backup_ru = dict(ratings[ru])

        # Fusion pondérée
        merged: dict[str, float] = {}
        for item in set(ratings[ru]) | set(ratings[rv]):
            merged[item] = (wu * ratings[ru].get(item, 0) +
                            wv * ratings[rv].get(item, 0)) / (wu + wv)

        # Appliquer la fusion
        ratings[ru] = merged
        weights[ru] = wu + wv
        parent[rv]  = ru

        # Tester le ΔE résultant
        new_root_topk = gtk(ru)
        test_topk = dict(cur_topk)
        test_topk[ru] = new_root_topk
        for u in users:
            if u != ru and find(u) == ru:
                test_topk[u] = new_root_topk

        new_dE = global_dE(test_topk)

        if new_dE <= dE0 + eps_E:
            cur_topk = test_topk
            na += 1
        else:
            # Rollback complet
            ratings[ru] = backup_ru
            weights[ru] = wu
            parent[rv]  = rv

    n_sn    = len({find(u) for u in users})
    elapsed = _time.time() - t0
    return cur_topk, n_sn, elapsed


# ── Expériences ───────────────────────────────────────────────────────────────
def run_experiments(ratings, gender_map, alpha_F, alpha_M):
    import time as _t
    results = {}

    for k in k_VALUES:
        key = f"k{k}"
        results[key] = {}
        print(f"\n{'='*60}\nk = {k}")

        # 1. Average Score
        t0 = _t.time()
        avg = _avg_scores(ratings)
        topk_avg = {u: fair_rerank(topk(avg, k*5), k, gender_map, alpha_F)
                    for u in ratings}
        # Average Score : top-k brut (pas de fair rerank)
        topk_avg = {u: topk(avg, k) for u in ratings}
        mtr = edi_metrics_item(topk_avg, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(_t.time()-t0, 3)
        results[key]["Average Score"] = mtr
        print(f"  Average Score... {mtr['elapsed_s']}s | "
              f"ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} F={mtr['frac_F']:.2f}")

        # 2. Borda
        t0 = _t.time()
        bsc = _borda(ratings)
        topk_borda = {u: topk(bsc, k) for u in ratings}
        mtr = edi_metrics_item(topk_borda, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(_t.time()-t0, 3)
        results[key]["Borda"] = mtr
        print(f"  Borda... {mtr['elapsed_s']}s | "
              f"ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} F={mtr['frac_F']:.2f}")

        # 3. Weighted Borda (identique à Borda sans genre des venues)
        t0 = _t.time()
        wbsc = _weighted_borda(ratings, alpha_F, alpha_M, gender_map)
        topk_wb = {u: topk(wbsc, k) for u in ratings}
        mtr = edi_metrics_item(topk_wb, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(_t.time()-t0, 3)
        results[key]["Weighted Borda"] = mtr
        print(f"  Weighted Borda... {mtr['elapsed_s']}s | "
              f"ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} F={mtr['frac_F']:.2f}")

        # 4. Condorcet
        t0 = _t.time()
        csc = _condorcet(ratings)
        topk_cond = {u: topk(csc, k) for u in ratings}
        mtr = edi_metrics_item(topk_cond, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(_t.time()-t0, 3)
        results[key]["Condorcet"] = mtr
        print(f"  Condorcet... {mtr['elapsed_s']}s | "
              f"ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} F={mtr['frac_F']:.2f}")

        # 5. Fair Re-rank
        t0 = _t.time()
        topk_fair = {u: fair_rerank(topk(bsc, k*3), k, gender_map, alpha_F)
                     for u in ratings}
        mtr = edi_metrics_item(topk_fair, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(_t.time()-t0, 3)
        results[key]["Fair Re-rank"] = mtr
        print(f"  Fair Re-rank... {mtr['elapsed_s']}s | "
              f"ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} F={mtr['frac_F']:.2f}")

        # 6. Ours
        print(f"  Ours (coarsening, max_merges=400)...")
        topk_ours, n_sn, elapsed = run_coarsening_item(
            ratings, gender_map, alpha_F, alpha_M, k, max_merges=400
        )
        mtr = edi_metrics_item(topk_ours, gender_map, alpha_F, alpha_M)
        mtr["elapsed_s"] = round(elapsed, 3)
        mtr["n_supernodes"] = n_sn
        results[key]["Ours"] = mtr
        print(f"  → {elapsed:.3f}s | ΔE={mtr['dE']:.4f} ILD={mtr['ILD']:.4f} "
              f"F={mtr['frac_F']:.2f} ({n_sn} super-nœuds)")

    return results


# ── Point d'entrée ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Expériences EDI — OpenAlex Academic ===")
    print(f"N_VENUES={N_VENUES}, MIN_AUTHORS={MIN_AUTHORS}, "
          f"θ={THETA}, k={k_VALUES}")

    raw, venue_names, author_gender = collect_data()
    ratings, gender_map, alpha_F, alpha_M, venue_list, venue_names = \
        build_matrix(raw, venue_names, author_gender)

    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = sum(1 for g in gender_map.values() if g == "M")
    n_venues  = len(ratings)
    n_authors = len(gender_map)
    n_ratings = sum(len(r) for r in ratings.values())

    results = run_experiments(ratings, gender_map, alpha_F, alpha_M)

    out = {
        "meta": {
            "dataset":    "OpenAlex",
            "domain":     "AI/ML/CS (2018-2023)",
            "user_type":  "venue (conference/journal)",
            "item_type":  "author (researcher)",
            "n_venues":   n_venues,
            "n_authors":  n_authors,
            "n_F":        n_F,
            "n_M":        n_M,
            "alpha_F":    round(alpha_F, 4),
            "alpha_M":    round(alpha_M, 4),
            "n_ratings":  n_ratings,
            "theta":      THETA,
            "k_values":   k_VALUES,
            "note":       "item-side equity : genre des auteurs recommandés"
        },
        "experiments": results
    }

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSauvegarde : {OUT_FILE}")
