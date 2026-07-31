"""
Ajoute Weighted Borda aux résultats existants (experiments_results.json)
sans relancer les expériences coûteuses (coarsening + condorcet).
Lance aussi sur 1M si experiments_results_1m.json existe déjà.
"""
import json, os, sys
import numpy as np
import pandas as pd
from itertools import combinations as comb

BASE = os.path.dirname(__file__)


def load_100k():
    ratings = pd.read_csv(os.path.join(BASE, "data", "ml-100k", "u.data"),
                          sep="\t", header=None,
                          names=["user_id", "item_id", "rating", "timestamp"])
    users = pd.read_csv(os.path.join(BASE, "data", "ml-100k", "u.user"),
                        sep="|", header=None,
                        names=["user_id", "age", "gender", "occupation", "zip"])
    return ratings, users.set_index("user_id")["gender"].to_dict()


def load_1m():
    ratings = pd.read_csv(os.path.join(BASE, "data", "ml-1m", "ratings.dat"),
                          sep="::", header=None, engine="python",
                          names=["user_id", "item_id", "rating", "timestamp"])
    users = pd.read_csv(os.path.join(BASE, "data", "ml-1m", "users.dat"),
                        sep="::", header=None, engine="python",
                        names=["user_id", "gender", "age", "occupation", "zip"])
    return ratings, users.set_index("user_id")["gender"].to_dict()


def top_k_weighted_borda(ratings, gender_map, k):
    """score(i) = Σ_F borda(u,i)/n_F  +  Σ_M borda(u,i)/n_M"""
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = sum(1 for g in gender_map.values() if g == "M")
    r = ratings.copy()
    r["gender"] = r["user_id"].map(gender_map)

    def borda_user(group):
        ranked = group.sort_values("rating", ascending=False).reset_index(drop=True)
        ranked["borda"] = range(len(ranked), 0, -1)
        return ranked[["item_id", "borda", "gender"]]

    borda = r.groupby("user_id", group_keys=False).apply(borda_user)
    scores_F = borda[borda["gender"] == "F"].groupby("item_id")["borda"].sum() / n_F
    scores_M = borda[borda["gender"] == "M"].groupby("item_id")["borda"].sum() / n_M
    return list(scores_F.add(scores_M, fill_value=0).nlargest(k).index)


def edi_metrics(ratings, top_k, gender_map, theta):
    s = set(top_k); k = len(top_k)
    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]

    def util(gu):
        gr = ratings[ratings["user_id"].isin(gu) & ratings["item_id"].isin(s)]
        return gr["rating"].sum() / (len(gu) * k) if (gu and k) else 0.0

    dE = abs(util(f_u) - util(m_u))

    pivot = ratings[ratings["item_id"].isin(s)].pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0)
    vecs = {iid: pivot[iid].values if iid in pivot.columns
            else np.zeros(len(pivot)) for iid in top_k}
    tot, np_ = 0.0, 0
    for a, b in comb(top_k, 2):
        va, vb = vecs[a], vecs[b]
        n = np.linalg.norm(va) * np.linalg.norm(vb)
        tot += 1 - (np.dot(va, vb) / n if n > 0 else 0.0)
        np_ += 1
    ild = tot / np_ if np_ > 0 else 0.0

    def inc(gu):
        gr = ratings[ratings["user_id"].isin(gu) & ratings["item_id"].isin(s)
                     & (ratings["rating"] >= theta)]
        return (gr.groupby("user_id")["item_id"].nunique()
                .reindex(gu, fill_value=0).sum() / (len(gu) * k))

    return dict(dE=round(dE, 4), ILD=round(ild, 4),
                inc_F=round(inc(f_u), 4), inc_M=round(inc(m_u), 4))


def patch_json(json_path, ratings, gender_map):
    with open(json_path) as f:
        results = json.load(f)

    for key, exp in results["experiments"].items():
        if "Weighted Borda" in exp:
            print(f"  {key} — déjà présent, ignoré")
            continue
        k = exp["k"]; theta = exp["theta"]
        print(f"  k={k}, theta={theta}...", end="", flush=True)
        wb_items = top_k_weighted_borda(ratings, gender_map, k)
        exp["Weighted Borda"] = edi_metrics(ratings, wb_items, gender_map, theta)
        v = exp["Weighted Borda"]
        print(f"  dE={v['dE']}  ILD={v['ILD']}  inc_F={v['inc_F']}  inc_M={v['inc_M']}")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Sauvegarde : {json_path}\n")


# ── MovieLens 100k ─────────────────────────────────────────────────────────────
print("=== MovieLens 100k ===")
print("Chargement...")
ratings_100k, gmap_100k = load_100k()
patch_json(os.path.join(BASE, "experiments_results.json"), ratings_100k, gmap_100k)

# ── MovieLens 1M (si disponible) ───────────────────────────────────────────────
json_1m = os.path.join(BASE, "experiments_results_1m.json")
if os.path.exists(json_1m):
    print("=== MovieLens 1M ===")
    print("Chargement...")
    ratings_1m, gmap_1m = load_1m()
    patch_json(json_1m, ratings_1m, gmap_1m)
else:
    print("experiments_results_1m.json introuvable — ignoré (run_experiments_1m.py toujours en cours ?)")

print("Terminé.")
