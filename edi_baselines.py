"""
EDI Baselines — Étapes 1 & 2
Chargement de MovieLens 100k, construction du graphe biparti,
calcul des quatre baselines (Average Score, Borda, Condorcet, Fair Re-rank)
et métriques EDI (ΔE, ILD, inclusion).

Ce module est importé par run_all_experiments.py et utilisé directement
depuis le notebook edi_baselines.ipynb.
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

# ── Chemins ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")


# ── 1. Chargement des données ─────────────────────────────────────────────────
def load_movielens():
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"),
        sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        os.path.join(DATA_DIR, "u.user"),
        sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    return ratings, users


# ── 2. Construction du graphe biparti G = (U ∪ I, E, w, s) ───────────────────
def build_graph(ratings, users):
    G = nx.Graph()
    gender_map = users.set_index("user_id")["gender"].to_dict()
    for uid, gender in gender_map.items():
        G.add_node(f"u{uid}", bipartite=0, gender=gender)
    for _, row in ratings.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        w   = float(row["rating"])
        G.add_node(f"i{iid}", bipartite=1)
        G.add_edge(f"u{uid}", f"i{iid}", weight=w)
    return G, gender_map


# ── 3. Baselines ──────────────────────────────────────────────────────────────
def top_k_average_score(ratings, k=10):
    scores = ratings.groupby("item_id")["rating"].mean()
    return list(scores.nlargest(k).index)


def top_k_borda(ratings, k=10):
    """Borda : chaque utilisateur classe ses films, on somme les rangs."""
    def borda_scores(group):
        ranked = group.sort_values("rating", ascending=False).reset_index(drop=True)
        ranked["borda"] = range(len(ranked), 0, -1)
        return ranked[["item_id", "borda"]]
    borda  = ratings.groupby("user_id", group_keys=False).apply(borda_scores)
    scores = borda.groupby("item_id")["borda"].sum()
    return list(scores.nlargest(k).index)


def top_k_condorcet(ratings, k=10):
    """Condorcet : item i bat j si la majorité des utilisateurs préfèrent i."""
    total_wins = {}
    for _, group in ratings.groupby("user_id"):
        items  = group["item_id"].values
        scores = group["rating"].values
        if len(items) < 2:
            continue
        wins_per_item = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, item_id in enumerate(items):
            total_wins[item_id] = total_wins.get(item_id, 0) + int(wins_per_item[i])
    ranked = sorted(total_wins.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:k]]


def top_k_fair_rerank(ratings, gender_map, k=10, theta=4.0, pool=40):
    """
    Fair Re-rank : sélection greedy sur un pool de candidats Borda.
    À chaque étape, choisit l'item qui maximise inc_F + inc_M - ΔE.
    """
    def bs(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["b"] = range(len(r), 0, -1)
        return r[["item_id", "b"]]

    b = ratings.groupby("user_id", group_keys=False).apply(bs)
    candidates = list(b.groupby("item_id")["b"].sum().nlargest(pool).index)

    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]

    def item_score(iid, group_users):
        sub = ratings[(ratings["item_id"] == iid) & (ratings["user_id"].isin(group_users))]
        return float(sub["rating"].sum()) / (len(group_users) + 1e-9)

    def item_inc(iid, group_users):
        sub = ratings[
            (ratings["item_id"] == iid) &
            (ratings["user_id"].isin(group_users)) &
            (ratings["rating"] >= theta)
        ]
        return len(sub["user_id"].unique()) / (len(group_users) + 1e-9)

    item_sf = {iid: item_score(iid, f_u) for iid in candidates}
    item_sm = {iid: item_score(iid, m_u) for iid in candidates}
    item_if = {iid: item_inc(iid,  f_u) for iid in candidates}
    item_im = {iid: item_inc(iid,  m_u) for iid in candidates}

    selected  = []
    remaining = list(candidates)
    for _ in range(k):
        if not remaining:
            break
        def gain(iid):
            sel = selected + [iid]
            sf  = sum(item_sf[i] for i in sel) / len(sel)
            sm  = sum(item_sm[i] for i in sel) / len(sel)
            return (sum(item_if[i] for i in sel) / len(sel) +
                    sum(item_im[i] for i in sel) / len(sel) -
                    abs(sf - sm))
        best = max(remaining, key=gain)
        selected.append(best)
        remaining.remove(best)
    return selected


# ── 4. Métriques EDI ─────────────────────────────────────────────────────────
def group_utility(ratings, top_k_items, gender_map):
    """utility_g(R) = moyenne sur le groupe g de leur satisfaction sur R."""
    results  = {}
    top_k_set = set(top_k_items)
    k = len(top_k_items)
    for gender in ["F", "M"]:
        group_users = [uid for uid, g in gender_map.items() if g == gender]
        gr = ratings[
            ratings["user_id"].isin(group_users) &
            ratings["item_id"].isin(top_k_set)
        ]
        results[gender] = gr["rating"].sum() / (len(group_users) * k) if (group_users and k) else 0.0
    return results


def delta_E(ratings, top_k_items, gender_map):
    """ΔE = |utility_F(R) − utility_M(R)|"""
    util = group_utility(ratings, top_k_items, gender_map)
    return abs(util["F"] - util["M"])


def ILD(ratings, top_k_items):
    """Intra-List Diversity : dissimilarité cosinus pairwise moyenne."""
    k = len(top_k_items)
    if k < 2:
        return 0.0
    pivot = ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
    vectors = {iid: pivot[iid].values if iid in pivot.columns else np.zeros(len(pivot))
               for iid in top_k_items}
    total_dist, n_pairs = 0.0, 0
    for i, j in combinations(top_k_items, 2):
        vi, vj = vectors[i], vectors[j]
        norm   = np.linalg.norm(vi) * np.linalg.norm(vj)
        total_dist += 1 - (np.dot(vi, vj) / norm if norm > 0 else 0.0)
        n_pairs += 1
    return total_dist / n_pairs if n_pairs > 0 else 0.0


def inclusion(ratings, top_k_items, gender_map, theta=4.0):
    """inclusion_g(R) = fraction d'items du top-k pertinents (≥ θ) pour le groupe g."""
    top_k_set = set(top_k_items)
    k = len(top_k_items)
    results = {}
    for gender in ["F", "M"]:
        group_users = [uid for uid, g in gender_map.items() if g == gender]
        if not group_users or k == 0:
            results[gender] = 0.0
            continue
        gr = ratings[
            ratings["user_id"].isin(group_users) &
            ratings["item_id"].isin(top_k_set) &
            (ratings["rating"] >= theta)
        ]
        relevant_per_user = gr.groupby("user_id")["item_id"].nunique()
        total = relevant_per_user.reindex(group_users, fill_value=0).sum()
        results[gender] = total / (len(group_users) * k)
    return results


# ── 5. Évaluation complète d'une méthode ─────────────────────────────────────
def evaluate(ratings, gender_map, method_name, top_k_items, theta=4.0):
    dE_val  = delta_E(ratings, top_k_items, gender_map)
    ild_val = ILD(ratings, top_k_items)
    inc     = inclusion(ratings, top_k_items, gender_map, theta)
    alpha_F = sum(1 for g in gender_map.values() if g == "F") / len(gender_map)
    alpha_M = 1 - alpha_F
    sep = "-" * 52
    print(f"\n{sep}")
    print(f"  {method_name}  (k={len(top_k_items)}, θ={theta})")
    print(sep)
    print(f"  ΔE        = {dE_val:.4f}   [cible ≈ 0]")
    print(f"  ILD       = {ild_val:.4f}   [cible : élevé]")
    print(f"  inc_F     = {inc['F']:.4f}   [cible ≥ α_F = {alpha_F:.2f}]")
    print(f"  inc_M     = {inc['M']:.4f}   [cible ≥ α_M = {alpha_M:.2f}]")
    return {"method": method_name, "dE": dE_val, "ILD": ild_val,
            "inc_F": inc["F"], "inc_M": inc["M"]}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement de MovieLens 100k...")
    ratings, users = load_movielens()
    gender_map = users.set_index("user_id")["gender"].to_dict()
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = len(gender_map) - n_F
    alpha_F = n_F / len(gender_map)
    print(f"  {len(ratings):,} ratings | {len(gender_map)} users "
          f"({n_F} F / {n_M} M) | {ratings['item_id'].nunique()} items")

    K_VALUES = [5, 10, 20]
    THETA    = 4.0

    # Condorcet calculé une seule fois (indépendant de k, on tronque ensuite)
    print("\nCondorcet (calcul unique, peut prendre ~1 min)...")
    total_wins = {}
    for _, group in ratings.groupby("user_id"):
        items  = group["item_id"].values
        sc     = group["rating"].values
        if len(items) < 2:
            continue
        wins_per_item = np.sum(sc[:, None] > sc[None, :], axis=1)
        for i, item_id in enumerate(items):
            total_wins[item_id] = total_wins.get(item_id, 0) + int(wins_per_item[i])
    cond_full = [item for item, _ in sorted(total_wins.items(), key=lambda x: x[1], reverse=True)]
    print("  Condorcet : OK")

    sep2 = "=" * 58
    print(f"\n{sep2}")
    print(f"  {'Methode':<16} {'k':>4} {'dE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7}")
    print(f"  {'-'*54}")
    for k in K_VALUES:
        print(f"\n  --- k={k}, θ={THETA} ---")
        for method, top_k in [
            ("Average Score", top_k_average_score(ratings, k)),
            ("Borda",         top_k_borda(ratings, k)),
            ("Condorcet",     cond_full[:k]),
            ("Fair Re-rank",  top_k_fair_rerank(ratings, gender_map, k, THETA)),
        ]:
            r = evaluate(ratings, gender_map, method, top_k, THETA)
    print(f"\n  α_F (cible inc_F) = {alpha_F:.2f}")
    print(f"  α_M (cible inc_M) = {1-alpha_F:.2f}")
    print(sep2)
