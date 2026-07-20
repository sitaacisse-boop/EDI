"""
EDI Baselines — Étape 1 & 2
Chargement de MovieLens 100k, construction du graphe biparti,
calcul des baselines (Average Score, Borda, Condorcet) et métriques EDI.
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
    # Ratings : user_id | item_id | rating | timestamp
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"),
        sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    # Utilisateurs : user_id | age | gender | occupation | zip
    users = pd.read_csv(
        os.path.join(DATA_DIR, "u.user"),
        sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    return ratings, users


# ── 2. Construction du graphe biparti G = (U ∪ I, E, w, s) ───────────────────
def build_graph(ratings, users):
    G = nx.Graph()

    # Attribut sensible s : U → {F, M}
    gender_map = users.set_index("user_id")["gender"].to_dict()

    # Nœuds utilisateurs
    for uid, gender in gender_map.items():
        G.add_node(f"u{uid}", bipartite=0, gender=gender)

    # Nœuds items + arêtes pondérées
    for _, row in ratings.iterrows():
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        w   = float(row["rating"])
        G.add_node(f"i{iid}", bipartite=1)
        G.add_edge(f"u{uid}", f"i{iid}", weight=w)

    return G, gender_map


# ── 3. Baselines : top-k collectif ───────────────────────────────────────────
def top_k_average_score(ratings, k=10):
    scores = ratings.groupby("item_id")["rating"].mean()
    return list(scores.nlargest(k).index)


def top_k_borda(ratings, k=10):
    """Borda : chaque utilisateur classe ses films, on somme les rangs."""
    def borda_scores(group):
        ranked = group.sort_values("rating", ascending=False).reset_index(drop=True)
        ranked["borda"] = range(len(ranked), 0, -1)
        return ranked[["item_id", "borda"]]

    borda = ratings.groupby("user_id", group_keys=False).apply(borda_scores)
    scores = borda.groupby("item_id")["borda"].sum()
    return list(scores.nlargest(k).index)


def top_k_condorcet(ratings, k=10):
    """Condorcet rapide : pour chaque utilisateur, comparaison vectorisee
    par paires d'items notes. Complexite O(users x avg_items^2) avec numpy."""
    total_wins = {}
    for _, group in ratings.groupby("user_id"):
        items  = group["item_id"].values
        scores = group["rating"].values
        n = len(items)
        if n < 2:
            continue
        # wins_per_item[i] = nombre d'items que l'item i bat pour cet utilisateur
        wins_per_item = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, item_id in enumerate(items):
            total_wins[item_id] = total_wins.get(item_id, 0) + int(wins_per_item[i])

    ranked = sorted(total_wins.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:k]]


# ── 4. Métriques EDI ─────────────────────────────────────────────────────────
def group_utility(ratings, top_k_items, gender_map):
    """utility_g(R) = moyenne sur les membres du groupe de leur satisfaction sur R."""
    results = {}
    top_k_set = set(top_k_items)
    for gender in ["F", "M"]:
        group_users = [uid for uid, g in gender_map.items() if g == gender]
        group_ratings = ratings[
            ratings["user_id"].isin(group_users) &
            ratings["item_id"].isin(top_k_set)
        ]
        k = len(top_k_items)
        if not group_users or k == 0:
            results[gender] = 0.0
            continue
        total = group_ratings["rating"].sum()
        results[gender] = total / (len(group_users) * k)
    return results


def delta_E(ratings, top_k_items, gender_map):
    """ΔE = |utility_F(R) − utility_M(R)|"""
    util = group_utility(ratings, top_k_items, gender_map)
    return abs(util["F"] - util["M"])


def ILD(ratings, top_k_items):
    """Intra-List Diversity : dissimilarité cosinus pairwise moyenne dans le top-k."""
    k = len(top_k_items)
    if k < 2:
        return 0.0

    # Vecteurs de notes par item
    pivot = ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
    vectors = {}
    for iid in top_k_items:
        if iid in pivot.columns:
            vectors[iid] = pivot[iid].values
        else:
            vectors[iid] = np.zeros(len(pivot))

    total_dist = 0.0
    n_pairs = 0
    for i, j in combinations(top_k_items, 2):
        vi, vj = vectors[i], vectors[j]
        norm = np.linalg.norm(vi) * np.linalg.norm(vj)
        cos_sim = np.dot(vi, vj) / norm if norm > 0 else 0.0
        total_dist += 1 - cos_sim
        n_pairs += 1

    return total_dist / n_pairs if n_pairs > 0 else 0.0


def inclusion(ratings, top_k_items, gender_map, theta=4.0):
    """inclusion_g(R) = fraction d'items du top-k pertinents pour le groupe g."""
    top_k_set = set(top_k_items)
    k = len(top_k_items)
    results = {}
    for gender in ["F", "M"]:
        group_users = [uid for uid, g in gender_map.items() if g == gender]
        if not group_users or k == 0:
            results[gender] = 0.0
            continue
        group_ratings = ratings[
            ratings["user_id"].isin(group_users) &
            ratings["item_id"].isin(top_k_set) &
            (ratings["rating"] >= theta)
        ]
        relevant_per_user = group_ratings.groupby("user_id")["item_id"].nunique()
        total = relevant_per_user.reindex(group_users, fill_value=0).sum()
        results[gender] = total / (len(group_users) * k)
    return results


# ── 5. Calcul complet ─────────────────────────────────────────────────────────
def evaluate(ratings, gender_map, method_name, top_k_items, theta=4.0):
    dE   = delta_E(ratings, top_k_items, gender_map)
    ild  = ILD(ratings, top_k_items)
    inc  = inclusion(ratings, top_k_items, gender_map, theta)
    alpha_F = sum(1 for g in gender_map.values() if g == "F") / len(gender_map)
    alpha_M = 1 - alpha_F

    sep = "-" * 50
    print(f"\n{sep}")
    print(f"  Methode : {method_name}  (top-{len(top_k_items)}, theta={theta})")
    print(sep)
    print(f"  delta_E (equite)  = {dE:.4f}   [cible ~ 0]")
    print(f"  ILD (diversite)   = {ild:.4f}   [cible : haut]")
    print(f"  inclusion_F       = {inc['F']:.4f}   [cible >= alpha_F = {alpha_F:.2f}]")
    print(f"  inclusion_M       = {inc['M']:.4f}   [cible >= alpha_M = {alpha_M:.2f}]")
    return {"method": method_name, "dE": dE, "ILD": ild,
            "inc_F": inc["F"], "inc_M": inc["M"]}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement de MovieLens 100k...")
    ratings, users = load_movielens()
    print(f"  {len(ratings)} ratings | {ratings['user_id'].nunique()} users "
          f"| {ratings['item_id'].nunique()} items")

    G, gender_map = build_graph(ratings, users)
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = sum(1 for g in gender_map.values() if g == "M")
    print(f"  Groupes : {n_F} femmes ({n_F/len(gender_map):.0%}) "
          f"| {n_M} hommes ({n_M/len(gender_map):.0%})")

    K = 10
    THETA = 4.0
    results = []

    print(f"\nCalcul des baselines (k={K}, theta={THETA})...")

    top_avg   = top_k_average_score(ratings, K)
    results.append(evaluate(ratings, gender_map, "Average Score", top_avg, THETA))

    top_borda = top_k_borda(ratings, K)
    results.append(evaluate(ratings, gender_map, "Borda", top_borda, THETA))

    print("\n  (Condorcet peut prendre quelques minutes sur 100k...)")
    top_cond  = top_k_condorcet(ratings, K)
    results.append(evaluate(ratings, gender_map, "Condorcet", top_cond, THETA))

    # Recapitulatif
    sep2 = "=" * 50
    sep3 = "-" * 46
    print(f"\n{sep2}")
    print("  RECAPITULATIF")
    print(sep2)
    print(f"  {'Methode':<16} {'dE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7}")
    print(f"  {sep3}")
    alpha_F = n_F / len(gender_map)
    for r in results:
        print(f"  {r['method']:<16} {r['dE']:>7.4f} {r['ILD']:>7.4f} "
              f"{r['inc_F']:>7.4f} {r['inc_M']:>7.4f}")
    print(f"  {sep3}")
    print(f"  alpha_F (cible inclusion_F) = {alpha_F:.2f}")
