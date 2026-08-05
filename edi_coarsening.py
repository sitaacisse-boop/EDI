"""
EDI-Constrained Graph Coarsening — contribution principale
Adji Marieme Sita Cissé — Stage M2 DataScale / University of Regina

Idee cle :
  En fusionnant preferentiellement les utilisateurs du meme groupe de genre
  (intra-groupe), on reduit le desequilibre numerique F/M dans le graphe coarsene.
  Sur MovieLens 100k : 273 F vs 670 M => apres coarsenisation => ~273 F et ~273 M supernodes.
  Le vote Borda sur le graphe coarsene devient plus equilibre => equite amelioree.

Algorithme :
  1. Chaque utilisateur = son propre super-noeud.
  2. Paires triees par : (meme genre -> priorite, similarite cosinus).
  3. Pour chaque fusion candidate (greedy) :
       - mise a jour incrementale des scores Borda collectifs
       - si le top-k change : verification des contraintes EDI
       - si ok -> fusion validee, sinon -> rollback
  4. Retourne : top-k, metriques EDI, taux de compression.
"""

import os
import time
import numpy as np
import pandas as pd
from itertools import combinations

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")


# ── Chargement ────────────────────────────────────────────────────────────────
def load_movielens():
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "u.data"), sep="\t", header=None,
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        os.path.join(DATA_DIR, "u.user"), sep="|", header=None,
        names=["user_id", "age", "gender", "occupation", "zip"]
    )
    return ratings, users


# ── Pre-calcul EDI (une seule fois) ──────────────────────────────────────────
def precompute_edi(ratings, gender_map, theta=4.0):
    f_users = {u for u, g in gender_map.items() if g == "F"}
    n_F = len(f_users)
    n_M = len(gender_map) - n_F
    f_sum, m_sum, f_rel, m_rel = {}, {}, {}, {}

    for _, row in ratings.iterrows():
        uid, iid, r = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
        if uid in f_users:
            f_sum[iid] = f_sum.get(iid, 0.0) + r
            if r >= theta:
                f_rel[iid] = f_rel.get(iid, 0) + 1
        else:
            m_sum[iid] = m_sum.get(iid, 0.0) + r
            if r >= theta:
                m_rel[iid] = m_rel.get(iid, 0) + 1

    pivot = ratings.pivot(index="user_id", columns="item_id", values="rating").fillna(0)
    item_ids   = list(pivot.columns)
    item_vecs  = pivot.values.T.astype(np.float32)
    item_norms = np.linalg.norm(item_vecs, axis=1)
    item_idx   = {iid: i for i, iid in enumerate(item_ids)}

    return dict(n_F=n_F, n_M=n_M,
                f_sum=f_sum, m_sum=m_sum, f_rel=f_rel, m_rel=m_rel,
                item_vecs=item_vecs, item_norms=item_norms,
                item_idx=item_idx, pivot=pivot)


def fast_delta_E(top_k, c):
    k = len(top_k)
    fT = sum(c["f_sum"].get(i, 0.0) for i in top_k)
    mT = sum(c["m_sum"].get(i, 0.0) for i in top_k)
    return abs(fT / (c["n_F"] * k) - mT / (c["n_M"] * k))


def fast_inc_F(top_k, c):
    return sum(c["f_rel"].get(i, 0) for i in top_k) / (c["n_F"] * len(top_k))


def fast_inc_M(top_k, c):
    return sum(c["m_rel"].get(i, 0) for i in top_k) / (c["n_M"] * len(top_k))


def fast_ILD(top_k, c):
    k = len(top_k)
    if k < 2:
        return 0.0
    idxs  = [c["item_idx"][i] for i in top_k if i in c["item_idx"]]
    vecs  = c["item_vecs"][idxs]
    norms = c["item_norms"][idxs]
    total, n = 0.0, 0
    for i in range(len(vecs) - 1):
        for j in range(i + 1, len(vecs)):
            nn = norms[i] * norms[j]
            cos = float(np.dot(vecs[i], vecs[j])) / nn if nn > 0 else 0.0
            total += 1 - cos
            n += 1
    return total / n if n > 0 else 0.0


# ── Scores Borda d'un super-noeud ────────────────────────────────────────────
def sn_borda_scores(sn_sums, sn_cnts, sn_id):
    """Scores Borda du super-noeud sn_id bases sur ses ratings moyens."""
    avgs = {iid: sn_sums[sn_id][iid] / sn_cnts[sn_id][iid]
            for iid in sn_sums[sn_id]}
    ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)
    return {iid: n - rank for rank, (iid, _) in enumerate(ranked)}


# ── Algorithme principal ──────────────────────────────────────────────────────
def edi_coarsening(ratings, gender_map, k=10, theta=4.0,
                   eps_E=0.10, eps_D=0.05, eps_I=0.05,
                   max_merges=472, intragroup_first=True, verbose=True):
    """
    Coarsenisation EDI-contrainte du graphe biparti de preferences.

    Parametres
    ----------
    eps_E            : tolerance ΔE   (on accepte ΔE' <= ref_dE + eps_E)
    eps_D            : tolerance ILD  (on accepte ILD' >= ref_ILD - eps_D)
    eps_I            : tolerance inc_F (on accepte inc_F' >= ref_incF - eps_I)
    max_merges       : nombre max de fusions
    intragroup_first : si True, privilegier les fusions au sein du meme genre
    """
    t0 = time.time()
    users_list = list(gender_map.keys())
    n_users    = len(users_list)
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = n_users - n_F

    if verbose:
        print(f"Pre-calcul EDI...")
    cache = precompute_edi(ratings, gender_map, theta)

    # ── Ratings par utilisateur ───────────────────────────────────────────────
    user_ratings = {uid: dict(zip(grp["item_id"], grp["rating"]))
                    for uid, grp in ratings.groupby("user_id")}

    # ── Initialisation des super-noeuds ──────────────────────────────────────
    # sn_sums[sn_id][iid] = somme des notes  ; sn_cnts[sn_id][iid] = nombre de noteurs
    sn_sums    = {uid: dict(user_ratings.get(uid, {}))      for uid in users_list}
    sn_cnts    = {uid: {iid: 1 for iid in user_ratings.get(uid, {})} for uid in users_list}
    sn_members = {uid: {uid}                                 for uid in users_list}
    user_to_sn = {uid: uid                                   for uid in users_list}
    active_sns = set(users_list)

    # ── Scores Borda collectifs ───────────────────────────────────────────────
    # collective_borda[iid] = somme des scores Borda de tous les SNs pour cet item
    # Initialisation = Borda standard sur le graphe complet
    sn_borda_cache = {}
    collective_borda = {}
    for uid in users_list:
        b = sn_borda_scores(sn_sums, sn_cnts, uid)
        sn_borda_cache[uid] = b
        for iid, score in b.items():
            collective_borda[iid] = collective_borda.get(iid, 0) + score

    def get_top_k_borda():
        ranked = sorted(collective_borda.items(), key=lambda x: x[1], reverse=True)
        return [iid for iid, _ in ranked[:k]]

    # ── Metriques de reference (Borda sur graphe complet) ────────────────────
    ref_top_k = get_top_k_borda()
    ref_dE    = fast_delta_E(ref_top_k, cache)
    ref_ILD   = fast_ILD(ref_top_k, cache)
    ref_incF  = fast_inc_F(ref_top_k, cache)

    max_dE   = ref_dE  + eps_E
    min_ILD  = ref_ILD - eps_D
    min_incF = ref_incF - eps_I
    cur_top_k = list(ref_top_k)

    if verbose:
        print(f"  Borda sur G (reference) :")
        print(f"    dE={ref_dE:.4f}  ILD={ref_ILD:.4f}  inc_F={ref_incF:.4f}")
        print(f"  Tolerances : eps_E={eps_E}  eps_D={eps_D}  eps_I={eps_I}")

    # ── Similarite cosinus + tri des paires ──────────────────────────────────
    if verbose:
        print(f"\nCalcul des similarites ({n_users} utilisateurs)...")
    M = cache["pivot"].values.astype(np.float32)
    nu = np.linalg.norm(M, axis=1, keepdims=True)
    nu[nu == 0] = 1
    sim_mat  = (M / nu) @ (M / nu).T
    uid_list = list(cache["pivot"].index)

    pairs = []
    for i in range(n_users):
        for j in range(i + 1, n_users):
            s = float(sim_mat[i, j])
            if s > 0:
                same = 1 if gender_map[uid_list[i]] == gender_map[uid_list[j]] else 0
                pairs.append((same if intragroup_first else 0, s, uid_list[i], uid_list[j]))
    pairs.sort(reverse=True)

    if verbose:
        print(f"  {len(pairs):,} paires  "
              f"({'intra-groupe en priorite' if intragroup_first else 'par similarite seule'})")

    # ── Fusions iteratives ────────────────────────────────────────────────────
    n_accepted = n_rejected = 0
    n_initial  = len(active_sns)

    if verbose:
        print(f"\nFusions (max={max_merges})...")

    for priority, sim, uid_a, uid_b in pairs:
        if n_accepted >= max_merges:
            break
        sn_a = user_to_sn.get(uid_a)
        sn_b = user_to_sn.get(uid_b)
        if sn_a is None or sn_b is None or sn_a == sn_b:
            continue

        items_a   = set(sn_sums[sn_a].keys())
        items_b   = set(sn_sums[sn_b].keys())
        items_both = items_a & items_b

        # Fusion tentative des ratings
        merged_sums = dict(sn_sums[sn_a])
        merged_cnts = dict(sn_cnts[sn_a])
        for iid in items_b:
            if iid in merged_sums:
                merged_sums[iid] += sn_sums[sn_b][iid]
                merged_cnts[iid] += sn_cnts[sn_b][iid]
            else:
                merged_sums[iid] = sn_sums[sn_b][iid]
                merged_cnts[iid] = sn_cnts[sn_b][iid]

        # Borda du super-noeud fusionne
        merged_avgs = {iid: merged_sums[iid] / merged_cnts[iid] for iid in merged_sums}
        ranked_merged = sorted(merged_avgs.items(), key=lambda x: x[1], reverse=True)
        n_merged = len(ranked_merged)
        borda_merged = {iid: n_merged - rank for rank, (iid, _) in enumerate(ranked_merged)}

        # Mise a jour tentative du Borda collectif
        # Retirer sn_a et sn_b, ajouter le fusionne
        undo_data = {}
        for iid, score in sn_borda_cache[sn_a].items():
            prev = collective_borda.get(iid, 0)
            collective_borda[iid] = prev - score
            undo_data[iid] = undo_data.get(iid, prev)
        for iid, score in sn_borda_cache[sn_b].items():
            prev = collective_borda.get(iid, 0)
            collective_borda[iid] = prev - score
            if iid not in undo_data:
                undo_data[iid] = prev + sn_borda_cache[sn_a].get(iid, 0)
        for iid, score in borda_merged.items():
            collective_borda[iid] = collective_borda.get(iid, 0) + score

        new_top_k = get_top_k_borda()

        if new_top_k == cur_top_k:
            ok = True
        else:
            new_dE   = fast_delta_E(new_top_k, cache)
            new_incF = fast_inc_F(new_top_k, cache)
            new_ILD  = fast_ILD(new_top_k, cache)
            ok = (new_dE <= max_dE) and (new_ILD >= min_ILD) and (new_incF >= min_incF)

        if ok:
            # Valider la fusion
            sn_sums[sn_a]    = merged_sums
            sn_cnts[sn_a]    = merged_cnts
            sn_borda_cache[sn_a] = borda_merged
            for uid in sn_members[sn_b]:
                user_to_sn[uid] = sn_a
            sn_members[sn_a] |= sn_members[sn_b]
            del sn_sums[sn_b]; del sn_cnts[sn_b]
            del sn_borda_cache[sn_b]; del sn_members[sn_b]
            active_sns.discard(sn_b)
            if new_top_k != cur_top_k:
                cur_top_k = new_top_k
            n_accepted += 1
            if verbose and n_accepted % 100 == 0:
                dE_now = fast_delta_E(cur_top_k, cache)
                iF_now = fast_inc_F(cur_top_k, cache)
                n_F_sn = sum(1 for sn in active_sns if gender_map[min(sn_members[sn])] == "F")
                print(f"  {n_accepted} fusions | {len(active_sns)} supernodes "
                      f"(F:{n_F_sn} M:{len(active_sns)-n_F_sn}) | "
                      f"dE={dE_now:.3f} inc_F={iF_now:.3f}")
        else:
            # Annuler
            for iid, score in borda_merged.items():
                collective_borda[iid] = collective_borda.get(iid, 0) - score
            for iid, score in sn_borda_cache[sn_b].items():
                collective_borda[iid] = collective_borda.get(iid, 0) + score
            for iid, score in sn_borda_cache[sn_a].items():
                collective_borda[iid] = collective_borda.get(iid, 0) + score
            n_rejected += 1

    # ── Resultats finaux ──────────────────────────────────────────────────────
    final_top_k = get_top_k_borda()
    final_dE    = fast_delta_E(final_top_k, cache)
    final_ILD   = fast_ILD(final_top_k, cache)
    final_incF  = fast_inc_F(final_top_k, cache)
    final_incM  = fast_inc_M(final_top_k, cache)
    ratio       = len(active_sns) / n_initial
    elapsed     = time.time() - t0

    # Repartition finale par genre
    n_F_sn_final = sum(1 for sn in active_sns if gender_map[min(sn_members[sn])] == "F")
    n_M_sn_final = len(active_sns) - n_F_sn_final

    if verbose:
        sep = "=" * 58
        print(f"\n{sep}")
        print("  RESULTAT FINAL")
        print(sep)
        print(f"  Super-noeuds     : {len(active_sns)} / {n_initial}  "
              f"(F:{n_F_sn_final} M:{n_M_sn_final})")
        print(f"  Ratio compression: {ratio:.3f}  ({(1-ratio)*100:.0f}% de reduction)")
        print(f"  Fusions ok / rej : {n_accepted} / {n_rejected}")
        print(f"  Temps            : {elapsed:.1f}s")
        print(f"\n  Metrique     Reference (Borda)    Notre methode")
        print(f"  delta_E      {ref_dE:.4f}             {final_dE:.4f}")
        print(f"  ILD          {ref_ILD:.4f}             {final_ILD:.4f}")
        print(f"  inc_F        {ref_incF:.4f}             {final_incF:.4f}")
        print(f"  inc_M        {'N/A':>6}             {final_incM:.4f}")
        print(sep)

    return dict(
        top_k=final_top_k,
        dE=final_dE, ILD=final_ILD, inc_F=final_incF, inc_M=final_incM,
        ratio=ratio,
        n_supernodes=len(active_sns), n_F_sn=n_F_sn_final, n_M_sn=n_M_sn_final,
        n_initial=n_initial, n_accepted=n_accepted, n_rejected=n_rejected,
        elapsed_s=elapsed,
        ref_dE=ref_dE, ref_ILD=ref_ILD, ref_inc_F=ref_incF,
    )


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement de MovieLens 100k...")
    ratings, users = load_movielens()
    gender_map = users.set_index("user_id")["gender"].to_dict()
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = sum(1 for g in gender_map.values() if g == "M")
    print(f"  {len(ratings):,} ratings | {len(gender_map)} users "
          f"({n_F} F / {n_M} M) | {ratings['item_id'].nunique()} items\n")

    result = edi_coarsening(
        ratings, gender_map,
        k=10, theta=4.0,
        eps_E=0.10,   # on tolere +0.10 sur ΔE (par rapport a Borda)
        eps_D=0.05,   # on tolere -0.05 sur ILD
        eps_I=0.05,   # on tolere -0.05 sur inc_F
        max_merges=472,
        intragroup_first=True,
        verbose=True,
    )
