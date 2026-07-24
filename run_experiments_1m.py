"""
Expériences EDI sur MovieLens 1M (6040 users, 3952 films, 1M ratings).
Même protocole que run_all_experiments.py — uniquement le chargement change.
Résultats → experiments_results_1m.json
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from itertools import combinations as comb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-1m")


# ── 1. Chargement ─────────────────────────────────────────────────────────────
def load():
    ratings = pd.read_csv(
        os.path.join(DATA_DIR, "ratings.dat"),
        sep="::", header=None, engine="python",
        names=["user_id", "item_id", "rating", "timestamp"]
    )
    users = pd.read_csv(
        os.path.join(DATA_DIR, "users.dat"),
        sep="::", header=None, engine="python",
        names=["user_id", "gender", "age", "occupation", "zip"]
    )
    return ratings, users


# ── 2. Baselines ──────────────────────────────────────────────────────────────
def top_k_avg(ratings, k):
    return list(ratings.groupby("item_id")["rating"].mean().nlargest(k).index)


def top_k_borda(ratings, k):
    def bs(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["b"] = range(len(r), 0, -1)
        return r[["item_id", "b"]]
    b = ratings.groupby("user_id", group_keys=False).apply(bs)
    return list(b.groupby("item_id")["b"].sum().nlargest(k).index)


def top_k_condorcet(ratings, k):
    w = {}
    for _, g in ratings.groupby("user_id"):
        items = g["item_id"].values
        scores = g["rating"].values
        if len(items) < 2:
            continue
        wp = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, iid in enumerate(items):
            w[iid] = w.get(iid, 0) + int(wp[i])
    return [i for i, _ in sorted(w.items(), key=lambda x: x[1], reverse=True)[:k]]


def top_k_weighted_borda(ratings, gender_map, k):
    """Weighted Borda : score(i) = Σ_F borda(u,i)/n_F + Σ_M borda(u,i)/n_M"""
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


def top_k_fair_rerank(ratings, gender_map, k, theta, pool=40):
    def bs(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["b"] = range(len(r), 0, -1)
        return r[["item_id", "b"]]
    b = ratings.groupby("user_id", group_keys=False).apply(bs)
    candidates = list(b.groupby("item_id")["b"].sum().nlargest(pool).index)
    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]
    def item_score(iid, gu):
        sub = ratings[(ratings["item_id"] == iid) & (ratings["user_id"].isin(gu))]
        return float(sub["rating"].sum()) / (len(gu) + 1e-9)
    def item_inc(iid, gu):
        sub = ratings[(ratings["item_id"] == iid) & (ratings["user_id"].isin(gu))
                      & (ratings["rating"] >= theta)]
        return len(sub["user_id"].unique()) / (len(gu) + 1e-9)
    item_sf = {iid: item_score(iid, f_u) for iid in candidates}
    item_sm = {iid: item_score(iid, m_u) for iid in candidates}
    item_if = {iid: item_inc(iid, f_u) for iid in candidates}
    item_im = {iid: item_inc(iid, m_u) for iid in candidates}
    selected = []
    remaining = list(candidates)
    for _ in range(k):
        if not remaining:
            break
        def gain(iid):
            sel = selected + [iid]
            sf = sum(item_sf[i] for i in sel) / len(sel)
            sm = sum(item_sm[i] for i in sel) / len(sel)
            return (sum(item_if[i] for i in sel) / len(sel) +
                    sum(item_im[i] for i in sel) / len(sel) - abs(sf - sm))
        best = max(remaining, key=gain)
        selected.append(best)
        remaining.remove(best)
    return selected


# ── 3. Métriques EDI ──────────────────────────────────────────────────────────
def edi_metrics(ratings, top_k, gender_map, theta):
    s = set(top_k)
    k = len(top_k)
    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]

    def util(gu):
        gr = ratings[ratings["user_id"].isin(gu) & ratings["item_id"].isin(s)]
        return gr["rating"].sum() / (len(gu) * k) if (gu and k) else 0.0

    dE = abs(util(f_u) - util(m_u))

    # ILD — vecteurs d'items dans l'espace utilisateurs
    pivot = ratings[ratings["item_id"].isin(s)].pivot_table(
        index="user_id", columns="item_id", values="rating", fill_value=0
    )
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


# ── 4. Coarsening ─────────────────────────────────────────────────────────────
def run_coarsening(ratings, gender_map, k, theta, max_merges=400):
    from edi_coarsening import precompute_edi, fast_delta_E, fast_ILD, fast_inc_F
    t0 = time.time()
    cache = precompute_edi(ratings, gender_map, theta)
    users_list = list(gender_map.keys())
    ur = {u: dict(zip(g["item_id"], g["rating"]))
          for u, g in ratings.groupby("user_id")}
    sn_s = {u: dict(ur.get(u, {})) for u in users_list}
    sn_c = {u: {i: 1 for i in ur.get(u, {})} for u in users_list}
    sn_m = {u: {u} for u in users_list}
    u2s = {u: u for u in users_list}
    active = set(users_list)
    sb = {}; cb = {}
    for u in users_list:
        avgs = {i: sn_s[u][i] / sn_c[u][i] for i in sn_s[u]}
        ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        b = {iid: n - r for r, (iid, _) in enumerate(ranked)}
        sb[u] = b
        for iid, sc in b.items():
            cb[iid] = cb.get(iid, 0) + sc

    def gtk():
        return sorted({i: cb[i] for i in cb if cb[i] > 0},
                      key=cb.get, reverse=True)[:k]

    ref = gtk()
    rdE = fast_delta_E(ref, cache)
    rILD = fast_ILD(ref, cache)
    riF = fast_inc_F(ref, cache)
    cur = list(ref)

    M = cache["pivot"].values.astype(np.float32)
    nu = np.linalg.norm(M, axis=1, keepdims=True); nu[nu == 0] = 1
    sim = (M / nu) @ (M / nu).T
    ul = list(cache["pivot"].index)
    nu2 = len(ul); pairs = []
    for i in range(nu2):
        for j in range(i + 1, nu2):
            sv = float(sim[i, j])
            if sv > 0:
                pairs.append((
                    1 if gender_map[ul[i]] == gender_map[ul[j]] else 0,
                    sv, ul[i], ul[j]
                ))
    pairs.sort(reverse=True)
    na = 0
    for _, sv, ua, ub in pairs:
        if na >= max_merges:
            break
        sna = u2s.get(ua); snb = u2s.get(ub)
        if sna is None or snb is None or sna == snb:
            continue
        ia = set(sn_s[sna].keys()); ib = set(sn_s[snb].keys())
        ms = {i: sn_s[sna].get(i, 0) + sn_s[snb].get(i, 0) for i in ia | ib}
        mc = {i: sn_c[sna].get(i, 0) + sn_c[snb].get(i, 0) for i in ia | ib}
        mavgs = {i: ms[i] / mc[i] for i in ms}
        rk = sorted(mavgs.items(), key=lambda x: x[1], reverse=True)
        nm = len(rk)
        bm = {iid: nm - r for r, (iid, _) in enumerate(rk)}
        for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in bm.items(): cb[iid] = cb.get(iid, 0) + sc
        nt = gtk()
        if nt == cur:
            ok = True
        else:
            ok = (fast_delta_E(nt, cache) <= rdE + 0.10 and
                  fast_ILD(nt, cache) >= rILD - 0.05 and
                  fast_inc_F(nt, cache) >= riF - 0.05)
        if ok:
            sn_s[sna] = ms; sn_c[sna] = mc; sb[sna] = bm
            for u in sn_m[snb]: u2s[u] = sna
            sn_m[sna] |= sn_m[snb]
            del sn_s[snb], sn_c[snb], sb[snb], sn_m[snb]
            active.discard(snb)
            if nt != cur:
                cur = nt; na += 1
        else:
            for iid, sc in bm.items(): cb[iid] = cb.get(iid, 0) - sc
            for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) + sc
            for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) + sc

    final = gtk()
    m = edi_metrics(ratings, final, gender_map, theta)
    m["ratio"] = round(len(active) / len(users_list), 3)
    m["n_supernodes"] = len(active)
    m["elapsed_s"] = round(time.time() - t0, 1)
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement MovieLens 1M...")
    ratings, users = load()
    gender_map = users.set_index("user_id")["gender"].to_dict()
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = len(gender_map) - n_F
    alpha_F = round(n_F / len(gender_map), 4)
    alpha_M = round(n_M / len(gender_map), 4)
    print(f"  {len(ratings):,} ratings | {len(gender_map)} users "
          f"({n_F} F / {n_M} M) | {ratings['item_id'].nunique()} items")

    results = {
        "meta": {
            "dataset": "MovieLens 1M",
            "n_users": len(gender_map), "n_F": n_F, "n_M": n_M,
            "alpha_F": alpha_F, "alpha_M": alpha_M,
            "n_items": int(ratings["item_id"].nunique()),
            "n_ratings": len(ratings),
            "density": round(len(ratings) / (len(gender_map) * ratings["item_id"].nunique()), 4),
            "mean_rating": round(float(ratings["rating"].mean()), 2),
            "std_rating": round(float(ratings["rating"].std()), 2),
        },
        "experiments": {}
    }

    for k in [5, 10, 20]:
        for theta in [4.0]:
            key = f"k{k}_t{str(theta).replace('.', '_')}"
            print(f"\n=== k={k}, theta={theta} ===")
            t0 = time.time()
            print("  Average Score...", end="", flush=True)
            avg_items = top_k_avg(ratings, k)
            print(f" {time.time()-t0:.1f}s")
            t0 = time.time()
            print("  Borda...", end="", flush=True)
            borda_items = top_k_borda(ratings, k)
            print(f" {time.time()-t0:.1f}s")
            t0 = time.time()
            print("  Weighted Borda...", end="", flush=True)
            wb_items = top_k_weighted_borda(ratings, gender_map, k)
            print(f" {time.time()-t0:.1f}s")
            t0 = time.time()
            print("  Condorcet...", end="", flush=True)
            cond_items = top_k_condorcet(ratings, k)
            print(f" {time.time()-t0:.1f}s")
            t0 = time.time()
            print("  Fair Re-rank...", end="", flush=True)
            fair_items = top_k_fair_rerank(ratings, gender_map, k, theta)
            print(f" {time.time()-t0:.1f}s")
            print("  Coarsening...", end="", flush=True)
            ours_m = run_coarsening(ratings, gender_map, k, theta)
            print(f" {ours_m['elapsed_s']}s")

            results["experiments"][key] = {
                "k": k, "theta": theta,
                "Average Score":  edi_metrics(ratings, avg_items,   gender_map, theta),
                "Borda":          edi_metrics(ratings, borda_items,  gender_map, theta),
                "Weighted Borda": edi_metrics(ratings, wb_items,     gender_map, theta),
                "Condorcet":      edi_metrics(ratings, cond_items,   gender_map, theta),
                "Fair Re-rank":   edi_metrics(ratings, fair_items,   gender_map, theta),
                "Ours":           ours_m,
            }
            r = results["experiments"][key]
            print(f"  {'Méthode':<16} {'dE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7}")
            for m in ["Average Score", "Borda", "Weighted Borda", "Condorcet", "Fair Re-rank", "Ours"]:
                v = r[m]
                print(f"  {m:<14} {v['dE']:>7.4f} {v['ILD']:>7.4f} "
                      f"{v['inc_F']:>7.4f} {v['inc_M']:>7.4f}")

    out = os.path.join(os.path.dirname(__file__), "experiments_results_1m.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSauvegarde : {out}")
