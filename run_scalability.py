"""
Expérience de scalabilité — MovieLens 1M
Pour n_users ∈ {500, 1000, 2000, 4000, 6040}, échantillon stratifié par genre,
6 méthodes à k=10, θ=4.0. Mesure runtime + métriques EDI.
Résultats → scalability_results.json
"""
import json, sys, os, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from itertools import combinations as comb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-1m")
SEED  = 42
SIZES = [500, 1000, 2000, 4000, 6040]
K     = 10
THETA = 4.0

# ── Chargement ────────────────────────────────────────────────────────────────
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

def stratified_sample(users_df, n_users, seed=SEED):
    """Échantillon stratifié par genre — maintient le ratio F/M."""
    rng = np.random.default_rng(seed)
    f_ids = users_df[users_df["gender"] == "F"]["user_id"].values
    m_ids = users_df[users_df["gender"] == "M"]["user_id"].values
    n_F = round(n_users * len(f_ids) / len(users_df))
    n_M = n_users - n_F
    sel_F = rng.choice(f_ids, size=min(n_F, len(f_ids)), replace=False)
    sel_M = rng.choice(m_ids, size=min(n_M, len(m_ids)), replace=False)
    return set(sel_F) | set(sel_M)

# ── Méthodes d'agrégation ─────────────────────────────────────────────────────
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
        items  = g["item_id"].values
        scores = g["rating"].values
        if len(items) < 2:
            continue
        wp = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, iid in enumerate(items):
            w[iid] = w.get(iid, 0) + int(wp[i])
    return [i for i, _ in sorted(w.items(), key=lambda x: x[1], reverse=True)[:k]]

def top_k_weighted_borda(ratings, gender_map, k):
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
    selected = []; remaining = list(candidates)
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
        selected.append(best); remaining.remove(best)
    return selected

# ── Métriques EDI ─────────────────────────────────────────────────────────────
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
    return dict(dE=round(dE,4), ILD=round(ild,4),
                inc_F=round(inc(f_u),4), inc_M=round(inc(m_u),4))

# ── Coarsening ────────────────────────────────────────────────────────────────
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
    u2s  = {u: u  for u in users_list}
    active = set(users_list)
    sb = {}; cb = {}
    for u in users_list:
        avgs   = {i: sn_s[u][i] / sn_c[u][i] for i in sn_s[u]}
        ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        b = {iid: n - r for r, (iid, _) in enumerate(ranked)}
        sb[u] = b
        for iid, sc in b.items():
            cb[iid] = cb.get(iid, 0) + sc
    def gtk():
        return sorted({i: cb[i] for i in cb if cb[i] > 0},
                      key=cb.get, reverse=True)[:k]
    ref  = gtk()
    rdE  = fast_delta_E(ref, cache)
    rILD = fast_ILD(ref, cache)
    riF  = fast_inc_F(ref, cache)
    cur  = list(ref)
    M  = cache["pivot"].values.astype(np.float32)
    nu = np.linalg.norm(M, axis=1, keepdims=True); nu[nu == 0] = 1
    sim = (M / nu) @ (M / nu).T
    ul  = list(cache["pivot"].index)
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
        ms = {i: sn_s[sna].get(i,0) + sn_s[snb].get(i,0) for i in ia | ib}
        mc = {i: sn_c[sna].get(i,0) + sn_c[snb].get(i,0) for i in ia | ib}
        mavgs = {i: ms[i]/mc[i] for i in ms}
        rk = sorted(mavgs.items(), key=lambda x: x[1], reverse=True)
        nm = len(rk)
        bm = {iid: nm - r for r, (iid, _) in enumerate(rk)}
        for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid,0) - sc
        for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid,0) - sc
        for iid, sc in bm.items():      cb[iid] = cb.get(iid,0) + sc
        nt = gtk()
        if nt == cur:
            ok = True
        else:
            ok = (fast_delta_E(nt, cache) <= rdE  + 0.10 and
                  fast_ILD(nt, cache)     >= rILD - 0.05 and
                  fast_inc_F(nt, cache)   >= riF  - 0.05)
        if ok:
            sn_s[sna] = ms; sn_c[sna] = mc; sb[sna] = bm
            for u in sn_m[snb]: u2s[u] = sna
            sn_m[sna] |= sn_m[snb]
            del sn_s[snb], sn_c[snb], sb[snb], sn_m[snb]
            active.discard(snb)
            if nt != cur:
                cur = nt; na += 1
        else:
            for iid, sc in bm.items():      cb[iid] = cb.get(iid,0) - sc
            for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid,0) + sc
            for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid,0) + sc
    final = gtk()
    m = edi_metrics(ratings, final, gender_map, theta)
    m["ratio"]       = round(len(active) / len(users_list), 3)
    m["n_supernodes"] = len(active)
    m["elapsed_s"]   = round(time.time() - t0, 3)
    return m

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Chargement MovieLens 1M...")
    ratings_full, users_df = load()
    gender_map_full = users_df.set_index("user_id")["gender"].to_dict()
    print(f"  {len(ratings_full):,} ratings | {len(gender_map_full)} utilisateurs")

    results = {
        "meta": {
            "dataset": "MovieLens 1M (sous-ensembles)",
            "k": K, "theta": THETA, "seed": SEED,
            "sizes": SIZES,
            "description": "Scalabilité : temps d'exécution et métriques EDI vs. n_users"
        },
        "experiments": {}
    }

    METHODS = ["Average Score","Borda","Weighted Borda","Condorcet","Fair Re-rank","AURORA"]

    for n_users in SIZES:
        print(f"\n{'='*60}")
        print(f"n_users = {n_users}")
        if n_users == max(SIZES):
            sel_ids = set(gender_map_full.keys())
        else:
            sel_ids = stratified_sample(users_df, n_users)

        ratings   = ratings_full[ratings_full["user_id"].isin(sel_ids)].copy()
        gender_map = {u: g for u, g in gender_map_full.items() if u in sel_ids}
        n_F   = sum(1 for g in gender_map.values() if g == "F")
        n_M   = len(gender_map) - n_F
        n_items = ratings["item_id"].nunique()
        print(f"  {len(ratings):,} notes | {n_F} F / {n_M} M | {n_items} items")

        exp = {"n_users": len(gender_map), "n_F": n_F, "n_M": n_M,
               "n_items": n_items, "n_ratings": len(ratings)}

        # Average Score
        print(f"  Average Score...", end="", flush=True)
        t0 = time.time(); items = top_k_avg(ratings, K); rt = time.time()-t0
        m = edi_metrics(ratings, items, gender_map, THETA); m["elapsed_s"] = round(rt, 4)
        exp["Average Score"] = m; print(f" {rt:.3f}s")

        # Borda
        print(f"  Borda...", end="", flush=True)
        t0 = time.time(); items = top_k_borda(ratings, K); rt = time.time()-t0
        m = edi_metrics(ratings, items, gender_map, THETA); m["elapsed_s"] = round(rt, 4)
        exp["Borda"] = m; print(f" {rt:.3f}s")

        # Weighted Borda
        print(f"  Weighted Borda...", end="", flush=True)
        t0 = time.time(); items = top_k_weighted_borda(ratings, gender_map, K); rt = time.time()-t0
        m = edi_metrics(ratings, items, gender_map, THETA); m["elapsed_s"] = round(rt, 4)
        exp["Weighted Borda"] = m; print(f" {rt:.3f}s")

        # Condorcet
        print(f"  Condorcet...", end="", flush=True)
        t0 = time.time(); items = top_k_condorcet(ratings, K); rt = time.time()-t0
        m = edi_metrics(ratings, items, gender_map, THETA); m["elapsed_s"] = round(rt, 4)
        exp["Condorcet"] = m; print(f" {rt:.3f}s")

        # Fair Re-rank
        print(f"  Fair Re-rank...", end="", flush=True)
        t0 = time.time(); items = top_k_fair_rerank(ratings, gender_map, K, THETA); rt = time.time()-t0
        m = edi_metrics(ratings, items, gender_map, THETA); m["elapsed_s"] = round(rt, 4)
        exp["Fair Re-rank"] = m; print(f" {rt:.3f}s")

        # AURORA — max_merges proportionnel à n_users pour éviter le paradoxe
        # (petit dataset = top-k très stable = plus de paires à parcourir)
        max_m = max(50, round(400 * len(gender_map) / 6040))
        print(f"  AURORA (coarsening, max_merges={max_m})...", end="", flush=True)
        m = run_coarsening(ratings, gender_map, K, THETA, max_merges=max_m)
        exp["AURORA"] = m
        print(f" {m['elapsed_s']}s => {m['n_supernodes']} super-noeuds "
              f"(ratio {m['ratio']:.1%})")

        results["experiments"][str(n_users)] = exp

        # Résumé
        print(f"\n  {'Méthode':<16} {'ΔE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7} {'temps':>10}")
        for meth in METHODS:
            v = exp[meth]
            print(f"  {meth:<16} {v['dE']:>7.4f} {v['ILD']:>7.4f} "
                  f"{v['inc_F']:>7.4f} {v['inc_M']:>7.4f} {v['elapsed_s']:>9.3f}s")

    out = os.path.join(os.path.dirname(__file__), "scalability_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSauvegardé : {out}")
