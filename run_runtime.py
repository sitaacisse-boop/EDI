"""
Mesure du temps d'execution de chaque methode pour k in {5, 10, 20}.
Produit runtime_results.json + affiche un tableau recapitulatif.
"""
import json, time, sys, os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")

# ── Chargement ────────────────────────────────────────────────────────────────
print("Chargement MovieLens 100k...")
ratings = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep="\t", header=None,
                      names=["user_id", "item_id", "rating", "timestamp"])
users   = pd.read_csv(os.path.join(DATA_DIR, "u.user"), sep="|", header=None,
                      names=["user_id", "age", "gender", "occupation", "zip"])
gender_map = users.set_index("user_id")["gender"].to_dict()
THETA = 4.0
K_VALUES = [5, 10, 20]
N_RUNS = 3  # répétitions pour stabiliser la mesure

# ── Méthodes ──────────────────────────────────────────────────────────────────
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
        items = g["item_id"].values; scores = g["rating"].values
        if len(items) < 2: continue
        wp = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, iid in enumerate(items):
            w[iid] = w.get(iid, 0) + int(wp[i])
    return [i for i, _ in sorted(w.items(), key=lambda x: x[1], reverse=True)[:k]]

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
        sub = ratings[(ratings["item_id"] == iid) & (ratings["user_id"].isin(gu)) & (ratings["rating"] >= theta)]
        return len(sub["user_id"].unique()) / (len(gu) + 1e-9)
    item_sf = {iid: item_score(iid, f_u) for iid in candidates}
    item_sm = {iid: item_score(iid, m_u) for iid in candidates}
    item_if = {iid: item_inc(iid, f_u) for iid in candidates}
    item_im = {iid: item_inc(iid, m_u) for iid in candidates}
    selected = []; remaining = list(candidates)
    for _ in range(k):
        if not remaining: break
        def gain(iid):
            sel = selected + [iid]
            sf = sum(item_sf[i] for i in sel) / len(sel)
            sm = sum(item_sm[i] for i in sel) / len(sel)
            return (sum(item_if[i] for i in sel) / len(sel) +
                    sum(item_im[i] for i in sel) / len(sel) - abs(sf - sm))
        best = max(remaining, key=gain)
        selected.append(best); remaining.remove(best)
    return selected

def run_coarsening(ratings, gender_map, k, theta, max_merges=472):
    from edi_coarsening import precompute_edi, fast_delta_E, fast_ILD, fast_inc_F
    from itertools import combinations as comb
    cache = precompute_edi(ratings, gender_map, theta)
    users_list = list(gender_map.keys())
    ur = {u: dict(zip(g["item_id"], g["rating"])) for u, g in ratings.groupby("user_id")}
    sn_s = {u: dict(ur.get(u, {})) for u in users_list}
    sn_c = {u: {i: 1 for i in ur.get(u, {})} for u in users_list}
    sn_m = {u: {u} for u in users_list}
    u2s = {u: u for u in users_list}
    active = set(users_list)
    sb = {}; cb = {}
    for u in users_list:
        avgs = {i: sn_s[u][i] / sn_c[u][i] for i in sn_s[u]}
        ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked); b = {iid: n - r for r, (iid, _) in enumerate(ranked)}
        sb[u] = b
        for iid, sc in b.items(): cb[iid] = cb.get(iid, 0) + sc
    def gtk():
        return sorted({i: cb[i] for i in cb if cb[i] > 0}, key=cb.get, reverse=True)[:k]
    ref = gtk(); rdE = fast_delta_E(ref, cache)
    rILD = fast_ILD(ref, cache); riF = fast_inc_F(ref, cache)
    cur = list(ref)
    M = cache["pivot"].values.astype(np.float32)
    nu = np.linalg.norm(M, axis=1, keepdims=True); nu[nu == 0] = 1
    sim = (M / nu) @ (M / nu).T
    ul = list(cache["pivot"].index); nu2 = len(ul); pairs = []
    for i in range(nu2):
        for j in range(i + 1, nu2):
            sv = float(sim[i, j])
            if sv > 0:
                pairs.append((1 if gender_map[ul[i]] == gender_map[ul[j]] else 0, sv, ul[i], ul[j]))
    pairs.sort(reverse=True); na = 0
    for _, sv, ua, ub in pairs:
        if na >= max_merges: break
        sna = u2s.get(ua); snb = u2s.get(ub)
        if sna is None or snb is None or sna == snb: continue
        ia = set(sn_s[sna].keys()); ib = set(sn_s[snb].keys())
        ms = {i: sn_s[sna].get(i, 0) + sn_s[snb].get(i, 0) for i in ia | ib}
        mc = {i: sn_c[sna].get(i, 0) + sn_c[snb].get(i, 0) for i in ia | ib}
        mavgs = {i: ms[i] / mc[i] for i in ms}
        rk = sorted(mavgs.items(), key=lambda x: x[1], reverse=True)
        nm = len(rk); bm = {iid: nm - r for r, (iid, _) in enumerate(rk)}
        for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in bm.items(): cb[iid] = cb.get(iid, 0) + sc
        nt = gtk()
        if nt == cur: ok = True
        else: ok = (fast_delta_E(nt, cache) <= rdE + 0.10 and
                    fast_ILD(nt, cache) >= rILD - 0.05 and
                    fast_inc_F(nt, cache) >= riF - 0.05)
        if ok:
            sn_s[sna] = ms; sn_c[sna] = mc; sb[sna] = bm
            for u in sn_m[snb]: u2s[u] = sna
            sn_m[sna] |= sn_m[snb]
            del sn_s[snb], sn_c[snb], sb[snb], sn_m[snb]; active.discard(snb)
            if nt != cur: cur = nt
            na += 1
        else:
            for iid, sc in bm.items(): cb[iid] = cb.get(iid, 0) - sc
            for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) + sc
            for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) + sc

# ── Mesure du temps ───────────────────────────────────────────────────────────
METHODS = {
    "Average Score": lambda k: top_k_avg(ratings, k),
    "Borda":         lambda k: top_k_borda(ratings, k),
    "Condorcet":     lambda k: top_k_condorcet(ratings, k),
    "Fair Re-rank":  lambda k: top_k_fair_rerank(ratings, gender_map, k, THETA),
    "AURORA":          lambda k: run_coarsening(ratings, gender_map, k, THETA),
}

runtime_results = {}
sep = "=" * 62
print(f"\n{sep}")
print(f"  {'Méthode':<16} {'k=5':>8} {'k=10':>8} {'k=20':>8}  (secondes)")
print(f"  {'-'*54}")

for method, fn in METHODS.items():
    row = {}
    times_by_k = []
    for k in K_VALUES:
        runs = []
        print(f"  {method} k={k}...", end="", flush=True)
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            fn(k)
            runs.append(time.perf_counter() - t0)
        avg = round(sum(runs) / len(runs), 3)
        row[f"k{k}"] = avg
        times_by_k.append(f"{avg:>7.3f}s")
        print(f" {avg:.3f}s", end="", flush=True)
    print()
    runtime_results[method] = row

print(f"\n{sep}")
print(f"  {'Méthode':<16} {'k=5':>8} {'k=10':>8} {'k=20':>8}")
print(f"  {'-'*54}")
for method, row in runtime_results.items():
    vals = " ".join(f"{row[f'k{k}']:>7.3f}s" for k in K_VALUES)
    print(f"  {method:<16} {vals}")
print(sep)

with open(os.path.join(os.path.dirname(__file__), "runtime_results.json"), "w") as f:
    json.dump({"theta": THETA, "n_runs": N_RUNS,
               "dataset": "MovieLens 100k", "results": runtime_results}, f, indent=2)
print("\nSauvegarde : runtime_results.json")
