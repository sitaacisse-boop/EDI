"""
Experiences EDI sur libimseti.cz (site de rencontres tcheque)
Dataset : 17.3M ratings de profils par des utilisateurs (echelle 1-10)
Particularite : les items recommandes SONT des personnes avec un genre (M/F)
Sous-echantillon stratifie : N_USERS=1000 rateurs (500 F / 500 M)
k=10, theta=7.0 (equivalent semantique de 4/5 sur MovieLens)
Resultats -> libimseti_results.json
"""
import json, sys, os, time
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from itertools import combinations as comb

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data", "libimseti")
EDGES_FILE = os.path.join(DATA_DIR, "rec-libimseti-dir.edges")
GENDER_FILE = os.path.join(DATA_DIR, "gender.dat")
SEED      = 42
N_USERS   = 1000   # sous-echantillon de rateurs (500 F + 500 M)
K_VALUES  = [5, 10, 20]
THETA     = 7.0    # sur echelle 1-10 (equivalent de 4.0/5 sur MovieLens)

def load():
    print("Chargement gender.dat...")
    gender = pd.read_csv(GENDER_FILE, header=None, names=["user_id", "gender"])
    gmap_full = gender.set_index("user_id")["gender"].to_dict()

    print("Chargement ratings (17M lignes)...")
    ratings_full = pd.read_csv(EDGES_FILE, sep="\t", comment="%", header=None,
                               names=["user_id", "profile_id", "rating"])
    return ratings_full, gmap_full

def filter_and_sample(ratings_full, gmap_full, n_users=N_USERS, seed=SEED):
    """Garde M/F uniquement, sous-echantillonne n_users rateurs (50/50)."""
    ratings_full["rater_g"]   = ratings_full["user_id"].map(gmap_full)
    ratings_full["profile_g"] = ratings_full["profile_id"].map(gmap_full)
    filt = ratings_full[
        ratings_full["rater_g"].isin(["M","F"]) &
        ratings_full["profile_g"].isin(["M","F"])
    ].copy()

    rng = np.random.default_rng(seed)
    f_ids = filt[filt["rater_g"]=="F"]["user_id"].unique()
    m_ids = filt[filt["rater_g"]=="M"]["user_id"].unique()
    n_F = n_users // 2
    n_M = n_users - n_F
    sel_F = rng.choice(f_ids, size=min(n_F, len(f_ids)), replace=False)
    sel_M = rng.choice(m_ids, size=min(n_M, len(m_ids)), replace=False)
    sel   = set(sel_F) | set(sel_M)

    ratings = filt[filt["user_id"].isin(sel)].drop(columns=["rater_g","profile_g"]).copy()
    ratings.columns = ["user_id", "profile_id", "rating"]
    gender_map = {u: gmap_full[u] for u in sel}
    return ratings, gender_map

# ── Methodes de recommandation (adaptees pour echelle 1-10) ─────────────────

def top_k_avg(ratings, k):
    return list(ratings.groupby("profile_id")["rating"].mean().nlargest(k).index)

def top_k_borda(ratings, k):
    def bs(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["b"] = range(len(r), 0, -1)
        return r[["profile_id","b"]]
    b = ratings.groupby("user_id", group_keys=False).apply(bs)
    return list(b.groupby("profile_id")["b"].sum().nlargest(k).index)

def top_k_condorcet(ratings, k):
    w = {}
    for _, g in ratings.groupby("user_id"):
        items  = g["profile_id"].values
        scores = g["rating"].values
        if len(items) < 2: continue
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
        return ranked[["profile_id","borda","gender"]]
    borda = r.groupby("user_id", group_keys=False).apply(borda_user)
    scores_F = borda[borda["gender"]=="F"].groupby("profile_id")["borda"].sum() / n_F
    scores_M = borda[borda["gender"]=="M"].groupby("profile_id")["borda"].sum() / n_M
    return list(scores_F.add(scores_M, fill_value=0).nlargest(k).index)

def top_k_fair_rerank(ratings, gender_map, k, theta, pool=40):
    def bs(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["b"] = range(len(r), 0, -1)
        return r[["profile_id","b"]]
    b = ratings.groupby("user_id", group_keys=False).apply(bs)
    candidates = list(b.groupby("profile_id")["b"].sum().nlargest(pool).index)
    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]
    def item_score(iid, gu):
        sub = ratings[(ratings["profile_id"]==iid) & (ratings["user_id"].isin(gu))]
        return float(sub["rating"].sum()) / (len(gu) + 1e-9)
    def item_inc(iid, gu):
        sub = ratings[(ratings["profile_id"]==iid) & (ratings["user_id"].isin(gu))
                      & (ratings["rating"] >= theta)]
        return len(sub["user_id"].unique()) / (len(gu) + 1e-9)
    item_sf = {iid: item_score(iid, f_u) for iid in candidates}
    item_sm = {iid: item_score(iid, m_u) for iid in candidates}
    item_if = {iid: item_inc(iid, f_u)   for iid in candidates}
    item_im = {iid: item_inc(iid, m_u)   for iid in candidates}
    selected = []; remaining = list(candidates)
    for _ in range(k):
        if not remaining: break
        def gain(iid):
            sel = selected + [iid]
            sf = sum(item_sf[i] for i in sel) / len(sel)
            sm = sum(item_sm[i] for i in sel) / len(sel)
            return (sum(item_if[i] for i in sel)/len(sel) +
                    sum(item_im[i] for i in sel)/len(sel) - abs(sf-sm))
        best = max(remaining, key=gain)
        selected.append(best); remaining.remove(best)
    return selected

def edi_metrics(ratings, top_k, gender_map, theta):
    """
    Metriques EDI adaptees pour libimseti :
    - 'item_id' s'appelle ici 'profile_id'
    - meme formule que MovieLens
    """
    s = set(top_k); k = len(top_k)
    f_u = [u for u, g in gender_map.items() if g == "F"]
    m_u = [u for u, g in gender_map.items() if g == "M"]
    def util(gu):
        gr = ratings[ratings["user_id"].isin(gu) & ratings["profile_id"].isin(s)]
        return gr["rating"].sum() / (len(gu) * k) if (gu and k) else 0.0
    dE = abs(util(f_u) - util(m_u))
    pivot = ratings[ratings["profile_id"].isin(s)].pivot_table(
        index="user_id", columns="profile_id", values="rating", fill_value=0)
    vecs = {iid: pivot[iid].values if iid in pivot.columns
            else np.zeros(len(pivot)) for iid in top_k}
    tot, np_ = 0.0, 0
    for a, b in comb(top_k, 2):
        va, vb = vecs[a], vecs[b]
        n = np.linalg.norm(va) * np.linalg.norm(vb)
        tot += 1 - (np.dot(va,vb)/n if n > 0 else 0.0); np_ += 1
    ild = tot/np_ if np_ > 0 else 0.0
    def inc(gu):
        gr = ratings[ratings["user_id"].isin(gu) & ratings["profile_id"].isin(s)
                     & (ratings["rating"] >= theta)]
        return (gr.groupby("user_id")["profile_id"].nunique()
                .reindex(gu, fill_value=0).sum() / (len(gu)*k))
    return dict(dE=round(dE,4), ILD=round(ild,4),
                inc_F=round(inc(f_u),4), inc_M=round(inc(m_u),4))

def run_coarsening(ratings, gender_map, k, theta, max_merges=500):
    import heapq
    from edi_coarsening import precompute_edi, fast_delta_E, fast_ILD, fast_inc_F
    # Renommer profile_id -> item_id pour compatibilite avec edi_coarsening
    r2 = ratings.rename(columns={"profile_id": "item_id"})
    t0 = time.time()
    cache = precompute_edi(r2, gender_map, theta)
    users_list = list(gender_map.keys())
    ur = {u: dict(zip(g["item_id"], g["rating"]))
          for u, g in r2.groupby("user_id")}
    sn_s = {u: dict(ur.get(u,{})) for u in users_list}
    sn_c = {u: {i:1 for i in ur.get(u,{})} for u in users_list}
    sn_m = {u: {u} for u in users_list}
    u2s  = {u: u  for u in users_list}
    active = set(users_list)
    sb = {}; cb = {}
    for u in users_list:
        avgs   = {i: sn_s[u][i]/sn_c[u][i] for i in sn_s[u]}
        ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
        n = len(ranked)
        b = {iid: n-r for r,(iid,_) in enumerate(ranked)}
        sb[u] = b
        for iid, sc in b.items(): cb[iid] = cb.get(iid,0) + sc
    # heapq.nlargest est O(n + k log k) au lieu de sorted O(n log n) — 15x plus rapide
    def gtk():
        return [i for _, i in heapq.nlargest(k, ((sc, i) for i, sc in cb.items() if sc > 0))]
    ref  = gtk(); rdE = fast_delta_E(ref, cache)
    rILD = fast_ILD(ref, cache); riF = fast_inc_F(ref, cache); cur = list(ref)
    M2 = cache["pivot"].values.astype(np.float32)
    nu = np.linalg.norm(M2, axis=1, keepdims=True); nu[nu==0] = 1
    sim = (M2/nu) @ (M2/nu).T
    ul  = list(cache["pivot"].index); nu2 = len(ul); pairs = []
    for i in range(nu2):
        for j in range(i+1, nu2):
            sv = float(sim[i,j])
            if sv > 0.05:  # seuil minimal de similarite pour reduire le nombre de paires
                pairs.append((1 if gender_map[ul[i]]==gender_map[ul[j]] else 0, sv, ul[i], ul[j]))
    pairs.sort(reverse=True)
    print(f"    {len(pairs):,} paires similaires (sv>0.05)")
    na = 0
    for _, sv, ua, ub in pairs:
        if na >= max_merges: break
        sna = u2s.get(ua); snb = u2s.get(ub)
        if sna is None or snb is None or sna == snb: continue
        ia = set(sn_s[sna].keys()); ib = set(sn_s[snb].keys())
        ms = {i: sn_s[sna].get(i,0)+sn_s[snb].get(i,0) for i in ia|ib}
        mc = {i: sn_c[sna].get(i,0)+sn_c[snb].get(i,0) for i in ia|ib}
        mavgs = {i: ms[i]/mc[i] for i in ms}
        rk = sorted(mavgs.items(), key=lambda x: x[1], reverse=True)
        nm = len(rk); bm = {iid: nm-r for r,(iid,_) in enumerate(rk)}
        for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid,0) - sc
        for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid,0) - sc
        for iid, sc in bm.items():      cb[iid] = cb.get(iid,0) + sc
        nt = gtk()
        if nt == cur: ok = True
        else:
            ok = (fast_delta_E(nt,cache) <= rdE+0.02 and
                  fast_ILD(nt,cache)     >= rILD-0.05 and
                  fast_inc_F(nt,cache)   >= riF-0.05)
        if ok:
            sn_s[sna]=ms; sn_c[sna]=mc; sb[sna]=bm
            for u in sn_m[snb]: u2s[u]=sna
            sn_m[sna] |= sn_m[snb]
            del sn_s[snb], sn_c[snb], sb[snb], sn_m[snb]
            active.discard(snb)
            if nt != cur: cur = nt
            na += 1
        else:
            for iid, sc in bm.items():      cb[iid] = cb.get(iid,0) - sc
            for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid,0) + sc
            for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid,0) + sc
    final = gtk()
    m = edi_metrics(r2.rename(columns={"item_id":"profile_id"}), final, gender_map, theta)
    m["ratio"]        = round(len(active)/len(users_list), 3)
    m["n_supernodes"] = len(active)
    m["elapsed_s"]    = round(time.time()-t0, 3)
    # Conserver les noms de profils recommandes (pour analyse du genre des profils)
    return m

if __name__ == "__main__":
    print("=== Experiences EDI — libimseti.cz ===")
    print(f"N_USERS={N_USERS}, theta={THETA}, k={K_VALUES}, seed={SEED}")

    ratings_full, gmap_full = load()

    print(f"\nFiltrage et sous-echantillonnage ({N_USERS} rateurs)...")
    ratings, gender_map = filter_and_sample(ratings_full, gmap_full)
    del ratings_full  # liberer memoire

    n_F = sum(1 for g in gender_map.values() if g=="F")
    n_M = len(gender_map) - n_F
    alpha_F = round(n_F/len(gender_map), 4)
    alpha_M = round(n_M/len(gender_map), 4)

    print(f"  {len(ratings):,} ratings | {n_F} F / {n_M} M rateurs")
    print(f"  {ratings['profile_id'].nunique():,} profils uniques")
    print(f"  alpha_F={alpha_F}, alpha_M={alpha_M}")

    results = {
        "meta": {
            "dataset": "libimseti.cz",
            "n_users": len(gender_map),
            "n_F": n_F, "n_M": n_M,
            "alpha_F": alpha_F, "alpha_M": alpha_M,
            "n_ratings": len(ratings),
            "n_profiles": int(ratings["profile_id"].nunique()),
            "k_values": K_VALUES,
            "theta": THETA,
            "seed": SEED,
            "note": "items = profils de personnes (genre M/F explicite)"
        },
        "experiments": {}
    }

    METHODS = ["Average Score","Borda","Weighted Borda","Condorcet","Fair Re-rank","AURORA"]

    for k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"k = {k}")
        key = f"k{k}_t{str(THETA).replace('.','_')}"
        exp = {"k": k, "theta": THETA}

        for method_name, fn in [
            ("Average Score", lambda: top_k_avg(ratings, k)),
            ("Borda",         lambda: top_k_borda(ratings, k)),
            ("Weighted Borda",lambda: top_k_weighted_borda(ratings, gender_map, k)),
            ("Condorcet",     lambda: top_k_condorcet(ratings, k)),
            ("Fair Re-rank",  lambda: top_k_fair_rerank(ratings, gender_map, k, THETA)),
        ]:
            print(f"  {method_name}...", end="", flush=True)
            t0 = time.time()
            items = fn()
            rt = time.time() - t0
            m = edi_metrics(ratings, items, gender_map, THETA)
            m["elapsed_s"] = round(rt, 4)
            exp[method_name] = m
            print(f" {rt:.3f}s | dE={m['dE']:.4f} ILD={m['ILD']:.4f}")

        print(f"  AURORA (coarsening, max_merges=500)...", end="", flush=True)
        m = run_coarsening(ratings, gender_map, k, THETA, max_merges=500)
        exp["AURORA"] = m
        print(f" {m['elapsed_s']}s => {m['n_supernodes']} super-noeuds | dE={m['dE']:.4f} ILD={m['ILD']:.4f}")

        results["experiments"][key] = exp

        print(f"\n  {'Methode':<16} {'dE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7} {'temps':>10}")
        for meth in METHODS:
            v = exp[meth]
            print(f"  {meth:<16} {v['dE']:>7.4f} {v['ILD']:>7.4f} {v['inc_F']:>7.4f} {v['inc_M']:>7.4f} {v['elapsed_s']:>9.3f}s")

    out = os.path.join(os.path.dirname(__file__), "libimseti_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSauvegarde : {out}")
