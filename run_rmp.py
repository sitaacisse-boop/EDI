"""
Experiences EDI — Rate My Professors (dataset vxuv/ratemyprofessor-dataset)
3.26M avis d'etudiants sur des professeurs (echelle quality 1-5)

Particularite : aucun identifiant etudiant → equite mesuree cote ITEMS (profs)
- user_id = code de cAURORA (ex. ENG101) → les etudiants de ce cAURORA ont note plusieurs profs
- item_id = identifiant professeur
- gender_map = professeur → M/F (infere depuis le prenom, gender_guesser)

Formulation item-side EDI :
  ΔE = |mean_quality(profs_F dans top-k) - mean_quality(profs_M dans top-k)|
  ILD = diversite cosinus des vecteurs de notation des profs
  inclusion_g = fraction profs-g avec mean_quality ≥ θ dans top-k

theta = 4.0  (meme echelle 1-5 que MovieLens)
N_CLASSES = 2000 cAURORA les plus notes (= utilisateurs)
Filtre profs : classes = genre connu (M/F) + avoir ete note dans ≥ 2 cAURORA top

Resultats -> rmp_results.json
"""
import json, sys, os, re, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from itertools import combinations as comb
import gender_guesser.detector as gd

# ─── Parametres ──────────────────────────────────────────────────────────────
DATA_FILE  = os.path.join(os.path.dirname(__file__), "data", "rmp", "reviews.csv")
SEED       = 42
N_CLASSES  = 2000   # nombre de codes-cAURORA (= utilisateurs) a conserver
MIN_PROFS  = 2      # un cAURORA doit avoir note ≥ MIN_PROFS profs differents
MIN_CLASSES = 2     # un prof doit avoir ete note dans ≥ MIN_CLASSES cAURORA du top
K_VALUES   = [10]
THETA      = 4.0    # seuil "bon prof" (1-5, identique a MovieLens)
MAX_MERGES = 400


# ─── 1. Chargement et nettoyage ──────────────────────────────────────────────

def infer_gender_fast(names_series):
    """Infere le genre depuis la serie de prenoms. Retourne un dict id→gender."""
    detector = gd.Detector()
    result = {}
    for fn in names_series.unique():
        if pd.isna(fn):
            result[fn] = "unknown"
            continue
        parts = str(fn).strip().split()
        if not parts:
            result[fn] = "unknown"
            continue
        first = parts[0].capitalize()
        g = detector.get_gender(first)
        result[fn] = g
    return result


def load_and_build(n_classes=N_CLASSES, min_profs=MIN_PROFS, min_classes=MIN_CLASSES):
    print("Chargement reviews.csv (3.26M lignes)...")
    cols = ["id", "first_name", "quality"]
    chunks = []
    for chunk in pd.read_csv(DATA_FILE,
                             usecols=cols + ["class"],
                             chunksize=500_000,
                             dtype={"id": "int32", "quality": "float32"}):
        chunk = chunk.rename(columns={"class": "cAURORAe"})
        chunk = chunk.dropna(subset=["quality", "cAURORAe"])
        chunks.append(chunk)
        print(f"  {sum(len(c) for c in chunks):,} lignes lues...", end="\r")
    df = pd.concat(chunks, ignore_index=True)
    print(f"\n  Total apres nettoyage: {len(df):,} avis")

    # ── Inférence du genre du professeur ─────────────────────────────────────
    print("Inference du genre (gender_guesser)...")
    prof_info = df.groupby("id")["first_name"].first().reset_index()
    g_map_name = infer_gender_fast(prof_info["first_name"])
    prof_info["gender_raw"] = prof_info["first_name"].map(g_map_name)
    prof_info["gender"] = prof_info["gender_raw"].map({
        "male": "M", "mostly_male": "M",
        "female": "F", "mostly_female": "F"
    })
    # Garde uniquement M/F classifies
    valid_profs = set(prof_info.dropna(subset=["gender"])["id"].tolist())
    df = df[df["id"].isin(valid_profs)].copy()
    gender_map_full = prof_info.dropna(subset=["gender"]).set_index("id")["gender"].to_dict()
    n_F = sum(1 for g in gender_map_full.values() if g == "F")
    n_M = len(gender_map_full) - n_F
    print(f"  Profs avec genre clair: {len(gender_map_full):,}  M={n_M:,} F={n_F:,}")

    # ── Sélection des N_CLASSES cAURORA les plus notes ──────────────────────────
    print(f"Selection des {n_classes} cAURORA les plus notes...")
    cAURORAe_counts = df["cAURORAe"].value_counts()
    top_cAURORAes = set(cAURORAe_counts.head(n_classes).index.tolist())
    df = df[df["cAURORAe"].isin(top_cAURORAes)].copy()

    # ── Agregation : (cAURORAe, professor) → mean_quality ─────────────────────
    agg = (df.groupby(["cAURORAe", "id"])["quality"]
              .mean()
              .reset_index()
              .rename(columns={"cAURORAe": "user_id", "id": "item_id",
                               "quality": "rating"}))
    agg["rating"] = agg["rating"].astype(float)

    # ── Filtrage par connectivite ─────────────────────────────────────────────
    print(f"Filtrage : cAURORA avec ≥{min_profs} profs, profs dans ≥{min_classes} cAURORA...")
    for _ in range(3):  # iterations pour stabiliser la matrice
        profs_per_cAURORAe = agg.groupby("user_id")["item_id"].nunique()
        valid_cAURORAes = profs_per_cAURORAe[profs_per_cAURORAe >= min_profs].index
        agg = agg[agg["user_id"].isin(valid_cAURORAes)]
        cAURORAes_per_prof = agg.groupby("item_id")["user_id"].nunique()
        valid_profs2 = cAURORAes_per_prof[cAURORAes_per_prof >= min_classes].index
        agg = agg[agg["item_id"].isin(valid_profs2)]

    gender_map = {p: gender_map_full[p] for p in agg["item_id"].unique()
                  if p in gender_map_full}
    agg = agg[agg["item_id"].isin(gender_map)].copy()

    n_users  = agg["user_id"].nunique()
    n_items  = agg["item_id"].nunique()
    n_F_items = sum(1 for g in gender_map.values() if g == "F")
    n_M_items = len(gender_map) - n_F_items
    alpha_F   = round(n_F_items / len(gender_map), 4)
    alpha_M   = round(n_M_items / len(gender_map), 4)

    print(f"  Matrice finale: {n_users} cAURORA × {n_items} profs, {len(agg):,} ratings")
    print(f"  Profs F={n_F_items} ({alpha_F*100:.1f}%), M={n_M_items} ({alpha_M*100:.1f}%)")

    return agg, gender_map, alpha_F, alpha_M


# ─── 2. Méthodes de recommandation ───────────────────────────────────────────

def top_k_avg(ratings, k):
    """Average Score : moyenne globale de quality par prof."""
    return list(ratings.groupby("item_id")["rating"].mean().nlargest(k).index)


def top_k_borda(ratings, k):
    """Borda : chaque cAURORA-utilisateur vote par rang."""
    def borda_user(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["borda"] = range(len(r), 0, -1)
        return r[["item_id", "borda"]]
    borda = ratings.groupby("user_id", group_keys=False).apply(borda_user)
    return list(borda.groupby("item_id")["borda"].sum().nlargest(k).index)


def top_k_weighted_borda(ratings, gender_map, k):
    """Weighted Borda : pondere par la proportion du genre du prof dans le classement."""
    n_F = sum(1 for g in gender_map.values() if g == "F")
    n_M = len(gender_map) - n_F
    def borda_user(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["borda"] = range(len(r), 0, -1)
        r["gender"] = r["item_id"].map(gender_map)
        return r[["item_id", "borda", "gender"]]
    borda = ratings.groupby("user_id", group_keys=False).apply(borda_user)
    scores_F = borda[borda["gender"]=="F"].groupby("item_id")["borda"].sum() / (n_F or 1)
    scores_M = borda[borda["gender"]=="M"].groupby("item_id")["borda"].sum() / (n_M or 1)
    return list(scores_F.add(scores_M, fill_value=0).nlargest(k).index)


def top_k_condorcet(ratings, k):
    """Condorcet : comptage des victoires pairwise par cAURORA."""
    wins = {}
    for _, g in ratings.groupby("user_id"):
        items  = g["item_id"].values
        scores = g["rating"].values
        if len(items) < 2:
            continue
        wp = np.sum(scores[:, None] > scores[None, :], axis=1)
        for i, iid in enumerate(items):
            wins[iid] = wins.get(iid, 0) + int(wp[i])
    return [i for i, _ in sorted(wins.items(), key=lambda x: x[1], reverse=True)[:k]]


def top_k_fair_rerank(ratings, gender_map, k, theta, pool=40):
    """Fair Re-rank : selectionne le top-k en equilibrant les genres profs."""
    def borda_user(g):
        r = g.sort_values("rating", ascending=False).reset_index(drop=True)
        r["borda"] = range(len(r), 0, -1)
        return r[["item_id", "borda"]]
    borda = ratings.groupby("user_id", group_keys=False).apply(borda_user)
    candidates = list(borda.groupby("item_id")["borda"].sum().nlargest(pool).index)
    # Score moyen et inclusion par genre de prof
    item_avg   = ratings[ratings["item_id"].isin(candidates)].groupby("item_id")["rating"].mean()
    item_inc   = (ratings[ratings["item_id"].isin(candidates) &
                          (ratings["rating"] >= theta)]
                  .groupby("item_id")["rating"].count() /
                  ratings[ratings["item_id"].isin(candidates)]
                  .groupby("item_id")["rating"].count()).fillna(0)
    alpha_F = sum(1 for g in gender_map.values() if g=="F") / len(gender_map)
    selected = []; remaining = list(candidates)
    for _ in range(k):
        if not remaining: break
        def gain(iid):
            sel = selected + [iid]
            frac_F = sum(1 for i in sel if gender_map.get(i)=="F") / len(sel)
            avg_q  = item_avg.reindex(sel).mean()
            avg_inc = item_inc.reindex(sel).mean()
            # Bonus pour prox alpha_F, penalite pour desequilibre
            return avg_q + avg_inc - 2 * abs(frac_F - alpha_F)
        best = max(remaining, key=gain)
        selected.append(best); remaining.remove(best)
    return selected


# ─── 3. Métriques EDI (item-side : genre du professeur) ─────────────────────

def edi_metrics_item(ratings, top_k, gender_map, theta):
    """
    Formulation item-side :
      - ΔE = |mean_quality(F_profs in top-k) - mean_quality(M_profs in top-k)|
      - ILD = diversite cosinus des vecteurs de notation (cAURORA × profs)
      - inclusion_g = fraction de profs-g dans top-k avec mean_quality ≥ θ
    """
    s = set(top_k); k = len(top_k)
    rated = ratings[ratings["item_id"].isin(s)]

    # ΔE : comparaison de la qualite moyenne des profs F vs M dans le top-k
    mean_q = rated.groupby("item_id")["rating"].mean()
    q_F = mean_q[[i for i in top_k if gender_map.get(i) == "F"]].mean()
    q_M = mean_q[[i for i in top_k if gender_map.get(i) == "M"]].mean()
    dE = abs((q_F if not np.isnan(q_F) else 0) - (q_M if not np.isnan(q_M) else 0))

    # ILD : diversite cosinus des vecteurs de notation des profs dans top-k
    pivot = rated.pivot_table(index="user_id", columns="item_id",
                              values="rating", fill_value=0)
    vecs = {i: pivot[i].values if i in pivot.columns
            else np.zeros(len(pivot)) for i in top_k}
    tot, np_ = 0.0, 0
    for a, b in comb(top_k, 2):
        va, vb = vecs[a], vecs[b]
        n_ = np.linalg.norm(va) * np.linalg.norm(vb)
        tot += 1 - (np.dot(va, vb) / n_ if n_ > 0 else 0.0)
        np_ += 1
    ild = tot / np_ if np_ > 0 else 0.0

    # Inclusion : fraction de profs-g dans top-k avec mean quality ≥ θ
    hi_rated = rated[rated["rating"] >= theta].groupby("item_id")["rating"].count()
    total    = rated.groupby("item_id")["rating"].count()
    inc_rate = (hi_rated / total).fillna(0)
    def inc(g):
        ids = [i for i in top_k if gender_map.get(i) == g]
        if not ids: return 0.0
        return inc_rate.reindex(ids).fillna(0).mean()

    # Fraction F/M dans le top-k (indicateur de representation)
    frac_F = sum(1 for i in top_k if gender_map.get(i)=="F") / k
    frac_M = 1 - frac_F

    return dict(dE=round(float(dE), 4), ILD=round(float(ild), 4),
                inc_F=round(float(inc("F")), 4), inc_M=round(float(inc("M")), 4),
                frac_F=round(frac_F, 3), frac_M=round(frac_M, 3))


# ─── 4. Coarsening EDI contraint (cote item) ─────────────────────────────────

def run_coarsening_item(ratings, gender_map, k, theta, max_merges=MAX_MERGES):
    """
    Coarsening adapte pour equite cote items (professeurs) :
    - Fusionne des cAURORA-utilisateurs similaires (pas de contrainte de genre sur les users)
    - Verifie que le top-k resultant maintient l'equite de genre des profs recommandes
    """
    t0 = time.time()
    users = ratings["user_id"].unique().tolist()
    items = ratings["item_id"].unique().tolist()
    n_users = len(users)

    # Construction du pivot users × items
    print(f"    Construction pivot {len(users)} × {len(items)}...", end="", flush=True)
    pivot = ratings.pivot_table(index="user_id", columns="item_id",
                                values="rating", fill_value=0)
    print(f" OK ({pivot.values.nbytes//1024//1024}MB)")

    # Scores de Borda initiaux par super-noeud
    ur = {u: dict(zip(g["item_id"], g["rating"]))
          for u, g in ratings.groupby("user_id")}
    sn_s = {u: dict(ur.get(u, {})) for u in users}   # somme ratings
    sn_c = {u: {i: 1 for i in ur.get(u, {})} for u in users}  # comptage
    sn_m = {u: {u} for u in users}   # membres de chaque super-noeud
    u2s  = {u: u for u in users}     # mapping user → super-noeud actuel
    active = set(users)

    # Scores de Borda globaux
    cb = {}
    sb = {}
    for u in users:
        avgs = {i: sn_s[u][i]/sn_c[u][i] for i in sn_s[u]}
        ranked = sorted(avgs.items(), key=lambda x: x[1], reverse=True)
        n_ = len(ranked)
        b = {iid: n_-r for r, (iid, _) in enumerate(ranked)}
        sb[u] = b
        for iid, sc in b.items():
            cb[iid] = cb.get(iid, 0) + sc

    def gtk():
        return sorted({i: cb[i] for i in cb if cb[i] > 0},
                      key=cb.get, reverse=True)[:k]

    ref_topk = gtk()
    m_ref = edi_metrics_item(ratings, ref_topk, gender_map, theta)
    ref_dE = m_ref["dE"]; ref_ild = m_ref["ILD"]; ref_fF = m_ref["frac_F"]
    cur = list(ref_topk)

    # Similarite cosinus entre cAURORA-utilisateurs
    print(f"    Similarite cosinus entre cAURORA...", end="", flush=True)
    M2 = pivot.values.astype(np.float32)
    nu = np.linalg.norm(M2, axis=1, keepdims=True); nu[nu == 0] = 1
    sim = (M2 / nu) @ (M2 / nu).T
    ul = list(pivot.index)
    nu2 = len(ul)

    pairs = []
    for i in range(nu2):
        for j in range(i + 1, nu2):
            sv = float(sim[i, j])
            if sv > 0.1:  # seuil minimal de similarite
                pairs.append((sv, ul[i], ul[j]))
    pairs.sort(reverse=True)
    print(f" {len(pairs):,} paires similaires")

    n_accepted = 0
    for sv, ua, ub in pairs:
        if n_accepted >= max_merges:
            break
        sna = u2s.get(ua); snb = u2s.get(ub)
        if sna is None or snb is None or sna == snb:
            continue
        # Fusion de sna et snb
        ia = set(sn_s[sna].keys()); ib = set(sn_s[snb].keys())
        ms = {i: sn_s[sna].get(i, 0) + sn_s[snb].get(i, 0) for i in ia | ib}
        mc = {i: sn_c[sna].get(i, 0) + sn_c[snb].get(i, 0) for i in ia | ib}
        mavgs = {i: ms[i]/mc[i] for i in ms}
        rk = sorted(mavgs.items(), key=lambda x: x[1], reverse=True)
        n_ = len(rk); bm = {iid: n_-r for r, (iid, _) in enumerate(rk)}
        # Mise a jour scores Borda globaux
        for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) - sc
        for iid, sc in bm.items():      cb[iid] = cb.get(iid, 0) + sc
        nt = gtk()
        # Contrainte EDI cote item
        if nt == cur:
            ok = True
        else:
            m_new = edi_metrics_item(ratings, nt, gender_map, theta)
            ok = (m_new["dE"]    <= ref_dE + 0.10 and
                  m_new["ILD"]   >= ref_ild - 0.05 and
                  abs(m_new["frac_F"] - ref_fF) <= 0.10)
        if ok:
            sn_s[sna] = ms; sn_c[sna] = mc; sb[sna] = bm
            for u in sn_m[snb]: u2s[u] = sna
            sn_m[sna] |= sn_m[snb]
            del sn_s[snb], sn_c[snb], sb[snb], sn_m[snb]
            active.discard(snb)
            if nt != cur:
                cur = nt; n_accepted += 1
        else:
            for iid, sc in bm.items():      cb[iid] = cb.get(iid, 0) - sc
            for iid, sc in sb[snb].items(): cb[iid] = cb.get(iid, 0) + sc
            for iid, sc in sb[sna].items(): cb[iid] = cb.get(iid, 0) + sc

    final = gtk()
    m = edi_metrics_item(ratings, final, gender_map, theta)
    m["ratio"]         = round(len(active) / n_users, 3)
    m["n_supernodes"]  = len(active)
    m["elapsed_s"]     = round(time.time() - t0, 3)
    return m


# ─── 5. Point d'entrée ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Experiences EDI — Rate My Professors ===")
    print(f"N_CLASSES={N_CLASSES}, MIN_PROFS={MIN_PROFS}, theta={THETA}, k={K_VALUES}")

    ratings, gender_map, alpha_F, alpha_M = load_and_build()

    n_users = ratings["user_id"].nunique()
    n_items = ratings["item_id"].nunique()
    n_F_items = sum(1 for g in gender_map.values() if g == "F")
    n_M_items = len(gender_map) - n_F_items

    results = {
        "meta": {
            "dataset": "Rate My Professors",
            "source": "github.com/vxuv/ratemyprofessor-dataset",
            "formulation": "item-side equity",
            "user": "code de cAURORA (ex. ENG101)",
            "item": "professeur (genre infere depuis prenom)",
            "n_users": int(n_users),
            "n_items": int(n_items),
            "n_F_items": int(n_F_items),
            "n_M_items": int(n_M_items),
            "alpha_F": float(alpha_F),
            "alpha_M": float(alpha_M),
            "n_ratings": int(len(ratings)),
            "k_values": K_VALUES,
            "theta": THETA,
            "seed": SEED,
            "gender_inference": "gender_guesser v0.4.0 (premier prenom)"
        },
        "experiments": {}
    }

    METHODS = ["Average Score","Borda","Weighted Borda","Condorcet","Fair Re-rank","AURORA"]

    for k in K_VALUES:
        print(f"\n{'='*65}")
        print(f"k = {k}")
        exp = {"k": k, "theta": THETA}

        for method_name, fn in [
            ("Average Score",  lambda: top_k_avg(ratings, k)),
            ("Borda",          lambda: top_k_borda(ratings, k)),
            ("Weighted Borda", lambda: top_k_weighted_borda(ratings, gender_map, k)),
            ("Condorcet",      lambda: top_k_condorcet(ratings, k)),
            ("Fair Re-rank",   lambda: top_k_fair_rerank(ratings, gender_map, k, THETA)),
        ]:
            print(f"  {method_name}...", end="", flush=True)
            t0 = time.time()
            items_out = fn()
            rt = time.time() - t0
            m = edi_metrics_item(ratings, items_out, gender_map, THETA)
            m["elapsed_s"] = round(rt, 4)
            exp[method_name] = m
            print(f" {rt:.3f}s | ΔE={m['dE']:.4f} ILD={m['ILD']:.4f} F={m['frac_F']:.2f}")

        print(f"  AURORA (coarsening, max_merges={MAX_MERGES})...")
        m = run_coarsening_item(ratings, gender_map, k, THETA, MAX_MERGES)
        exp["AURORA"] = m
        print(f"  → {m['elapsed_s']}s | ΔE={m['dE']:.4f} ILD={m['ILD']:.4f} "
              f"F={m['frac_F']:.2f} ({m['n_supernodes']} super-noeuds)")

        results["experiments"][f"k{k}_t{str(THETA).replace('.','_')}"] = exp

        print(f"\n  {'Methode':<16} {'ΔE':>7} {'ILD':>7} {'inc_F':>7} {'inc_M':>7} "
              f"{'frac_F':>7} {'temps':>10}")
        for meth in METHODS:
            v = exp[meth]
            print(f"  {meth:<16} {v['dE']:>7.4f} {v['ILD']:>7.4f} {v['inc_F']:>7.4f} "
                  f"{v['inc_M']:>7.4f} {v['frac_F']:>7.3f} {v['elapsed_s']:>9.3f}s")

    out = os.path.join(os.path.dirname(__file__), "rmp_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSauvegarde : {out}")
