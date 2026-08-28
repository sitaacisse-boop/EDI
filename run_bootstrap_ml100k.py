"""
Test de robustesse statistique — MovieLens 100k.
Sous-échantillonnage sans remise des utilisateurs (90% de |U|, 30
répétitions) pour estimer la moyenne et l'écart-type de ΔE et ILD pour
les 6 méthodes, à k=10, θ=4.0. Sans remise plutôt qu'avec remise : un
tirage bootstrap classique (avec remise) crée ~63% d'utilisateurs
uniques et de nombreux doublons exacts, qui biaisent artificiellement
le coarsening (les paires dupliquées, similarité=1.0, sont fusionnées
en priorité) sans refléter la vraie variabilité de population.
Résultats -> bootstrap_ml100k_results.json
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from run_all_experiments import (
    load, top_k_avg, top_k_borda, top_k_condorcet, top_k_weighted_borda,
    top_k_fair_rerank, edi_metrics, run_coarsening,
)

K = 10
THETA = 4.0
N_REPEATS = 30
FRACTION = 0.90
SEED = 42

def subsample(ratings, users_df, rng):
    """Tire une fraction des utilisateurs SANS remise (pas de doublons)."""
    orig_ids = users_df["user_id"].values
    n_sub = round(FRACTION * len(orig_ids))
    sel = rng.choice(orig_ids, size=n_sub, replace=False)
    sel_set = set(sel)
    sub_ratings = ratings[ratings["user_id"].isin(sel_set)].copy()
    gender_map = users_df[users_df["user_id"].isin(sel_set)].set_index("user_id")["gender"].to_dict()
    return sub_ratings, gender_map

if __name__ == "__main__":
    print("Chargement MovieLens 100k...")
    ratings, users_df = load()
    rng = np.random.default_rng(SEED)

    METHODS = ["Average Score", "Borda", "Weighted Borda", "Condorcet", "Fair Re-rank", "AURORA"]
    runs = {m: {"dE": [], "ILD": []} for m in METHODS}

    for b in range(N_REPEATS):
        sub_ratings, gender_map = subsample(ratings, users_df, rng)
        n_users = len(gender_map)
        max_merges = round(0.50 * n_users)
        print(f"  run {b+1}/{N_REPEATS} (n_users={n_users})...", end="", flush=True)

        avg_items  = top_k_avg(sub_ratings, K)
        borda_items = top_k_borda(sub_ratings, K)
        wb_items   = top_k_weighted_borda(sub_ratings, gender_map, K)
        cond_items = top_k_condorcet(sub_ratings, K)
        fair_items = top_k_fair_rerank(sub_ratings, gender_map, K, THETA)
        aurora_m   = run_coarsening(sub_ratings, gender_map, K, THETA, max_merges=max_merges)

        m_avg  = edi_metrics(sub_ratings, avg_items,   gender_map, THETA)
        m_bor  = edi_metrics(sub_ratings, borda_items, gender_map, THETA)
        m_wb   = edi_metrics(sub_ratings, wb_items,    gender_map, THETA)
        m_cond = edi_metrics(sub_ratings, cond_items,  gender_map, THETA)
        m_fair = edi_metrics(sub_ratings, fair_items,  gender_map, THETA)

        for name, m in [("Average Score", m_avg), ("Borda", m_bor), ("Weighted Borda", m_wb),
                         ("Condorcet", m_cond), ("Fair Re-rank", m_fair), ("AURORA", aurora_m)]:
            runs[name]["dE"].append(m["dE"])
            runs[name]["ILD"].append(m["ILD"])
        print(f" AURORA dE={aurora_m['dE']:.4f} ILD={aurora_m['ILD']:.4f}")

    summary = {}
    for m in METHODS:
        dE_arr = np.array(runs[m]["dE"]); ILD_arr = np.array(runs[m]["ILD"])
        summary[m] = {
            "dE_mean": round(float(dE_arr.mean()), 4), "dE_std": round(float(dE_arr.std(ddof=1)), 4),
            "ILD_mean": round(float(ILD_arr.mean()), 4), "ILD_std": round(float(ILD_arr.std(ddof=1)), 4),
        }
        print(f"{m:<16} dE={summary[m]['dE_mean']:.4f}±{summary[m]['dE_std']:.4f}  "
              f"ILD={summary[m]['ILD_mean']:.4f}±{summary[m]['ILD_std']:.4f}")

    n = N_REPEATS
    win_rates = {}
    for opp in ["Borda", "Weighted Borda", "Condorcet", "Fair Re-rank"]:
        w_de = sum(1 for i in range(n) if runs["AURORA"]["dE"][i] < runs[opp]["dE"][i])
        w_ild = sum(1 for i in range(n) if runs["AURORA"]["ILD"][i] > runs[opp]["ILD"][i])
        win_rates[opp] = {"dE_win": w_de, "ILD_win": w_ild, "n": n}
        print(f"AURORA vs {opp}: dE better in {w_de}/{n}, ILD better in {w_ild}/{n}")

    out = {
        "meta": {"dataset": "MovieLens 100k", "k": K, "theta": THETA,
                  "n_repeats": N_REPEATS, "fraction": FRACTION, "seed": SEED,
                  "description": "Sous-échantillonnage sans remise (90% de |U|, 30 répétitions)"},
        "summary": summary,
        "win_rates": win_rates,
        "raw": runs,
    }
    with open("bootstrap_ml100k_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSauvegardé : bootstrap_ml100k_results.json")
