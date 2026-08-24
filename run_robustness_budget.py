"""
Robustesse au budget de fusion — MovieLens 1M et Rate My Professors.
Ré-exécute AURORA à 4 budgets (30%, 42.4%, 50%, 60% de |U|) pour vérifier
que le choix de 50% n'est pas un réglage étroit sélectionné après coup.
Résultats -> robustness_budget_results.json
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

BUDGETS = [0.30, 0.424, 0.50, 0.60]
K_ML1M = 20  # matches Table "robustness" caption in samplepaper_FINAL.tex
K_RMP = 10
THETA = 4.0

def run_ml1m():
    import run_scalability as S
    ratings, users = S.load()
    gender_map = users.set_index("user_id")["gender"].to_dict()
    n_users = len(gender_map)
    print(f"\n=== MovieLens 1M ({n_users} users, k={K_ML1M}) ===")
    out = []
    for frac in BUDGETS:
        mm = round(frac * n_users)
        print(f"  budget={frac:.1%} -> max_merges={mm}...", end="", flush=True)
        m = S.run_coarsening(ratings, gender_map, K_ML1M, THETA, max_merges=mm)
        print(f" dE={m['dE']} ILD={m['ILD']} n_sn={m['n_supernodes']}")
        out.append({"budget_frac": frac, "max_merges": mm, **m})
    return out

def run_rmp():
    import run_rmp as R
    ratings, gender_map, alpha_F, alpha_M = R.load_and_build()
    n_users = ratings["user_id"].nunique()
    print(f"\n=== Rate My Professors ({n_users} users, k={K_RMP}) ===")
    out = []
    for frac in BUDGETS:
        mm = round(frac * n_users)
        print(f"  budget={frac:.1%} -> max_merges={mm}...", end="", flush=True)
        m = R.run_coarsening_item(ratings, gender_map, K_RMP, THETA, max_merges=mm)
        print(f" dE={m['dE']} ILD={m['ILD']} n_sn={m['n_supernodes']}")
        out.append({"budget_frac": frac, "max_merges": mm, **m})
    return out

if __name__ == "__main__":
    results = {
        "meta": {"k_ml1m": K_ML1M, "k_rmp": K_RMP, "theta": THETA, "budgets": BUDGETS,
                  "description": "Balayage du budget de fusion (30-60% de |U|) sur ML1M et RMP"},
        "ml1m": run_ml1m(),
        "rmp": run_rmp(),
    }
    out_name = sys.argv[1] if len(sys.argv) > 1 else "robustness_budget_results.json"
    with open(out_name, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSauvegardé : {out_name}")
