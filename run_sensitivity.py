"""
Analyse de sensibilité des paramètres epsilon de la coarsenisation EDI.
Fixe k=20, theta=4.0 et fait varier eps_E et eps_D pour montrer
l'impact sur ΔE, ILD et le taux de compression.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import pandas as pd
from itertools import combinations as comb

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "ml-100k")

def load():
    ratings = pd.read_csv(os.path.join(DATA_DIR,"u.data"), sep="\t", header=None,
                          names=["user_id","item_id","rating","timestamp"])
    users   = pd.read_csv(os.path.join(DATA_DIR,"u.user"), sep="|", header=None,
                          names=["user_id","age","gender","occupation","zip"])
    return ratings, users

def edi_metrics(ratings, top_k, gender_map, theta):
    s = set(top_k); k = len(top_k)
    f_u = [u for u,g in gender_map.items() if g=="F"]
    m_u = [u for u,g in gender_map.items() if g=="M"]
    def util(gu):
        gr = ratings[ratings["user_id"].isin(gu)&ratings["item_id"].isin(s)]
        return gr["rating"].sum()/(len(gu)*k) if gu and k else 0.0
    dE = abs(util(f_u)-util(m_u))
    pivot = ratings.pivot(index="user_id",columns="item_id",values="rating").fillna(0)
    vecs = {iid: pivot[iid].values if iid in pivot.columns else np.zeros(len(pivot))
            for iid in top_k}
    tot,np_ = 0.0,0
    for a,b in comb(top_k,2):
        va,vb=vecs[a],vecs[b]; n=np.linalg.norm(va)*np.linalg.norm(vb)
        tot+=1-(np.dot(va,vb)/n if n>0 else 0.0); np_+=1
    ild = tot/np_ if np_>0 else 0.0
    def inc(gu):
        gr = ratings[ratings["user_id"].isin(gu)&ratings["item_id"].isin(s)&(ratings["rating"]>=theta)]
        return gr.groupby("user_id")["item_id"].nunique().reindex(gu,fill_value=0).sum()/(len(gu)*k)
    return dict(dE=round(dE,4),ILD=round(ild,4),inc_F=round(inc(f_u),4),inc_M=round(inc(m_u),4))

def run_coarsening_eps(ratings, gender_map, k, theta, eps_E, eps_D, eps_inc, max_merges=472):
    from edi_coarsening import precompute_edi, fast_delta_E, fast_ILD, fast_inc_F
    cache = precompute_edi(ratings, gender_map, theta)
    users_list = list(gender_map.keys())
    ur = {u: dict(zip(g["item_id"],g["rating"])) for u,g in ratings.groupby("user_id")}
    sn_s={u:dict(ur.get(u,{})) for u in users_list}
    sn_c={u:{i:1 for i in ur.get(u,{})} for u in users_list}
    sn_m={u:{u} for u in users_list}; u2s={u:u for u in users_list}; active=set(users_list)
    sb={}; cb={}
    for u in users_list:
        avgs={i:sn_s[u][i]/sn_c[u][i] for i in sn_s[u]}
        ranked=sorted(avgs.items(),key=lambda x:x[1],reverse=True); n=len(ranked)
        b={iid:n-r for r,(iid,_) in enumerate(ranked)}; sb[u]=b
        for iid,sc in b.items(): cb[iid]=cb.get(iid,0)+sc
    def gtk():
        scores={i:cb[i] for i in cb if cb[i]>0}
        return sorted(scores,key=scores.get,reverse=True)[:k]
    ref=gtk(); rdE=fast_delta_E(ref,cache); rILD=fast_ILD(ref,cache); riF=fast_inc_F(ref,cache)
    cur=list(ref)
    M=cache["pivot"].values.astype(np.float32); nu=np.linalg.norm(M,axis=1,keepdims=True); nu[nu==0]=1
    sim=(M/nu)@(M/nu).T; ul=list(cache["pivot"].index); nu2=len(ul); pairs=[]
    for i in range(nu2):
        for j in range(i+1,nu2):
            sv=float(sim[i,j])
            if sv>0: pairs.append((1 if gender_map[ul[i]]==gender_map[ul[j]] else 0,sv,ul[i],ul[j]))
    pairs.sort(reverse=True); na=0
    for _,sv,ua,ub in pairs:
        if na>=max_merges: break
        sna=u2s.get(ua); snb=u2s.get(ub)
        if sna is None or snb is None or sna==snb: continue
        ia=set(sn_s[sna].keys()); ib=set(sn_s[snb].keys())
        ms={i:sn_s[sna].get(i,0)+sn_s[snb].get(i,0) for i in ia|ib}
        mc={i:sn_c[sna].get(i,0)+sn_c[snb].get(i,0) for i in ia|ib}
        mavgs={i:ms[i]/mc[i] for i in ms}
        rk=sorted(mavgs.items(),key=lambda x:x[1],reverse=True); nm=len(rk)
        bm={iid:nm-r for r,(iid,_) in enumerate(rk)}
        for iid,sc in sb[sna].items(): cb[iid]=cb.get(iid,0)-sc
        for iid,sc in sb[snb].items(): cb[iid]=cb.get(iid,0)-sc
        for iid,sc in bm.items(): cb[iid]=cb.get(iid,0)+sc
        nt=gtk()
        if nt==cur: ok=True
        else: ok=(fast_delta_E(nt,cache)<=rdE+eps_E and
                  fast_ILD(nt,cache)>=rILD-eps_D and
                  fast_inc_F(nt,cache)>=riF-eps_inc)
        if ok:
            sn_s[sna]=ms; sn_c[sna]=mc; sb[sna]=bm
            for u in sn_m[snb]: u2s[u]=sna
            sn_m[sna]|=sn_m[snb]
            del sn_s[snb],sn_c[snb],sb[snb],sn_m[snb]; active.discard(snb)
            if nt!=cur: cur=nt
            na+=1
        else:
            for iid,sc in bm.items(): cb[iid]=cb.get(iid,0)-sc
            for iid,sc in sb[snb].items(): cb[iid]=cb.get(iid,0)+sc
            for iid,sc in sb[sna].items(): cb[iid]=cb.get(iid,0)+sc
    final=gtk()
    m=edi_metrics(ratings,final,gender_map,theta)
    m["ratio"]=round(len(active)/len(users_list),3)
    m["n_supernodes"]=len(active)
    return m

# ── Main ──────────────────────────────────────────────────────────────────────
print("Chargement MovieLens 100k...")
ratings, users = load()
gender_map = users.set_index("user_id")["gender"].to_dict()

k = 20
theta = 4.0

# Varie eps_E de 0.05 à 0.30 (eps_D et eps_inc fixes)
eps_E_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
eps_D_fixed  = 0.05
eps_inc_fixed= 0.05

# Varie eps_D de 0.01 à 0.12 (eps_E et eps_inc fixes)
eps_D_values = [0.01, 0.03, 0.05, 0.07, 0.10, 0.12]
eps_E_fixed  = 0.10

results = {"vary_eps_E": [], "vary_eps_D": []}

print("\n=== Variation de eps_E (eps_D=0.05 fixe) ===")
for eps_E in eps_E_values:
    print(f"  eps_E={eps_E:.2f}...", flush=True)
    m = run_coarsening_eps(ratings, gender_map, k, theta, eps_E, eps_D_fixed, eps_inc_fixed)
    print(f"    -> dE={m['dE']} ILD={m['ILD']} n_sn={m['n_supernodes']} ratio={m['ratio']}")
    results["vary_eps_E"].append({"eps_E": eps_E, **m})

print("\n=== Variation de eps_D (eps_E=0.10 fixe) ===")
for eps_D in eps_D_values:
    print(f"  eps_D={eps_D:.2f}...", flush=True)
    m = run_coarsening_eps(ratings, gender_map, k, theta, eps_E_fixed, eps_D, eps_inc_fixed)
    print(f"    -> dE={m['dE']} ILD={m['ILD']} n_sn={m['n_supernodes']} ratio={m['ratio']}")
    results["vary_eps_D"].append({"eps_D": eps_D, **m})

print("\n=== Variation de max_merges (eps fixes) ===")
results["vary_max_merges"] = []
for mm in [25, 50, 100, 200, 300, 400, 500]:
    print(f"  max_merges={mm}...", flush=True)
    m = run_coarsening_eps(ratings, gender_map, k, theta, 0.10, 0.05, 0.05, max_merges=mm)
    print(f"    -> dE={m['dE']} ILD={m['ILD']} n_sn={m['n_supernodes']} ratio={m['ratio']}")
    results["vary_max_merges"].append({"max_merges": mm, **m})

with open("sensitivity_results.json","w") as f:
    json.dump(results, f, indent=2)
print("\nSauvegarde : sensitivity_results.json")
