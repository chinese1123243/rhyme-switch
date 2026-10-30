# -*- coding: utf-8 -*-
"""v1 实验: E1 策略对比 / E2 冷门律 / E3 阈值扫描 / E4 参数鲁棒性 / E5 收敛。

用法: python experiments/run_v1.py
产出: data/results.json
依赖: numpy
"""

import json, numpy as np, time, os
import simulate as S
from simulate import RHYMES, R, PI0
from pathlib import Path

T0=time.time(); RES={}
BASE=dict(N_lib=5000, cap=200, K=4, law='uniform', depletion=False)
B,C,SW=10,2,99
def make_pols(w,B,C):
    return [('① 死磕不转韵（新手）',S.NeverSwitch(),False),
      ('② 固定2次失败后转韵',S.FixedK(2),False),
      ('② 固定3次失败后转韵',S.FixedK(3),False),
      ('② 固定5次失败后转韵',S.FixedK(5),False),
      ('③ 瞎转（30%概率乱转）',S.RandomSwitch(0.3),False),
      ('④ 闭式阈值 q*（已知库）',S.ClosedFormOracle(w,C),False),
      ('⑤ 预算感知一步判据（已知库）',S.BudgetOracle(w,C),False),
      ('⑥ DP最优·随机转韵（上界）',S.DPOracle(w,B,C),False),
      ('⑦ 在线估计+一步判据',S.OnlineOneStep(C=C,K=w.K),True),
      ('⑧ 在线估计+轻量DP',S.OnlineDP(C=C,K=w.K),True),
      ('⑨ 在线估计+轻量DP+推荐最优律',S.OnlineDP(C=C,K=w.K,pick_best=True),True),
      ('⑩ DP最优·可自选律（上界）',S.DPOraclePickBest(w,B,C),False)]

# ================= E1 主实验
print("=== E1 主实验 20000局/策略 ===",flush=True)
w=S.World(**BASE)
RES['meta']=dict(N=w.N,q_min=float(w.q.min()),q_max=float(w.q.max()),qbar=w.qbar,qstar=w.qstar,
                 K=w.K,B=B,C=C,N_lib=BASE['N_lib'],cap=BASE['cap'],law=BASE['law'],
                 top=[RHYMES[i] for i in np.argsort(-w.q)[:5]],bot=[RHYMES[i] for i in np.argsort(w.q)[:5]])
print(f"  库N={w.N}  q[{w.q.min():.4f},{w.q.max():.4f}]  q̄={w.qbar:.4f}  q*={w.qstar:.4f}")
rows=[]
for nm,p,pers in make_pols(w,B,C):
    r=S.evaluate(w,p,np.random.default_rng(101),B,C,SW,20000,persistent=pers); r['policy']=nm; rows.append(r)
    print(f"  {nm:<26} 通关{r['win_rate']*100:5.1f}%  刷{r['avg_refresh']:4.2f}  转{r['avg_switch']:4.2f}  3连败{r['pain3']*100:5.1f}%",flush=True)
RES['E1']=rows

# ================= E2 冷门律专项
print("=== E2 冷门律专项（出题只在最冷8韵里抽）===",flush=True)
cold=np.argsort(w.q)[:8]
w2=S.World(**BASE); w2.omega=np.zeros(R); w2.omega[cold]=1/8; w2.qbar=float(w2.omega@w2.q)
rows2=[]
for nm,p,pers in make_pols(w2,B,C):
    r=S.evaluate(w2,p,np.random.default_rng(202),B,C,SW,12000,persistent=pers); r['policy']=nm; rows2.append(r)
    print(f"  {nm:<26} 通关{r['win_rate']*100:5.1f}%  刷{r['avg_refresh']:4.2f}  转{r['avg_switch']:4.2f}",flush=True)
RES['E2']=rows2; RES['E2_cold']=[RHYMES[i] for i in cold]

# ================= E3 阈值扫描：验证闭式解
print("=== E3 阈值扫描 ===",flush=True)
grid=np.linspace(0.0,0.19,39); curve=[]
for t in grid:
    r=S.evaluate(w,S.ThrPolicy(w,t),np.random.default_rng(303),B,C,SW,6000)
    curve.append([float(t),r['win_rate'],r['avg_refresh'],r['avg_switch']])
best=max(curve,key=lambda x:x[1]); RES['E3']=curve
print(f"  经验最优阈值={best[0]:.4f}(通关{best[1]*100:.1f}%)  理论q*={w.qstar:.4f}  q̄={w.qbar:.4f}")

# ================= E4 参数鲁棒性
print("=== E4 参数鲁棒性 ===",flush=True)
scen=[]
for K in [3,4,6]: scen.append((f'候选数K={K}',dict(BASE,K=K),B,C))
for b_ in [6,10,15]: scen.append((f'预算B={b_}',dict(BASE),b_,C))
for c_ in [1,2,3]: scen.append((f'转韵代价C={c_}',dict(BASE),B,c_))
for cap in [100,200,400,10**6]: scen.append((f'库上限cap={cap}',dict(BASE,cap=cap),B,C))
for lw in ['uniform','natural']: scen.append((f'出题分布={lw}',dict(BASE,law=lw),B,C))
for dp in [False,True]: scen.append((f'库消耗={dp}',dict(BASE,depletion=dp),B,C))
rows4=[]
for nm,kw,b_,c_ in scen:
    ww=S.World(**kw); rec={'scenario':nm}
    for tnm,p in [('死磕',S.NeverSwitch()),('固定3次',S.FixedK(3)),
                  ('在线估计+DP',S.OnlineDP(C=c_,K=kw['K'])),('在线估计+DP+自选律',S.OnlineDP(C=c_,K=kw['K'],pick_best=True))]:
        rec[tnm]=S.evaluate(ww,p,np.random.default_rng(404),b_,c_,SW,6000,persistent=True)['win_rate']
    rec['上界DP']=S.evaluate(ww,S.DPOracle(ww,b_,c_),np.random.default_rng(404),b_,c_,SW,6000)['win_rate']
    rows4.append(rec)
    print(f"  {nm:<18} 死磕{rec['死磕']*100:5.1f}%  固定3{rec['固定3次']*100:5.1f}%  在线{rec['在线估计+DP']*100:5.1f}%  在线+选律{rec['在线估计+DP+自选律']*100:5.1f}%  上界{rec['上界DP']*100:5.1f}%",flush=True)
RES['E4']=rows4

# ================= E5 在线学习收敛
print("=== E5 在线学习收敛 30000局 ===",flush=True)
p=S.OnlineDP(C=C,K=4); p.reset(); rng=np.random.default_rng(505); wins=[]
for g in range(30000): wins.append(S.play(w,p,rng,B,C,SW)['win'])
wins=np.array(wins,float)
curve=[float(wins[i:i+1000].mean()) for i in range(0,30000,1000)]
p2=S.OnlineDP(C=C,K=4); r2=np.random.default_rng(505)
nl=[]
for g in range(8000):
    p2.reset(); nl.append(S.play(w,p2,r2,B,C,SW)['win'])
nl=np.array(nl,float)
RES['E5']={'curve':curve,'nolearn':float(nl.mean()),'oracle':rows[5]['win_rate']}
print("  第1千局 %.1f%% → 第30千局 %.1f%%  |  不跨局学习对照 %.1f%%"%(curve[0]*100,curve[-1]*100,nl.mean()*100))

json.dump(RES,open(Path(__file__).resolve().parents[1] / 'data' / 'results.json', 'w'),ensure_ascii=False,indent=1)
print("ALL DONE %.0fs"%(time.time()-T0),flush=True)
