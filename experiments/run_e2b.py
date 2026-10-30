# -*- coding: utf-8 -*-
"""E2b: 律池分布被误判时一步式判据的表现。

产出: 写入 data/results.json 的 E2b 字段
依赖: numpy
"""

import json,numpy as np,simulate as S
from simulate import R
from rhyme_algo import decision_budget
from pathlib import Path
BASE=dict(N_lib=5000,cap=200,K=4,law='uniform')
w=S.World(**BASE); cold=np.argsort(w.q)[:8]
w2=S.World(**BASE); w2.omega=np.zeros(R); w2.omega[cold]=1/8
class NaiveWrongOmega(S.Policy):
    name='一步判据·误判律池'
    def __init__(self,w,C): self.w,self.C=w,C
    def act(self,st): return decision_budget(self.w.q[st['r']], self.w.qbar, st['b'], self.C, True)
class CorrectOmega(S.Policy):
    name='一步判据·正确律池'
    def __init__(self,w,C): self.w,self.C=w,C
    def act(self,st): return decision_budget(self.w.q[st['r']], float(self.w.omega@self.w.q), st['b'], self.C, True)
out={}
for p,pers in [(NaiveWrongOmega(w,2),False),(CorrectOmega(w2,2),False),(S.DPOracle(w2,10,2),False),
               (S.OnlineDP(C=2,K=4),True),(S.NeverSwitch(),False)]:
    r=S.evaluate(w2,p,np.random.default_rng(7),10,2,99,12000,persistent=pers)
    out[p.name]=r; print(f"  {p.name:<24} 通关{r['win_rate']*100:5.1f}%  转韵{r['avg_switch']:4.2f}")
res=json.load(open(str(Path(__file__).resolve().parents[1] / 'data' / 'results.json'))); res['E2b']=out
json.dump(res,open(Path(__file__).resolve().parents[1] / 'data' / 'results.json', 'w'),ensure_ascii=False,indent=1)
