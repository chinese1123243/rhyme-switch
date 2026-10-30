# -*- coding: utf-8 -*-
"""打印在线估计值与决策示例。

用法: python scripts/demo.py
依赖: numpy
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
sys.path.insert(0, str(ROOT))

import json, numpy as np, random, simulate as S
from rhyme_algo import RhymeCoach, dp_values
w=S.World(); rng=np.random.default_rng(9)
coach=RhymeCoach(S.RHYMES)
# 用真实韵库喂 300 次刷新（相当于玩家打了 ~60 局）
for _ in range(300):
    hand=rng.multivariate_hypergeometric(w.n0,4)
    coach.observe_refresh(hand)
    coach.observe_law(int(rng.choice(S.R, p=w.omega)))
print("== 在线估计 vs 真实命中率（前8热 / 后8冷）==")
qa=coach.q_all2(4)
ordq=np.argsort(-w.q)
for i in list(ordq[:8])+list(ordq[-8:]):
    print(f"  {S.RHYMES[i]:<5} 真实 {w.q[i]*100:5.2f}%   在线估计 {qa[i]*100:5.2f}%")
print("\n== 决策演示（预算B=10, 转韵代价C=2）==")
for law in ['i','ang','e','ueng','o']:
    a=coach.advise(law,budget_left=10,switches_left=3,C=2,K=4,pick_best=True)
    print(f"  律[{law:<4}] → {a['action']:<8} q={a['q_current']*100:5.2f}%  q*={a['q_star']*100:5.2f}%  "
          f"留={a['p_win_stay']*100:4.1f}%  转={a['p_win_switch']*100:4.1f}%  目标律={a['target_law']}")
print("\n== 新手提示文案示例 ==")
for law in ['ueng','ang']:
    for bl in [10,4]:
        print("  剩余%d次刷新:"%bl, coach.advise(law,budget_left=bl,switches_left=2,C=2,K=4,pick_best=True)['tip'])
# 真实诗句示例
import collections
pool=collections.defaultdict(list)
for ln in open(ROOT/'corpus_lines.jsonl',encoding='utf-8'):
    d=json.loads(ln)
    if len(pool[d['r']])<3: pool[d['r']].append(d['t'])
print("\n== 真实诗句示例 ==")
for r_ in ['i','ang','e','ueng']:
    print(f"  [{r_}] "+"  /  ".join(pool[r_]))
json.dump({'examples':{r_:pool[r_] for r_ in ['i','ang','e','ueng','o']}},open(ROOT/'demo_examples.json','w'),ensure_ascii=False,indent=1)
