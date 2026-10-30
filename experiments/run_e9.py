# -*- coding: utf-8 -*-
"""E9 单跑: 非平稳律池下的遗忘因子与悲观分位数。

产出: data/results_e9.json
依赖: numpy
"""

import json, numpy as np, time, simulate2 as S2
from simulate2 import World, NeverSwitch, FixedK, DPOracle2D, DPOracle3D, WrapV2
from pathlib import Path

T0=time.time(); RES={}
B,C=10,2
def elapsed(): return f"[{time.time()-T0:5.0f}s]"

def paired_eval(w, pols, n_games, B=B, C=C, S=3, p_miss=0.0, seed=1,
                stop_frac=None, omega2=None, omega1=None):
    if omega1 is None: omega1 = w.omega0.copy()
    """配对设计：所有策略面对完全相同的第 i 局（同一初始律、同一抽样流）"""
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, size=n_games)
    out = {}
    for nm, p, pers in pols:
        if pers: p.reset()
        wins = np.zeros(n_games); used = np.zeros(n_games); sw = np.zeros(n_games)
        for i, sd in enumerate(seeds):
            if not pers: p.reset()
            if omega2 is not None and stop_frac:
                w.omega0 = omega2 if i >= int(n_games * stop_frac) else omega1
            d = S2.play(w, p, np.random.default_rng(sd), B, C, S, p_miss)
            wins[i] = d['win']; used[i] = d['used']; sw[i] = d['sw']
        out[nm] = dict(win=wins, used=used, sw=sw, wr=float(wins.mean()),
                       ar=float(used.mean()), asw=float(sw.mean()))
    return out

def boot_ci(a, b, n=4000, seed=7):
    """配对 bootstrap：返回 (差值, 95%CI, p值)"""
    rng = np.random.default_rng(seed); d = a - b; m = d.mean(); N = len(d)
    bs = np.array([rng.choice(d, N, replace=True).mean() for _ in range(n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return m, lo, hi, max(p, 1.0 / n)

def cmp_table(out, base, keys):
    rows = []
    for k in keys:
        m, lo, hi, pv = boot_ci(out[k]['win'], out[base]['win'])
        rows.append(dict(a=k, b=base, delta=m, lo=lo, hi=hi, p=pv))
    return rows

# ===================== E9 非平稳律池（出题规则中途变化）
print("=== E9 非平稳律池：第4000局起出题偏向冷门 ===", flush=True)
w = World(mode='fine')
om1 = w.omega0.copy()
cold = np.argsort(w.q)[:int(w.R * 0.3)]
om2 = np.full(w.R, 0.05 / w.R); om2[cold] = 0.95 / len(cold); om2 = om2 / om2.sum()
e9 = []
for lam in [1.0, 0.995, 0.98]:
    for rb in [None, 0.3]:
        pols = [('死磕', NeverSwitch(), False),
                ('固定3次', FixedK(3), False),
                ('v2', WrapV2(w.R, 'v2', C=C, K=4, thompson=True,
                  lam_om=lam, robust_beta=rb, pick_best=True), True)]
        o = paired_eval(w, pols, 12000, seed=44, stop_frac=1/3, omega2=om2, omega1=om1)
        v = o['v2']['win']; late = v[4000:].mean(); early = v[:4000].mean()
        e9.append(dict(lam=lam, robust=rb, early=float(early), late=float(late),
                       all=float(v.mean()), dead=o['死磕']['wr'], fix=o['固定3次']['wr']))
        print(f"  λω={lam:<6} 鲁棒={str(rb):<5} 变化前{early*100:5.1f}% → 变化后{late*100:5.1f}%  (死磕{o['死磕']['wr']*100:.1f}%)", flush=True)
    w.omega0 = om1.copy()
RES['E9'] = e9


json.dump({'E9': e9}, open(Path(__file__).resolve().parents[1] / 'data' / 'results_e9.json', 'w'), ensure_ascii=False, indent=1)
print('E9 fixed done %.0fs'%(time.time()-T0))
