# -*- coding: utf-8 -*-
"""v2 实验: E6 押韵口径 / E7 转韵额度 / E8 采样方式 / E9 非平稳 / E10 漏看。

用法: python experiments/run_v2.py
产出: data/results2.json, data/results_e9.json
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

# ===================== E6 韵部口径对比
print("=== E6 判定口径对比（36韵母 vs 十八韵归并）===", flush=True)
e6 = {}
for mode in ['fine', 'g18']:
    w = World(mode=mode)
    pols = [('死磕', NeverSwitch(), False), ('固定3次', FixedK(3), False),
            ('DP上界3D', DPOracle3D(w, B, C, 3), False),
            ('DP上界·挑律', DPOracle3D(w, B, C, 3, True), False),
            ('v2-TS', WrapV2(w.R, 'v2-TS', C=C, K=4, thompson=True), True),
            ('v2-TS+挑律+鲁棒', WrapV2(w.R, 'v2-TS+挑律+鲁棒', C=C, K=4, thompson=True, pick_best=True, robust_beta=0.3), True)]
    o = paired_eval(w, pols, 15000, seed=11)
    e6[mode] = {k: dict(wr=v['wr'], ar=v['ar'], asw=v['asw']) for k, v in o.items()}
    e6[mode]['_meta'] = dict(R=w.R, N=w.N, qmin=float(w.q.min()), qmax=float(w.q.max()),
                             spread=float(w.q.max() / w.q.min()), qbar=w.qbar, qstar=w.qstar)
    print(f"  [{mode}] R={w.R} N={w.N} 命中率极差 {w.q.max()/w.q.min():.1f}倍 q̄={w.qbar:.3f}", flush=True)
    for k, v in o.items():
        print(f"      {k:<18} {v['wr']*100:5.1f}%", flush=True)
    rows = cmp_table(o, '死磕', ['DP上界3D', 'v2-TS', 'v2-TS+挑律+鲁棒'])
    e6[mode]['_cmp'] = rows
    for r in rows:
        print(f"      vs死磕 {r['a']:<18} Δ={r['delta']*100:+5.1f}pp [{r['lo']*100:+.1f},{r['hi']*100:+.1f}] p={r['p']:.4f}", flush=True)
RES['E6'] = e6

# ===================== E7 有限转韵额度
print("=== E7 转韵额度稀缺时（3D DP 的价值）===", flush=True)
w = World(mode='fine'); e7 = []
for S_ in [1, 2, 3, 5, 99]:
    pols = [('死磕', NeverSwitch(), False), ('固定3次', FixedK(3), False),
            ('DP上界2D(无视额度)', DPOracle2D(w, B, C), False),
            ('DP上界3D(知额度)', DPOracle3D(w, B, C, S_), False),
            ('v2-TS(知额度)', WrapV2(w.R, 'v2-TS', C=C, K=4, thompson=True), True),
            ('v2-TS+挑律', WrapV2(w.R, 'v2-TS+挑律', C=C, K=4, thompson=True, pick_best=True), True)]
    o = paired_eval(w, pols, 12000, S=S_, seed=22)
    rec = dict(S=S_, **{k: v['wr'] for k, v in o.items()})
    rec['_asw'] = {k: v['asw'] for k, v in o.items()}
    e7.append(rec)
    print(f"  额度S={S_:<3} 死磕{o['死磕']['wr']*100:5.1f}%  2D上界{o['DP上界2D(无视额度)']['wr']*100:5.1f}%  "
          f"3D上界{o['DP上界3D(知额度)']['wr']*100:5.1f}%  v2-TS{o['v2-TS(知额度)']['wr']*100:5.1f}%  "
          f"v2+挑律{o['v2-TS+挑律']['wr']*100:5.1f}%", flush=True)
RES['E7'] = e7

# ===================== E8 TS vs 确定性等价（大样本）
print("=== E8 Thompson采样 vs 后验均值 ===", flush=True)
w = World(mode='fine'); e8 = []
for pb in [False, True]:
    pols = [('CE', WrapV2(w.R, 'CE', C=C, K=4, pick_best=pb), True),
            ('TS', WrapV2(w.R, 'TS', C=C, K=4, thompson=True, pick_best=pb), True)]
    o = paired_eval(w, pols, 30000, seed=33)
    m, lo, hi, pv = boot_ci(o['TS']['win'], o['CE']['win'])
    e8.append(dict(pick_best=pb, ce=o['CE']['wr'], ts=o['TS']['wr'], delta=m, lo=lo, hi=hi, p=pv))
    print(f"  可挑律={pb}: CE {o['CE']['wr']*100:.2f}%  TS {o['TS']['wr']*100:.2f}%  "
          f"Δ={m*100:+.2f}pp [{lo*100:+.2f},{hi*100:+.2f}] p={pv:.4f}", flush=True)
RES['E8'] = e8

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

# ===================== E10 玩家漏看 —— 量化工具价值
print("=== E10 玩家漏看（没有自动判韵工具时）===", flush=True)
w = World(mode='fine'); e10 = []
for pm in [0.0, 0.10, 0.20, 0.30]:
    pols = [('死磕', NeverSwitch(), False), ('固定3次', FixedK(3), False),
            ('v2-TS', WrapV2(w.R, 'v2-TS', C=C, K=4, thompson=True), True),
            ('v2-TS+挑律', WrapV2(w.R, 'v2-TS+挑律', C=C, K=4, thompson=True, pick_best=True), True)]
    o = paired_eval(w, pols, 12000, seed=55, p_miss=pm)
    rec = dict(p_miss=pm, **{k: v['wr'] for k, v in o.items()})
    e10.append(rec)
    print(f"  漏看率{pm*100:>4.0f}%: 死磕{o['死磕']['wr']*100:5.1f}%  固定3次{o['固定3次']['wr']*100:5.1f}%  "
          f"v2{o['v2-TS']['wr']*100:5.1f}%  v2+挑律{o['v2-TS+挑律']['wr']*100:5.1f}%", flush=True)
RES['E10'] = e10

json.dump(RES, open(str(Path(__file__).resolve().parents[1] / 'data' / 'results2.json'), 'w'), ensure_ascii=False, indent=1)
print("EXP2 DONE %.0fs" % (time.time() - T0), flush=True)
