# -*- coding: utf-8 -*-
"""择律玩法模拟器，v2。在 v1 之上扩展：押韵口径、转韵额度、非平稳律池、玩家漏看。

原理: 配对设计时所有策略在第 i 局共用同一随机种子
实测: K=4 B=10 C=2, fine 死磕60.4%→80.9%; g18 死磕95.2%→96.4% (run_v2.py E6)
依赖: numpy
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
from rhyme_switch.algo import (OnlineV2, decision_budget, hit_prob, q_star)
from rhyme_switch.groups import NAMES, remap

_st = json.load(open(Path(__file__).resolve().parents[1] / 'data/rhyme_stats.json'))
COUNTS = _st['counts']
class World:
    """游戏世界。

    参数
    ----
    mode : str
        押韵口径，'fine' / 'g18' / 'g13'。
    N_lib : int
        韵库总句数目标值。
    cap : int
        每个韵的句数上限。
    K : int
        每次刷新的候选句数。
    law : str
        'uniform' 均匀出题，'natural' 按语料频率出题。
    depletion : bool
        刷出的诗句是否从库中移除。
    cold_frac : float or None
        非 None 时，出题以该比例偏向最冷的韵。
    """
    def __init__(self, mode='fine', N_lib=5000, cap=200, K=4, law='uniform',
                 depletion=False, cold_frac=None):
        self.mode = mode
        d, mp = remap(COUNTS, mode)
        self.rhymes = NAMES[mode] if mode == 'fine' else sorted(d.keys())
        self.rhymes = [r for r in self.rhymes if d.get(r, 0) > 0]
        self.R = len(self.rhymes)
        pi = np.array([d[r] for r in self.rhymes], float); pi /= pi.sum()
        self.pi_true = pi
        n = np.minimum(cap, np.round(pi * N_lib)).astype(int)
        self.n0, self.N, self.K = n, int(n.sum()), K
        self.depletion = depletion
        self.omega0 = pi.copy() if law == 'natural' else np.ones(self.R) / self.R
        if cold_frac:                       # 出题偏向冷门（难度开关）
            k = max(1, int(self.R * cold_frac)); cold = np.argsort(n)[:k]
            w = np.full(self.R, 0.05 / self.R); w[cold] = 0.95 / k; self.omega0 = w
        self.q = np.array([hit_prob(n[i], self.N, K) for i in range(self.R)])
        self.qbar = float(self.omega0 @ self.q)
        self.qstar = q_star(self.qbar, 2)
        o = np.argsort(-self.q); self.bucket = np.empty(self.R, dtype=object)
        k1 = max(1, self.R // 4); self.bucket[o[:k1]] = '热门'
        self.bucket[o[k1:3 * k1]] = '中等'; self.bucket[o[3 * k1:]] = '冷门'

# ---- 策略 ----
class Policy:
    name = 'policy'; can_pick = False
    def reset(self): pass
    def observe_hand(self, h): pass
    def observe_law(self, r): pass
    def new_episode(self): pass
    def choose_law(self, b, s=1): return 0
    def act(self, st): return 'refresh'

class NeverSwitch(Policy):
    name = '死磕不转韵'
    def act(self, st): return 'refresh'

class FixedK(Policy):
    def __init__(self, k): self.k, self.name = k, f'固定{k}次不出就转'
    def act(self, st): return 'switch' if st['fails'] >= self.k else 'refresh'

class RandomSwitch(Policy):
    name = '瞎转(30%)'
    def __init__(self, p=0.3, seed=0): self.p = p; self.rng = np.random.default_rng(seed)
    def act(self, st): return 'switch' if self.rng.random() < self.p else 'refresh'

class ClosedForm(Policy):
    name = '闭式阈值q*(已知库)'
    def __init__(self, w, C): self.w, self.C = w, C
    def act(self, st): return decision_closed_form(self.w.q[st['r']], self.w.qbar, self.C, True)

class Budget1(Policy):
    name = '一步判据(已知库)'
    def __init__(self, w, C): self.w, self.C = w, C
    def act(self, st): return decision_budget(self.w.q[st['r']], self.w.qbar, st['b'], self.C, True)

class DPOracle2D(Policy):
    name = 'DP上界(2D·无限额度)'
    def __init__(self, w, B, C):
        from rhyme_algo2 import dp_values
        self.w, self.C = w, C; self.V = dp_values(w.q, w.omega0, B, C, False)
    def act(self, st):
        b, r = st['b'], st['r']
        stay = self.w.q[r] + (1 - self.w.q[r]) * self.V[r, b - 1]
        sw = float(self.w.omega0 @ self.V[:, b - self.C]) if b > self.C else 0.0
        return 'switch' if sw > stay + 1e-12 else 'refresh'

class DPOracle3D(Policy):
    """动态规划上界，已知真实概率与剩余转韵额度。"""
    def __init__(self, w, B, C, Smax, pick_best=False):
        from rhyme_algo2 import dp_values3
        self.w, self.C = w, C; self.pick_best = pick_best; self.can_pick = pick_best
        self.V, self.S = dp_values3(w.q, w.omega0, B, C, Smax, pick_best)
        self.name = 'DP上界(3D·额度%d)' % Smax + ('·可挑律' if pick_best else '')
    def act(self, st):
        b, r, s = st['b'], st['r'], st['s']
        ss = min(s, self.S)
        stay = self.w.q[r] + (1 - self.w.q[r]) * self.V[r, b - 1, ss]
        if b > self.C and ss > 0:
            vc = self.V[:, b - self.C, ss - 1]
            sw = vc.max() if self.pick_best else float(self.w.omega0 @ vc)
        else:
            sw = 0.0
        return 'switch' if sw > stay + 1e-12 else 'refresh'
    def choose_law(self, b, s=1):
        return int(np.argmax(self.V[:, min(b, self.V.shape[1] - 1), min(s, self.S)]))

class WrapV2(Policy):
    """把 rhyme_algo2.OnlineV2 包装成 Policy。"""
    def __init__(self, R, label, **kw):
        from rhyme_algo2 import OnlineV2
        self.p = OnlineV2(R, **kw); self.name = label
        self.can_pick = self.p.pick_best
    def reset(self): self.p.reset()
    def new_episode(self): self.p.new_episode()
    def observe_hand(self, h): self.p.observe_hand(h)
    def observe_law(self, r): self.p.observe_law(r)
    def act(self, st):
        return self.p.act(st['r'], st['b'], st['s'],
                          self.p.eta / (1.0 + self.p.obs) if self.p.eta else 0.0)
    def choose_law(self, b, s=1): return self.p.choose_law(b, s)

# ---- 对局 ----
def play(w, pol, rng, B, C, S, p_miss=0.0):
    n = w.n0.copy(); pol.new_episode()
    r = int(rng.choice(w.R, p=w.omega0)); pol.observe_law(r)
    b, s, fails, used, sw, streak, mx = B, S, 0, 0, 0, 0, 0
    while True:
        if pol.act({'r': r, 'b': b, 's': s, 'fails': fails, 'n': n}) == 'switch' and s > 0 and b > C:
            b -= C; s -= 1; sw += 1; fails = 0
            r = int(pol.choose_law(b, s)) if pol.can_pick else int(rng.choice(w.R, p=w.omega0))
            pol.observe_law(r)
        if b <= 0: return dict(win=False, used=used, sw=sw, law=r, mx=mx)
        hand = rng.multivariate_hypergeometric(n, w.K)
        b -= 1; used += 1; pol.observe_hand(hand)
        if w.depletion: n = (n - hand).astype(np.int64)
        if hand[r] > 0:
            if p_miss > 0 and rng.random() < p_miss:      # 抽中了但没认出来
                pass
            else:
                return dict(win=True, used=used, sw=sw, law=r, mx=mx)
        fails += 1; streak += 1; mx = max(mx, streak)
        if b <= 0: return dict(win=False, used=used, sw=sw, law=r, mx=mx)

def evaluate(w, pol, rng, B, C, S, n_games, persistent=False, p_miss=0.0):
    res = []; pol.reset()
    for g in range(n_games):
        if not persistent: pol.reset()
        res.append(play(w, pol, rng, B, C, S, p_miss))
    win = np.array([x['win'] for x in res]); used = np.array([x['used'] for x in res])
    sw = np.array([x['sw'] for x in res]); mx = np.array([x['mx'] for x in res])
    bk = w.bucket[np.array([x['law'] for x in res])]
    out = dict(policy=pol.name, win_rate=float(win.mean()), avg_refresh=float(used.mean()),
               avg_switch=float(sw.mean()), pain3=float((mx >= 3).mean()),
               pain5=float((mx >= 5).mean()), win=win)
    for bn in ['热门', '中等', '冷门']:
        m = bk == bn
        out['win_' + bn] = float(win[m].mean()) if m.any() else None
    return out
