# -*- coding: utf-8 -*-
"""择律玩法模拟器，v1。

原理: 每局按 ω 抽一个律 → 策略选刷新或转韵 → 多元超几何抽 K 张 → 命中则通关
实测: K=4 B=10 C=2, 20000局/策略, 死磕60.0% / 在线估计+轻量DP 70.8% / 上界71.1% (run_v1.py E1)
依赖: numpy
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rhyme_switch.algo import (decision_budget, decision_closed_form,
                               dp_values, hit_prob, q_star)

_st = json.load(open(Path(__file__).resolve().parents[1] / 'data/rhyme_stats.json'))
RHYMES = list(_st['counts'].keys())
PI0 = np.array([_st['counts'][r] for r in RHYMES], dtype=float); PI0 /= PI0.sum()
R = len(RHYMES)

# ---- 世界 ----
class World:
    """韵库。N_lib 总句数目标值, cap 每韵句数上限, law 出题分布, depletion 刷出的句是否移出库。"""
    def __init__(self, N_lib=5000, cap=200, K=4, law='uniform', depletion=False, pi=PI0):
        n = np.minimum(cap, np.round(pi * N_lib)).astype(int)
        self.n0, self.N, self.K = n, int(n.sum()), K
        self.depletion = depletion
        self.omega = (pi / pi.sum()) if law == 'natural' else np.ones(R) / R
        self.law, self.pi_lib = law, n / n.sum()
        self.q = np.array([hit_prob(n[i], self.N, K) for i in range(R)])
        self.qbar = float(self.omega @ self.q)
        self.qstar = q_star(self.qbar, 2)
        order = np.argsort(-self.q)
        self.bucket = np.empty(R, dtype=object)
        self.bucket[order[:8]] = '热门'; self.bucket[order[8:28]] = '中等'; self.bucket[order[28:]] = '冷门'

    def refresh_q(self, n):
        N = int(n.sum())
        return np.array([hit_prob(n[i], N, self.K) for i in range(R)]) if N > 0 else np.zeros(R)

# ---- 动态规划 ----
def dp_values(q, omega, B, C, pick_best=False):
    """动态规划值表，shape (R, B+1)。"""
    q = np.asarray(q, float); R = len(q); V = np.zeros((R, B + 1))
    for b in range(1, B + 1):
        stay = q + (1 - q) * V[:, b - 1]
        sw = (V[:, b - C].max() if pick_best else float(omega @ V[:, b - C])) if b > C else 0.0
        V[:, b] = np.maximum(stay, sw)
    return V

# ---- 策略 ----
class Policy:
    name = 'policy'; can_pick = False
    def reset(self): pass
    def observe_hand(self, hand): pass
    def observe_law(self, r): pass
    def choose_law(self, b): return 0

class NeverSwitch(Policy):
    name = '死磕不转韵（新手）'
    def act(self, st): return 'refresh'

class FixedK(Policy):
    def __init__(self, k): self.k, self.name = k, f'固定{k}次失败后转韵'
    def act(self, st): return 'switch' if st['fails'] >= self.k else 'refresh'

class RandomSwitch(Policy):
    name = '瞎转（30%概率乱转）'
    def __init__(self, p=0.3, seed=0):
        self.p = p; self.rng = np.random.default_rng(seed)
    def act(self, st): return 'switch' if self.rng.random() < self.p else 'refresh'

class ThrPolicy(Policy):
    def __init__(self, w, t): self.w, self.t, self.name = w, t, '阈值扫描'
    def act(self, st): return 'switch' if self.w.q[st['r']] < self.t else 'refresh'

class ClosedFormOracle(Policy):
    name = '闭式阈值 q*（已知韵库）'
    def __init__(self, w, C): self.w, self.C = w, C
    def act(self, st): return decision_closed_form(self.w.q[st['r']], self.w.qbar, self.C, True)

class BudgetOracle(Policy):
    name = '预算感知一步判据（已知韵库）'
    def __init__(self, w, C): self.w, self.C = w, C
    def act(self, st):
        q = self.w.refresh_q(st['n']) if self.w.depletion else self.w.q
        return decision_budget(q[st['r']], self.w.qbar, st['b'], self.C, True)

class DPOracle(Policy):
    name = 'DP最优（随机转韵·上界）'
    def __init__(self, w, B, C): self.w, self.C = w, C; self.V = dp_values(w.q, w.omega, B, C, False)
    def act(self, st):
        b, r = st['b'], st['r']
        stay = self.w.q[r] + (1 - self.w.q[r]) * self.V[r, b - 1]
        sw = float(self.w.omega @ self.V[:, b - self.C]) if b > self.C else 0.0
        return 'switch' if sw > stay + 1e-12 else 'refresh'

class DPOraclePickBest(Policy):
    name = 'DP最优（可自选律·上界）'; can_pick = True
    def __init__(self, w, B, C): self.w, self.C = w, C; self.V = dp_values(w.q, w.omega, B, C, True)
    def choose_law(self, b): return int(np.argmax(self.V[:, max(b, 0)]))
    def act(self, st):
        b, r = st['b'], st['r']
        stay = self.w.q[r] + (1 - self.w.q[r]) * self.V[r, b - 1]
        sw = self.V[:, b - self.C].max() if b > self.C else 0.0
        return 'switch' if sw > stay + 1e-12 else 'refresh'

# ---- 在线估计策略 ----

class OnlineBase(Policy):
    def __init__(self, C=2, K=4, tau=2.0, lam=1.0, pick_best=False, prior=None):
        self.C, self.K, self.tau, self.lam = C, K, tau, lam
        self.pick_best = pick_best; self.can_pick = pick_best
        self.prior = None if prior is None else np.asarray(prior, float)
    def reset(self):
        self.a_pi = np.full(R, self.tau / R) if self.prior is None else self.tau * self.prior / self.prior.sum()
        self.a_om = np.full(R, 1.0 / R)
    def observe_hand(self, hand): self.a_pi = self.a_pi * self.lam + np.asarray(hand, float)
    def observe_law(self, r): self.a_om[r] += 1.0
    def _q(self):
        pi = self.a_pi / self.a_pi.sum(); return 1.0 - (1.0 - pi) ** self.K
    def _om(self): return self.a_om / self.a_om.sum()

class OnlineOneStep(OnlineBase):
    name = '在线估计+一步判据'
    def act(self, st):
        if st['s'] <= 0 or st['b'] <= self.C: return 'refresh'
        q = self._q(); qb = float(q.max()) if self.pick_best else float(self._om() @ q)
        return decision_budget(q[st['r']], qb, st['b'], self.C, True)
    def choose_law(self, b): return int(np.argmax(self._q()))

class OnlineDP(OnlineBase):
    name = '在线估计+轻量DP'
    def act(self, st):
        b, r = st['b'], st['r']
        if st['s'] <= 0 or b <= 0: return 'refresh'
        q = self._q(); om = self._om()
        V = dp_values(q, om, b, self.C, self.pick_best)
        stay = q[r] + (1 - q[r]) * V[r, b - 1]
        sw = (V[:, b - self.C].max() if self.pick_best else float(om @ V[:, b - self.C])) if b > self.C else 0.0
        return 'switch' if sw > stay + 1e-12 else 'refresh'
    def choose_law(self, b):
        q = self._q(); V = dp_values(q, self._om(), max(b, 1), self.C, self.pick_best)
        return int(np.argmax(V[:, max(b, 0)]))

# ---- 对局 ----
def play(world, pol, rng, B, C, S):
    n = world.n0.copy()
    r = int(rng.choice(R, p=world.omega)); pol.observe_law(r)
    b, s, fails, used, sw, streak, mx = B, S, 0, 0, 0, 0, 0
    while True:
        if pol.act({'r': r, 'b': b, 's': s, 'fails': fails, 'n': n}) == 'switch' and s > 0 and b > C:
            b -= C; s -= 1; sw += 1; fails = 0
            r = int(pol.choose_law(b)) if pol.can_pick else int(rng.choice(R, p=world.omega))
            pol.observe_law(r)
        if b <= 0: return dict(win=False, used=used, sw=sw, law=r, mx=mx)
        hand = rng.multivariate_hypergeometric(n, world.K)
        b -= 1; used += 1; pol.observe_hand(hand)
        if world.depletion: n = (n - hand).astype(np.int64)
        if hand[r] > 0: return dict(win=True, used=used, sw=sw, law=r, mx=mx)
        fails += 1; streak += 1; mx = max(mx, streak)
        if b <= 0: return dict(win=False, used=used, sw=sw, law=r, mx=mx)

def evaluate(world, pol, rng, B, C, S, n_games, persistent=False):
    res = []; pol.reset()
    for g in range(n_games):
        if not persistent: pol.reset()
        res.append(play(world, pol, rng, B, C, S))
    win = np.array([x['win'] for x in res]); used = np.array([x['used'] for x in res])
    sw = np.array([x['sw'] for x in res]); mx = np.array([x['mx'] for x in res])
    bk = world.bucket[np.array([x['law'] for x in res])]
    out = dict(policy=pol.name, win_rate=float(win.mean()), avg_refresh=float(used.mean()),
               avg_refresh_win=float(used[win].mean()) if win.any() else float('nan'),
               avg_switch=float(sw.mean()), wasted=float((used - win).mean()),
               pain3=float((mx >= 3).mean()), pain5=float((mx >= 5).mean()))
    for bn in ['热门', '中等', '冷门']:
        m = bk == bn
        out['win_' + bn] = float(win[m].mean()) if m.any() else None
    return out
