# -*- coding: utf-8 -*-
"""转韵决策，numpy 版。coach 是同一算法的零依赖实现。

原理: q_r = 1-Π(N-n_r-j)/(N-j); V[r,b]=max(q_r+(1-q_r)V[r,b-1], E_ω[V[:,b-C]]), 三维增 s
实测: 36韵 K=4 B=10 C=2, 死磕60.4% / 在线估计+DP 70.9% / 上界71.0% (experiments/run_v2.py)
依赖: numpy
"""

import numpy as np


def hit_prob(n_r, N, K):
    """一次刷新至少命中一个韵 r 的概率。"""
    if N <= 0 or K <= 0:
        return 0.0
    K, n_r = min(K, N), min(n_r, N)
    if N - n_r < K:
        return 1.0
    p = 1.0
    for j in range(K):
        p *= (N - n_r - j) / (N - j)
    return 1.0 - p


def q_star(q_bar, C):
    """转韵闭式阈值，q_r < q_star 时转韵。q_bar 为按 ω 加权的平均命中率。"""
    return 0.0 if q_bar <= 0 else 1.0 / (C + 1.0 / q_bar)


def decision_closed_form(q_cur, q_bar, C, can_switch=True):
    """闭式阈值判据，q_cur < q_star(q_bar, C) 时转韵。"""
    return 'switch' if can_switch and q_cur < q_star(q_bar, C) else 'refresh'


def decision_budget(q_cur, q_bar, b, C, can_switch=True):
    """预算判据，比较剩余 b 次刷新内的通关概率。"""
    if not can_switch or b <= C:
        return 'refresh'
    return 'switch' if 1 - (1 - q_bar) ** (b - C) > 1 - (1 - q_cur) ** b + 1e-12 else 'refresh'


def dirichlet_sample(alpha, rng, size=None):
    """Dirichlet 采样。size 为 None 时返回一维。"""
    a = np.maximum(np.asarray(alpha, dtype=float), 1e-6)
    g = rng.gamma(a, 1.0, size=(a.shape[0],) if size is None else (size, a.shape[0]))
    s = g.sum(axis=-1, keepdims=True)
    return g / np.where(s <= 0, 1.0, s)


def dp_values(q, omega, B, C, pick_best=False):
    """二维值表，shape (R, B+1)。"""
    q = np.asarray(q, dtype=float)
    V = np.zeros((len(q), B + 1))
    for b in range(1, B + 1):
        stay = q + (1 - q) * V[:, b - 1]
        sw = (V[:, b - C].max() if pick_best else float(omega @ V[:, b - C])) if b > C else 0.0
        V[:, b] = np.maximum(stay, sw)
    return V


def dp_values3(q, omega, B, C, S, pick_best=False):
    """三维值表，shape (R, B+1, S+1)，s 为剩余转韵额度。返回 (V, S_eff)。"""
    q = np.asarray(q, dtype=float)
    S = min(S, B // max(C, 1) + 1)
    V = np.zeros((len(q), B + 1, S + 1))
    for b in range(1, B + 1):
        stay = q[:, None] + (1 - q)[:, None] * V[:, b - 1, :]
        V[:, b, :] = stay
        if b > C:
            for s in range(1, S + 1):
                sw = V[:, b - C, s - 1].max() if pick_best else float(omega @ V[:, b - C, s - 1])
                V[:, b, s] = np.maximum(stay[:, s], sw)
    return V, S


class OnlineV2:
    """在线转韵控制器。参数同 rhyme_switch.coach.LiteCoach，R 为韵数。"""

    name = 'OnlineV2'

    def __init__(self, R, C=2, K=4, tau=2.0, lam_pi=1.0, lam_om=1.0,
                 thompson=False, pick_best=False, robust_beta=None,
                 eta=0.0, rng=None, n_om_samples=8):
        self.R, self.C, self.K = R, C, K
        self.tau, self.lam_pi, self.lam_om = tau, lam_pi, lam_om
        self.thompson, self.pick_best = thompson, pick_best
        self.robust_beta, self.eta, self.M = robust_beta, eta, n_om_samples
        self.rng = rng or np.random.default_rng(0)
        self.can_pick = pick_best
        self.obs = 0
        self.reset()

    def reset(self):
        self.a_pi = np.full(self.R, self.tau / self.R)
        self.a_om = np.full(self.R, 1.0 / self.R)
        self.new_episode()

    def new_episode(self):
        """每局开头调用。"""
        if self.thompson:
            self.pi_s = dirichlet_sample(self.a_pi, self.rng)
            self.om_s = dirichlet_sample(self.a_om, self.rng)
        else:
            self.pi_s = self.a_pi / self.a_pi.sum()
            self.om_s = self.a_om / self.a_om.sum()

    def observe_hand(self, hand):
        """记一次刷新的观测，hand 为长度 R 的计数向量。"""
        self.a_pi = self.a_pi * self.lam_pi + np.asarray(hand, dtype=float)
        self.obs += 1

    def observe_law(self, r):
        """记本关的律。"""
        self.a_om = self.a_om * self.lam_om
        self.a_om[int(r)] += 1.0

    @property
    def pi_now(self):
        return self.pi_s if self.thompson else self.a_pi / self.a_pi.sum()

    @property
    def om_now(self):
        return self.om_s if self.thompson else self.a_om / self.a_om.sum()

    def q_hat(self):
        """各韵单次刷新命中率。"""
        return 1.0 - (1.0 - self.pi_now) ** self.K

    def _sw_value(self, Vcol):
        """转韵分支价值。pick_best 取 max，否则取 ω 期望或分位数。"""
        if self.pick_best:
            return float(Vcol.max())
        if self.robust_beta is None:
            return float(self.om_now @ Vcol)
        vals = dirichlet_sample(self.a_om, self.rng, size=self.M) @ Vcol
        return float(np.quantile(vals, self.robust_beta))

    def act(self, r, b, s, refresh_bonus=0.0):
        """返回 'refresh' 或 'switch'。b 剩余刷新次数，s 剩余转韵次数。"""
        if b <= 0:
            return 'refresh'
        q = self.q_hat()
        if s <= 0 or b <= self.C:
            return 'refresh'
        V, S = dp_values3(q, self.om_now, b, self.C, s, self.pick_best)
        ss = min(s, S)
        stay = q[r] + (1 - q[r]) * V[r, b - 1, ss] + refresh_bonus
        return 'switch' if self._sw_value(V[:, b - self.C, ss - 1]) > stay + 1e-12 else 'refresh'

    def choose_law(self, b, s=1):
        """转韵时推荐的下标。"""
        q = self.q_hat()
        if b <= 0:
            return int(np.argmax(q))
        V, S = dp_values3(q, self.om_now, b, self.C, max(s, 1), self.pick_best)
        return int(np.argmax(V[:, min(b, V.shape[1] - 1), min(s, S)]))
