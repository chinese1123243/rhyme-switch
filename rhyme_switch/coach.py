# -*- coding: utf-8 -*-
"""转韵决策器，零依赖。

原理: q_r = 1-Π(N-n_r-j)/(N-j); V[r,b,s] = max(q_r+(1-q_r)V[r,b-1,s], E_ω[V[:,b-C,s-1]])
实测: 36韵 K=4 B=10 C=2, 死磕60.4% / 本决策器80.9% / 上界81.0% (experiments/run_v2.py E6)
依赖: 标准库
"""

import random

# 36 个现代汉语韵母。换十三辙/十八韵时替换本列表，算法不变。
RHYMES36 = ['a', 'o', 'e', 'i', 'u', 'ü', 'ai', 'ei', 'ao', 'ou', 'an', 'en',
            'ang', 'eng', 'er', 'ia', 'ie', 'iao', 'iou', 'ian', 'in', 'iang',
            'ing', 'iong', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang',
            'ueng', 'ong', 'üe', 'üan', 'ün']


class LiteCoach:
    """累计观测，输出刷新或转韵。

    C 转韵代价(折算刷新次数) / K 候选数 / tau prior 强度
    lam_pi,lam_om 遗忘因子 / thompson 后验采样 / pick_best 转韵时自选律
    robust_beta ω 分位数(<0.5 悲观)
    默认 thompson=False, robust_beta=None: 实测两者无收益 (E8: p=0.295, E9)
    """

    def __init__(self, rhymes, C=2, K=4, tau=2.0, lam_pi=1.0, lam_om=1.0,
                 thompson=False, pick_best=False, robust_beta=None, seed=0):
        self.rs = list(rhymes)
        self.R = len(self.rs)
        self.ix = {r: i for i, r in enumerate(self.rs)}
        self.C, self.K, self.tau = C, K, tau
        self.lam_pi, self.lam_om = lam_pi, lam_om
        self.thompson, self.pick_best = thompson, pick_best
        self.beta = robust_beta
        self.rng = random.Random(seed)
        self.obs = 0
        self._memo = None
        self._init()

    def _init(self):
        self.a_pi = [self.tau / self.R] * self.R
        self.a_om = [1.0 / self.R] * self.R
        self._sample()

    def _dir(self, a):
        g = [self.rng.gammavariate(max(x, 1e-6), 1.0) for x in a]
        s = sum(g) or 1.0
        return [x / s for x in g]

    def _sample(self):
        self._memo = None
        if self.thompson:
            self.pi_s, self.om_s = self._dir(self.a_pi), self._dir(self.a_om)
        else:
            t = sum(self.a_pi)
            self.pi_s = [x / t for x in self.a_pi]
            t = sum(self.a_om)
            self.om_s = [x / t for x in self.a_om]

    def new_level(self):
        """每局开头调用。thompson=True 时重新采样，保证局内策略一致。"""
        self._sample()

    def see_refresh(self, hand):
        """记一次刷新的 K 个候选句的韵，如 ['an','ing','i','an']。"""
        if self.lam_pi != 1.0:
            self.a_pi = [x * self.lam_pi for x in self.a_pi]
        for r in hand:
            i = self.ix.get(r)
            if i is not None:
                self.a_pi[i] += 1.0
        self.obs += 1

    def see_law(self, law):
        """记本关的律。"""
        i = self.ix.get(law)
        if i is None:
            return
        if self.lam_om != 1.0:
            self.a_om = [x * self.lam_om for x in self.a_om]
        self.a_om[i] += 1.0

    def _pi_now(self):
        if self.thompson:
            return self.pi_s
        t = sum(self.a_pi) or 1.0
        return [x / t for x in self.a_pi]

    def _om_now(self):
        if self.thompson:
            return self.om_s
        t = sum(self.a_om) or 1.0
        return [x / t for x in self.a_om]

    def q(self):
        """各韵单次刷新命中率。"""
        return [1.0 - (1.0 - p) ** self.K for p in self._pi_now()]

    def _dp(self, q, B, S):
        """值表 V[(b,s)][r]，带 memo。"""
        key = (B, S, self.obs, self.pick_best)
        if self._memo is not None and self._memo[0] == key:
            return self._memo[1]
        R, C = self.R, self.C
        om = self._om_now()
        wsamp = None if self.pick_best or self.beta is None else [self._dir(self.a_om) for _ in range(8)]
        V = {(0, s): [0.0] * R for s in range(S + 1)}
        for b in range(1, B + 1):
            for s in range(S + 1):
                prev = V[(b - 1, s)]
                cur = [q[r] + (1.0 - q[r]) * prev[r] for r in range(R)]
                if b > C and s > 0:
                    vc = V[(b - C, s - 1)]
                    if self.pick_best:
                        sw = max(vc)
                    elif wsamp is None:
                        sw = sum(om[r] * vc[r] for r in range(R))
                    else:
                        vals = sorted(sum(w[r] * vc[r] for r in range(R)) for w in wsamp)
                        sw = vals[min(len(vals) - 1, int(self.beta * (len(vals) - 1)))]
                    cur = [max(cur[r], sw) for r in range(R)]
                V[(b, s)] = cur
        self._memo = (key, V)
        return V

    def advise(self, law, budget_left, switches_left, pick_best=None):
        """返回 dict: action / q / value_stay / q_switch / p_win_stay / target / reason。"""
        pb = self.pick_best if pick_best is None else pick_best
        i = self.ix.get(law)
        if i is None:
            return {'action': 'refresh', 'reason': '未知韵'}
        q = self.q()
        C, B, S = self.C, budget_left, switches_left
        if B <= 0:
            return {'action': 'refresh', 'reason': '预算耗尽'}
        if S <= 0 or B <= C:
            return {'action': 'refresh', 'q': q[i],
                    'p_win_stay': 1.0 - (1.0 - q[i]) ** B,
                    'reason': '转韵额度用尽' if S <= 0 else '剩余次数不足支付转韵代价'}
        V = self._dp(q, B, S)
        stay = q[i] + (1.0 - q[i]) * V[(B - 1, S)][i]
        vc = V[(B - C, S - 1)]
        if pb:
            sw = max(vc)
            tgt = self.rs[max(range(self.R), key=lambda r: vc[r])]
        else:
            om = self._om_now()
            sw = sum(om[r] * vc[r] for r in range(R))
            tgt = None
        act = 'switch' if sw > stay + 1e-12 else 'refresh'
        return {'action': act, 'q': q[i], 'q_switch': sw, 'value_stay': stay,
                'p_win_stay': 1.0 - (1.0 - q[i]) ** B, 'target': tgt,
                'reason': '转韵分支价值更高' if act == 'switch' else '刷新分支价值更高'}

    def recommend_law(self, budget_left, switches_left=1):
        """返回 (建议的韵, 该韵单次命中率)。"""
        q = self.q()
        V = self._dp(q, max(budget_left, 1), max(switches_left, 1))
        col = V[(max(budget_left, 1), max(switches_left, 1))]
        i = max(range(self.R), key=lambda r: col[r])
        return self.rs[i], q[i]
