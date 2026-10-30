# -*- coding: utf-8 -*-
"""押韵判定口径。换口径即换 π 的维度，冷热门差距随之变化。

实测: 25.1万句语料, fine 36韵极差94.4倍; g18 13韵极差7.5倍
      K=4 B=10 C=2, fine 死磕60.4%→80.9%; g18 死磕95.2%→96.4% (run_v2.py E6)
依赖: 标准库
"""

FINE = ['a', 'o', 'e', 'i', 'u', 'ü', 'ai', 'ei', 'ao', 'ou', 'an', 'en',
        'ang', 'eng', 'er', 'ia', 'ie', 'iao', 'iou', 'ian', 'in', 'iang',
        'ing', 'iong', 'ua', 'uo', 'uai', 'uei', 'uan', 'uen', 'uang',
        'ueng', 'ong', 'üe', 'üan', 'ün']

GROUP18 = {'麻': ['a', 'ia', 'ua'], '波': ['o', 'uo', 'e'], '皆': ['ie', 'üe'],
           '开': ['ai', 'uai'], '微': ['ei', 'uei', 'ui'], '豪': ['ao', 'iao'],
           '侯': ['ou', 'iou', 'iu'], '寒': ['an', 'ian', 'uan', 'üan'],
           '文': ['en', 'in', 'uen', 'un', 'ün'], '唐': ['ang', 'iang', 'uang'],
           '庚': ['eng', 'ing', 'ueng', 'ong', 'iong'], '齐': ['i', 'ü', 'er'],
           '支': [], '姑': ['u']}

GROUP13 = {'发花': ['a', 'ia', 'ua'], '梭波': ['o', 'uo', 'e'], '乜斜': ['ie', 'üe'],
           '怀来': ['ai', 'uai'], '灰堆': ['ei', 'uei', 'ui'], '遥条': ['ao', 'iao'],
           '由求': ['ou', 'iou', 'iu'], '言前': ['an', 'ian', 'uan', 'üan'],
           '人辰': ['en', 'in', 'uen', 'un', 'ün'], '江阳': ['ang', 'iang', 'uang'],
           '中东': ['eng', 'ing', 'ueng', 'ong', 'iong'], '一七': ['i', 'ü', 'er'],
           '姑苏': ['u']}

NAMES = {'fine': FINE, 'g18': sorted(GROUP18), 'g13': sorted(GROUP13)}


def build(group):
    """韵母到韵部的映射。"""
    return {f: g for g, fs in group.items() for f in fs}


MAP18 = build(GROUP18)
MAP13 = build(GROUP13)


def remap(counts, mode):
    """按口径归并 36 韵母计数，返回 (计数 dict, 韵母→韵部 函数)。"""
    if mode == 'fine':
        return dict(counts), (lambda r: r)
    mp = MAP18 if mode == 'g18' else MAP13
    out = {}
    for f, c in counts.items():
        g = mp.get(f)
        if g is not None:
            out[g] = out.get(g, 0) + c
    return out, (lambda r: mp.get(r))
