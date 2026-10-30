# -*- coding: utf-8 -*-
"""从 chinese-poetry 语料构建「诗句末字 → 韵母」库。

用法: python scripts/build_corpus.py   (需先下载语料到 corpus/)
产出: corpus_lines.jsonl, data/rhyme_stats.json
依赖: pypinyin, opencc-python-reimplemented
"""

import json
import os
import re
import time
import collections
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import json, os, re, time, collections
from opencc import OpenCC
from pypinyin import pinyin, Style

t2s = OpenCC('t2s')
CJK = re.compile(r'[一-鿿]')

def norm_fin(f):
    if not f or not CJK.match(f[0]) and not ('a' <= f[0] <= 'z'):
        return ''
    f = re.sub(r'\d', '', f).replace('v', 'ü')
    return f if f and all(('a' <= c <= 'z') or c == 'ü' or c == 'ê' for c in f) else ''

def main():
    files = sorted(f for f in os.listdir(ROOT/'corpus') if f.endswith('.json'))
    out = open(str(ROOT/'corpus_lines.jsonl'), 'w', encoding='utf-8')
    nline = 0; nkept = 0
    stat = collections.Counter()
    t0 = time.time()
    for fi, fn in enumerate(files):
        data = json.load(open(ROOT/'corpus'/fn, encoding='utf-8'))
        buf = []
        for poem in data:
            for line in (poem.get('paragraphs') or []):
                nline += 1
                if line:
                    buf.append(t2s.convert(line))
        CH = 4000
        for s in range(0, len(buf), CH):
            chunk = buf[s:s+CH]
            joined = '\n'.join(chunk)
            finals = pinyin(joined, style=Style.FINALS, errors=lambda x: [[c] for c in x])
            off = 0
            for line in chunk:
                seg = finals[off:off+len(line)]
                off += len(line) + 1
                pos = [i for i, c in enumerate(line) if CJK.match(c)]
                if not pos: continue
                p = pos[-1]
                if p >= len(seg): continue
                fin = norm_fin(seg[p][0])
                if not fin or len(line) < 4: continue
                nkept += 1
                stat[fin] += 1
                out.write(json.dumps({'t': line, 'r': fin}, ensure_ascii=False)+'\n')
        if fi % 8 == 0:
            print(f"[{fi}/{len(files)}] lines={nline} kept={nkept} {time.time()-t0:.0f}s", flush=True)
    out.close()
    total = sum(stat.values())
    json.dump({'total_lines': nline, 'total_kept': nkept,
               'counts': dict(stat.most_common()),
               'dist': {k: v/total for k, v in stat.most_common()}},
              open(ROOT/'data'/'rhyme_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print("DONE lines=%d kept=%d rhymes=%d %.0fs" % (nline, nkept, len(stat), time.time()-t0))
    for k, v in stat.most_common(30):
        print(f"  {k:<6} {v:>7} {v/total*100:6.2f}%")
main()
