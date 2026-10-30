# -*- coding: utf-8 -*-
"""绘图 fig7~fig10，数据来自 data/results2.json

用法: python experiments/plot_v2.py
产出: report/figs/*.png
依赖: numpy, matplotlib
"""

import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'report' / 'figs'
OUT.mkdir(parents=True, exist_ok=True)
DATA = ROOT / 'data'
for _f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
           '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        break
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Noto Serif CJK SC']
plt.rcParams['axes.unicode_minus'] = False
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import json, base64, os, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
for f in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc']:
    if os.path.exists(f): font_manager.fontManager.addfont(f); break
plt.rcParams['font.sans-serif']=['Noto Sans CJK JP','Noto Sans CJK SC','Noto Serif CJK SC']
plt.rcParams['axes.unicode_minus']=False
R2=json.load(open(DATA/'results2.json')); E9=json.load(open(DATA/'results_e9.json'))['E9']
def b64(p): return 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode()

# ---- fig7 口径对比（关键发现）
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
for j,(mode,ttl) in enumerate([('fine','口径A：36韵母精确相等'),('g18','口径B：十八韵/十三辙通押')]):
    d=R2['E6'][mode]; m=d['_meta']
    ks=['死磕','固定3次','DP上界3D','DP上界·挑律','v2-TS','v2-TS+挑律+鲁棒']
    v=[d[k]['wr']*100 for k in ks]
    c=['#9e9e9e','#bcaaa4','#90caf9','#1565c0','#43a047','#1b5e20']
    y=np.arange(len(ks)); ax[j].barh(y,v,color=c)
    for i,x in enumerate(v): ax[j].text(x+0.5,i,f"{x:.1f}%",va='center',fontsize=8.5)
    ax[j].set_yticks(y); ax[j].set_yticklabels(ks,fontsize=9); ax[j].invert_yaxis()
    ax[j].set_xlim(0,105); ax[j].set_xlabel('通关率 %')
    ax[j].set_title(f"{ttl}\n命中率极差 {m['spread']:.1f}倍（{m['qmin']*100:.1f}%~{m['qmax']*100:.1f}%）",fontsize=10.5)
    ax[j].axvline(d['死磕']['wr']*100,ls='--',c='#c62828',lw=1)
plt.tight_layout(); plt.savefig(str(OUT/'fig7_mode.png'),dpi=130); plt.close()

# ---- fig8 转韵额度
e7=R2['E7']; xs=[str(x['S']) if x['S']<90 else '∞' for x in e7]
fig,ax=plt.subplots(figsize=(9.5,4.3))
for k,c,mk in [('死磕','#9e9e9e','o'),('固定3次','#bcaaa4','s'),('DP上界2D(无视额度)','#90caf9','^'),
               ('DP上界3D(知额度)','#1565c0','D'),('v2-TS(知额度)','#2e7d32','o'),('v2-TS+挑律','#1b5e20','v')]:
    ax.plot(xs,[x[k]*100 for x in e7],marker=mk,lw=1.7,ms=5,label=k,color=c)
ax.set_xlabel('转韵额度上限 S（一局最多能转几次）'); ax.set_ylabel('通关率 %')
ax.set_title('转韵额度稀缺时：额度从1放宽到3有收益，之后边际为0',fontsize=11)
ax.legend(fontsize=8.5,ncol=3); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(str(OUT/'fig8_budget.png'),dpi=130); plt.close()

# ---- fig9 漏看
e10=R2['E10']; pm=[x['p_miss']*100 for x in e10]
fig,ax=plt.subplots(figsize=(9.5,4.3))
for k,c,mk in [('死磕','#9e9e9e','o'),('固定3次','#bcaaa4','s'),('v2-TS','#2e7d32','o'),('v2-TS+挑律','#1565c0','v')]:
    ax.plot(pm,[x[k]*100 for x in e10],marker=mk,lw=1.8,ms=6,label=k,color=c)
ax.fill_between(pm,[x['死磕']['wr']*100 if False else 0 for x in e10],1)
ax.set_xlabel('玩家漏看率 %（有押韵句却没认出来）'); ax.set_ylabel('通关率 %')
ax.set_title('漏看率越高越惨 —— 这正是「自动判韵工具」本身的价值',fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=.3)
ax.annotate('算法增益在30%漏看下仍有 +8.8pp',xy=(30,58),xytext=(14,62),
            fontsize=9,arrowprops=dict(arrowstyle='->',color='#2e7d32'))
plt.tight_layout(); plt.savefig(str(OUT/'fig9_miss.png'),dpi=130); plt.close()

# ---- fig10 非平稳
fig,ax=plt.subplots(figsize=(9.5,4.3))
lab=['λω=1.0','λω=1.0\n鲁棒','λω=0.995','λω=0.995\n鲁棒','λω=0.98','λω=0.98\n鲁棒']
ea=[x['early']*100 for x in E9]; la=[x['late']*100 for x in E9]
xx=np.arange(len(lab)); wd=0.36
ax.bar(xx-wd/2,ea,wd,label='规则变化前',color='#90caf9')
ax.bar(xx+wd/2,la,wd,label='规则变化后（出题偏向冷门）',color='#1565c0')
ax.axhline(E9[0]['dead']*100,ls='--',c='#c62828',label=f"死磕（变化后{E9[0]['dead']*100:.1f}%）")
ax.set_xticks(xx); ax.set_xticklabels(lab,fontsize=8.5); ax.set_ylabel('通关率 %')
ax.set_title('非平稳律池：在线估计 ω 自动适应，鲁棒/遗忘因子无额外收益',fontsize=11)
ax.legend(fontsize=8.5); ax.set_ylim(0,95)
plt.tight_layout(); plt.savefig(str(OUT/'fig10_nonstat.png'),dpi=130); plt.close()
print('v2 charts done')
