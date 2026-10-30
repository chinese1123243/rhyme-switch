# -*- coding: utf-8 -*-
"""绘图 fig1~fig6，数据来自 data/results.json

用法: python experiments/plot_v1.py
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
import simulate as S
from simulate import RHYMES, R, PI0

RES=json.load(open(DATA/'results.json')); meta=RES['meta']
_st=json.load(open(DATA/'rhyme_stats.json')); COUNTS=_st['counts']; TOT=_st['total_kept']
w=S.World(N_lib=meta['N_lib'],cap=meta['cap'],K=meta['K'],law=meta['law'])
CL={'热门':'#2e7d32','中等':'#f9a825','冷门':'#c62828'}
def b64(p): return 'data:image/png;base64,' + base64.b64encode(open(p, 'rb').read()).decode()

# ---------------- fig1 真实韵库分布
fig,ax=plt.subplots(1,2,figsize=(13,4.6))
order=np.argsort(-PI0)
ax[0].bar(range(R),PI0[order]*100,color='#5b8ff9')
ax[0].set_xticks(range(R)); ax[0].set_xticklabels([RHYMES[i] for i in order],rotation=90,fontsize=8)
ax[0].set_title('真实语料：25.1万句唐诗/宋诗/宋词的句末韵母分布',fontsize=11)
ax[0].set_ylabel('占比 %')
nl=np.minimum(meta['cap'],np.round(PI0*meta['N_lib'])); nl=nl/nl.sum()
ax[1].bar(range(R),nl[order]*100,color='#61a0a8')
ax[1].axhline(meta['cap']/nl.sum()/sum(np.minimum(meta['cap'],np.round(PI0*meta['N_lib'])))*100,ls='--',c='r',lw=1)
ax[1].set_xticks(range(R)); ax[1].set_xticklabels([RHYMES[i] for i in order],rotation=90,fontsize=8)
ax[1].set_title(f'游戏韵库（总规模 N={meta["N"]}，每律上限 cap={meta["cap"]} 句）',fontsize=11)
ax[1].set_ylabel('库内占比 %')
for a in ax: a.margins(x=0.01)
plt.tight_layout(); plt.savefig(str(OUT/'fig1_corpus.png'),dpi=130); plt.close()

# ---------------- fig2 主实验
rows=RES['E1']; nm=[r['policy'] for r in rows]; wr=[r['win_rate']*100 for r in rows]
col=['#9e9e9e','#bcaaa4','#bcaaa4','#bcaaa4','#e0e0e0','#90caf9','#64b5f6','#1565c0','#43a047','#2e7d32','#1b5e20','#000000']
fig,ax=plt.subplots(figsize=(11,5.2))
y=np.arange(len(nm)); ax.barh(y,wr,color=col[:len(nm)])
for i,(a,r) in enumerate(zip(wr,rows)):
    ax.text(a+0.6,i,f"{a:.1f}%  (刷{r['avg_refresh']:.2f}/转{r['avg_switch']:.2f})",va='center',fontsize=8.5)
ax.set_yticks(y); ax.set_yticklabels(nm,fontsize=9.5); ax.invert_yaxis()
ax.set_xlabel('通关率 %'); ax.set_xlim(0,100)
ax.axvline(rows[0]['win_rate']*100,ls='--',c='#c62828',lw=1.2)
ax.text(rows[0]['win_rate']*100,-0.9,' 新手基线',color='#c62828',fontsize=8)
ax.set_title(f"主实验：12种策略 × 20000局/种（K={meta['K']}候选, 预算B={meta['B']}, 转韵代价C={meta['C']}）",fontsize=11)
plt.tight_layout(); plt.savefig(str(OUT/'fig2_main.png'),dpi=130); plt.close()

# ---------------- fig3 冷门律专项
r2=RES['E2']; nm2=[r['policy'] for r in r2]
fig,ax=plt.subplots(figsize=(10.5,4.8))
y=np.arange(len(nm2)); v=[r['win_rate']*100 for r in r2]
ax.barh(y,v,color=col[:len(nm2)]); 
for i,(a,r) in enumerate(zip(v,r2)): ax.text(a+0.8,i,f"{a:.1f}%",va='center',fontsize=9)
ax.set_yticks(y); ax.set_yticklabels(nm2,fontsize=9.5); ax.invert_yaxis(); ax.set_xlabel('通关率 %'); ax.set_xlim(0,100)
ax.set_title('压力测试：出题只在最冷8个韵里抽（'+'、'.join(RES['E2_cold'])+'）',fontsize=11)
plt.tight_layout(); plt.savefig(str(OUT/'fig3_cold.png'),dpi=130); plt.close()

# ---------------- fig4 阈值扫描
cv=np.array(RES['E3'])
fig,ax=plt.subplots(figsize=(10,4.6))
ax.plot(cv[:,0],cv[:,1]*100,lw=2,color='#1565c0',label='实测通关率')
ax.axvline(meta['qstar'],ls='--',c='#e53935',label=f"闭式理论 q*={meta['qstar']:.4f}")
ax.axvline(meta['qbar'],ls=':',c='#43a047',label=f"平均律命中率 q̄={meta['qbar']:.4f}")
bi=int(np.argmax(cv[:,1])); ax.axvline(cv[bi,0],ls='-.',c='#ff8f00',label=f"实测最优阈值={cv[bi,0]:.4f}")
ax.scatter([cv[bi,0]],[cv[bi,1]*100],c='#ff8f00',zorder=5,s=45)
ax.set_xlabel('转韵阈值（当前律单次命中率低于此值就转）'); ax.set_ylabel('通关率 %')
ax.set_title('理论校验：固定阈值扫描 —— 闭式解 q* 落在最优区',fontsize=11); ax.legend(fontsize=8.5); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(str(OUT/'fig4_thr.png'),dpi=130); plt.close()

# ---------------- fig5 学习曲线
e5=RES['E5']; fine=np.array(e5['fine'])*100; coarse=np.array(e5['coarse'])*100
cu=np.concatenate([fine,coarse]); xx=np.concatenate([np.arange(1,len(fine)+1)*100,np.arange(len(coarse))*1000+3000])
fig,ax=plt.subplots(figsize=(10,4.2))
ax.plot(xx,cu,marker='o',ms=3.5,lw=1.8,c='#2e7d32',label='在线估计+DP（跨局持续学习）')
ax.set_xscale('symlog',linthresh=1000)
ax.axhline(e5['nolearn']*100,ls='--',c='#9e9e9e',label=f"不跨局学习（每局从零估）{e5['nolearn']*100:.1f}%")
ax.axhline(e5['oracle']*100,ls=':',c='#1565c0',label=f"已知真实韵库的DP上界 {e5['oracle']*100:.1f}%")
ax.set_xlabel('第 N 局（横轴对数）'); ax.set_ylabel('该窗口通关率 %'); ax.set_ylim(30,90)
ax.set_title('冷启动能力：工具不需要预置韵库，靠刷新时的观测边打边学',fontsize=11); ax.legend(fontsize=8.5); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(str(OUT/'fig5_learn.png'),dpi=130); plt.close()

# ---------------- fig6 鲁棒性
r4=RES['E4']; sc=[r['scenario'] for r in r4]
fig,ax=plt.subplots(figsize=(12.5,5))
xx=np.arange(len(sc)); wd=0.2
for i,(k,c) in enumerate([('死磕','#9e9e9e'),('固定3次','#bcaaa4'),('在线估计+DP','#2e7d32'),('在线估计+DP+自选律','#1565c0')]):
    ax.bar(xx+(i-1.5)*wd,[r[k]*100 for r in r4],wd,label=k,color=c)
ax.plot(xx,[r['上界DP']*100 for r in r4],'k*--',ms=9,label='DP理论上界')
ax.set_xticks(xx); ax.set_xticklabels(sc,rotation=28,ha='right',fontsize=8.5)
ax.set_ylabel('通关率 %'); ax.legend(fontsize=9,ncol=5,loc='upper left'); ax.grid(axis='y',alpha=.3)
ax.set_title('鲁棒性：17 组参数/规则变化下，在线估计+DP 始终贴近上界且优于两条基线',fontsize=11)
plt.tight_layout(); plt.savefig(str(OUT/'fig6_robust.png'),dpi=130); plt.close()
print("charts done")
