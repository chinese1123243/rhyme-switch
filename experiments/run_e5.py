# -*- coding: utf-8 -*-
"""E5: 在线估计的收敛窗口。对照组每局开头重置后验。

产出: 写入 data/results.json 的 E5 字段
依赖: numpy
"""

import json,numpy as np,simulate as S
from pathlib import Path
T0=__import__('time').time()
w=S.World(); rng=np.random.default_rng(505)
p=S.OnlineDP(C=2,K=4); p.reset(); wins=[]
for g in range(24000): wins.append(S.play(w,p,rng,10,2,99)['win'])
wins=np.array(wins,float)
fine=[float(wins[i:i+100].mean()) for i in range(0,3000,100)]
coarse=[float(wins[i:i+1000].mean()) for i in range(3000,24000,1000)]
p2=S.OnlineDP(C=2,K=4); r2=np.random.default_rng(505); nl=[]
for g in range(8000):
    p2.reset(); nl.append(S.play(w,p2,r2,10,2,99)['win'])
r=json.load(open(str(Path(__file__).resolve().parents[1] / 'data' / 'results.json')))
r['E5']={'fine':fine,'coarse':coarse,'nolearn':float(np.mean(nl)),'oracle':r['E1'][5]['win_rate']}
json.dump(r,open(Path(__file__).resolve().parents[1] / 'data' / 'results.json', 'w'),ensure_ascii=False,indent=1)
print('前100局 %.1f%% → 300局 %.1f%% → 3000局 %.1f%% → 24000局 %.1f%% | 不学习 %.1f%% | 上界 %.1f%%'%(
 fine[0]*100,fine[2]*100,coarse[0]*100,coarse[-1]*100,np.mean(nl)*100,r['E1'][5]['win_rate']*100))
print('%.0fs'%(__import__('time').time()-T0))
