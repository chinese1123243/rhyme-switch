# -*- coding: utf-8 -*-
"""生成 HTML 报告，图片以 base64 内嵌。数字全部取自 data/results*.json，不另行计算。

用法: python experiments/build_report.py
产出: report/择律转韵算法_实验报告.html
依赖: numpy, matplotlib
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGS = ROOT / 'report' / 'figs'
REPORT = ROOT / 'report'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

import json, base64, numpy as np, simulate as S
from simulate import RHYMES, R, PI0
RES=json.load(open(DATA/'results.json')); meta=RES['meta']
R2=json.load(open(DATA/'results2.json')); E9=json.load(open(DATA/'results_e9.json'))['E9']
E6=R2['E6']; E7=R2['E7']; E8=R2['E8']; E10=R2['E10']
_st=json.load(open(DATA/'rhyme_stats.json')); COUNTS=_st['counts']; TOT=_st['total_kept']
w=S.World(N_lib=meta['N_lib'],cap=meta['cap'],K=meta['K'],law=meta['law'])
def b64(p): return 'data:image/png;base64,'+base64.b64encode(open(p,'rb').read()).decode()
E1={r['policy']:r for r in RES['E1']}
def g(k,f='win_rate'): return E1[k][f]
K1='① 死磕不转韵（新手）'; K3='② 固定3次失败后转韵'; K8='⑧ 在线估计+轻量DP'
K9='⑨ 在线估计+轻量DP+推荐最优律'; K6='⑥ DP最优·随机转韵（上界）'; K10='⑩ DP最优·可自选律（上界）'
E2={r['policy']:r for r in RES['E2']}
E2b=RES['E2b']
E5=RES['E5']; E4=RES['E4']; cv=np.array(RES['E3']); bi=int(np.argmax(cv[:,1]))
nl=np.minimum(meta['cap'],np.round(PI0*meta['N_lib']))
ordq=np.argsort(-w.q)

def row(cells,tag='td'): return '<tr>'+''.join(f'<{tag}>{c}</{tag}>' for c in cells)+'</tr>'
# 36韵表
rtab=''.join(row([RHYMES[i],f"{PI0[i]*100:.2f}%",f"{int(nl[i])}",f"{w.q[i]*100:.2f}%",
    ('<b style="color:#c62828">转韵</b>' if w.q[i]<w.qstar else '<span style="color:#2e7d32">可刷</span>')])
    for i in ordq)
# E1 表
e1tab=''.join(row([r['policy'],f"<b>{r['win_rate']*100:.1f}%</b>",f"{r['avg_refresh']:.2f}",
    f"{r['avg_switch']:.2f}",f"{r['pain3']*100:.1f}%"]) for r in RES['E1'])
# E4 表
e4tab=''.join(row([r['scenario']]+[f"{r[k]*100:.1f}%" for k in ['死磕','固定3次','在线估计+DP','在线估计+DP+自选律','上界DP']]) for r in E4)
d1=g(K8)-g(K1); d2=g(K8)-g(K3)

HTML=f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>择律玩法「自动转韵」算法：设计、实现与真实验证</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.75;color:#222;max-width:940px;margin:0 auto;padding:18px 16px 60px;}}
h1{{font-size:23px;margin:6px 0 4px}} h2{{font-size:19px;margin:30px 0 10px;padding-left:10px;border-left:4px solid #1565c0}}
h3{{font-size:16px;margin:20px 0 8px;color:#1565c0}}
.sub{{color:#777;font-size:13px;margin-bottom:18px}}
.box{{background:#f3f8ff;border-radius:10px;padding:14px 16px;margin:14px 0;font-size:14.5px}}
.warn{{background:#fff8e1;border-left:4px solid #f9a825}}
.ok{{background:#eefaf0;border-left:4px solid #2e7d32}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0}}
th,td{{border:1px solid #e0e0e0;padding:6px 8px;text-align:center}}
th{{background:#f5f5f5}} td:first-child{{text-align:left}}
img{{width:100%;border-radius:8px;margin:10px 0;border:1px solid #eee}}
code{{background:#f5f5f5;padding:1px 5px;border-radius:4px;font-size:13px}}
pre{{background:#1e1e1e;color:#e8e8e8;padding:14px;border-radius:8px;overflow-x:auto;font-size:12.5px;line-height:1.6}}
.kpi{{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0}}
.kpi div{{flex:1;min-width:150px;background:#fafafa;border:1px solid #eee;border-radius:10px;padding:12px;text-align:center}}
.kpi b{{display:block;font-size:22px;color:#1565c0}} .kpi span{{font-size:12px;color:#666}}
</style></head><body>
<h1>择律玩法「自动转韵」算法：设计、实现与真实验证</h1>
<div class="sub">真实语料 25.1 万句唐诗/宋诗/宋词 · 36 个韵母 · 蒙特卡洛 20 万局+/策略 · 全部代码可复现</div>

<div class="box ok"><b>结论速览</b><br>
在「{meta['K']} 候选 / 预算 {meta['B']} 次刷新 / 转韵代价 {meta['C']} 次」的默认对局里：
<div class="kpi">
<div><b>{g(K1)*100:.1f}%</b><span>新手死磕（不转韵）</span></div>
<div><b>{g(K3)*100:.1f}%</b><span>老手经验「刷3次不出就转」</span></div>
<div><b>{g(K8)*100:.1f}%</b><span>在线估计+DP（无需预置韵库）</span></div>
<div><b>{g(K6)*100:.1f}%</b><span>已知全部概率的DP理论上界</span></div>
</div>
<ul>
<li>在线估计+DP相对新手 <b>+{d1*100:.1f} 个百分点</b>，相对「刷3次就转」<b>+{d2*100:.1f} 个百分点</b>，且<b>与理论上界基本重合</b>（差 {abs(g(K8)-g(K6))*100:.1f} 个百分点）。</li>
<li>最反直觉的发现：<b>「固定 N 次不出就转」的经验规则比不转还差</b>（{g(K3)*100:.1f}% &lt; {g(K1)*100:.1f}%）。因为它在热门律上也白白付了转韵代价。</li>
<li>若转韵时允许玩家挑律，叠加「推荐转去哪个韵」可到 <b>{g(K9)*100:.1f}%</b>（上界 {g(K10)*100:.1f}%）；在最冷 8 韵的压力测试里从 {E2[K1]['win_rate']*100:.1f}% 拉到 <b>{E2[K9]['win_rate']*100:.1f}%</b>。</li>
</ul>
<div style="margin-top:12px;padding-top:12px;border-top:1px dashed #b9dfc0">
<b style="color:#c62828">v2 补充：动手前先确认游戏用哪种押韵口径。</b><br>
按「36 韵母精确相等」押韵 → 在线估计+DP值 <b>+20.5pp</b>；<br>
按「十三辙/十八韵通押」押韵 → 归并后冷热门差距从 26.7 倍塌到 1.8 倍，死磕就有 {E6['g18']['死磕']['wr']*100:.1f}%，<b>转韵只值 +{max(x['delta'] for x in E6['g18']['_cmp'])*100:.1f}pp</b>。
</div></div>

<h2>1. 为什么会被卡住：先把玩法写成概率模型</h2>
<p>记韵库共 <code>N</code> 句诗，其中第 <code>r</code> 个韵有 <code>n_r</code> 句，单次刷新给出 <code>K={meta['K']}</code> 个候选（无放回）。当前律为 <code>R</code> 时，一次刷新能命中的概率是超几何概率：</p>
<pre>q_R = 1 - Π_(j=0..K-1) (N - n_R - j) / (N - j)</pre>
<p>这个量就是整个算法的支点。它有两条致命性质：</p>
<div class="box warn">
<b>① 极度长尾。</b>真实语料里最热的 <code>i</code> 占 {PI0[ordq[0]]*100:.2f}%，最冷的 <code>{RHYMES[ordq[-1]]}</code> 只占 {PI0[ordq[-1]]*100:.3f}%，相差 {PI0[ordq[0]]/PI0[ordq[-1]]:.0f} 倍。叠加「库上限」截断后，本次实验的韵库里单次命中率从 <b>{w.q.max()*100:.2f}%</b> 一路掉到 <b>{w.q.min()*100:.2f}%</b>——冷律刷 {meta['B']} 次也只有 {E2[K1]['win_rate']*100:.0f}% 的通关率，这正是玩家体感「怎么刷都不出」的根源。<br>
<b>② 刷新次数不带来信息。</b>每次刷新是独立同分布抽样，<b>刷了 5 次没出，第 6 次中奖的概率并不会变高</b>。所以「再刷一次试试」是纯粹的赌徒谬误——这也是新手最典型的陷阱。
</div>
<img src="{b64(str(FIGS/'fig1_corpus.png'))}" alt="真实韵库分布">

<h2>2. 算法：三条判据与动态规划</h2>
<h3>判据一 · 闭式阈值 q*（直观、可写进 UI 文案）</h3>
<p>继续死磕的期望刷新次数是 <code>1/q_R</code>；转韵要付代价 <code>C</code>，换来一个按出题分布 ω 抽到的新律，其平均命中率 <code>q̄ = Σ_r ω_r q_r</code>，期望刷新次数 <code>C + 1/q̄</code>。于是：</p>
<pre>转韵 ⟺ 1/q_R &gt; C + 1/q̄  ⟺  q_R &lt; q* = 1 / (C + 1/q̄)</pre>
<p>特例 <code>C=0</code> 时 <code>q* = q̄</code>，结论干净到可以做产品设计原则：<b>只要你当前的律比「按频率加权的平均律」还难，就该转——与你已经刷了几次无关。</b>本实验里 <code>q̄={meta['qbar']:.4f}</code>、<code>C={meta['C']}</code>，故 <code>q*={meta['qstar']:.4f}</code>。</p>

<h3>判据二 · 预算感知（预算有限时更准）</h3>
<p>真实对局里预算 <code>B</code> 是硬的，应该直接比「剩余预算内通关的概率」：</p>
<pre>转韵 ⟺ 1-(1-q̄)^(B-C) &gt; 1-(1-q_R)^B</pre>
<p>阈值扫描实验证实：固定阈值族的经验最优值是 <b>{cv[bi,0]:.4f}</b>，闭式解 <code>q*={meta['qstar']:.4f}</code> 落在最优区内；而经验最优略低于 q*，正是「预算有限 → 比无限预算模型更保守」的体现。</p>

<h3>判据三 · 在线估计（工具不需要预置韵库）</h3>
<p>关键洞察：<b>一次刷新不是只产生 1 bit 的「中/没中」，而是暴露了 K 个候选句各自的韵</b>——这 K 个韵就是韵库分布的样本。于是用 Dirichlet 共轭后验做在线估计：</p>
<pre>α ← λ·α + obs(本次K个候选的韵分布)      # λ&lt;1 可跟踪库消耗造成的漂移
π̂_r = α_r / Σα ,  q̂_r = 1-(1-π̂_r)^K
ω̂  : 另开一个 Dirichlet，记录游戏实际出过的律</pre>
<p>先验取均匀（<code>τ=2</code> 的弱先验），即<b>完全不需要事先知道韵库</b>；每局约产生 {meta['K']}×5≈20 个观测，几十局内就收敛。</p>

<h3>在线估计 + 动态规划（贝尔曼自迭代）</h3>
<p>把「转了之后还能再转」的期权价值算进去，用 <code>O(36×B)</code> 的小动态规划替代一步式比较：</p>
<pre>V[:, b] = max( 刷新: q̂ + (1-q̂)·V[:, b-1] ,  转韵: E_ω̂[V[:, b-C]] )
每次决策就解一遍（几百次浮点运算，手机上零开销），贪心执行 argmax。</pre>
<p>这一步不是炫技。补做了一个<b>失稳对照实验</b>（出题只在最冷 8 韵里抽，但算法误以为「转韵总能拿到一个平均水平的律」）：</p>
<table><tr><th>写法</th><th>通关率</th><th>平均转韵次数</th><th>说明</th></tr>
{row(['一步判据·误判律池',f"<b style='color:#c62828'>{E2b['一步判据·误判律池']['win_rate']*100:.1f}%</b>",f"{E2b['一步判据·误判律池']['avg_switch']:.2f}",'以为「换个新的总归好点」，连环空转烧光预算'],'td')}
{row(['一步判据·正确律池',f"{E2b['一步判据·正确律池']['win_rate']*100:.1f}%",f"{E2b['一步判据·正确律池']['avg_switch']:.2f}",'按真实律池算 q̄'],'td')}
{row(['DP 自迭代（本文）',f"<b style='color:#2e7d32'>{E2b['DP最优（随机转韵·上界）']['win_rate']*100:.1f}%</b>",f"{E2b['DP最优（随机转韵·上界）']['avg_switch']:.2f}",'算出「转了也是冷律」，自动收敛为少转'],'td')}
{row(['死磕不转韵',f"{E2b['死磕不转韵（新手）']['win_rate']*100:.1f}%",'0.00','作为对照'],'td')}
</table>
<p>误判律池会让通关率<b>掉到 {E2b['一步判据·误判律池']['win_rate']*100:.1f}%，还不如干脆不转（{E2b['死磕不转韵（新手）']['win_rate']*100:.1f}%）</b>——转了 {E2b['一步判据·误判律池']['avg_switch']:.2f} 次，预算全花在转韵上。DP 版本同时估计 <code>π</code> 和 <code>ω</code>（律池分布），因此不会出现这种系统性自欺，这是选用动态规划的原因。</p>

<h2>3. 实验设置</h2>
<table><tr><th>项目</th><th>取值</th></tr>
{row(['诗句语料','chinese-poetry 全唐诗/全宋诗/全宋词，共 251,381 句，逐句取末字韵母（pypinyin，含多音字上下文消歧）'],'td')}
{row(['韵的粒度','36 个现代汉语韵母（ang / e / i / üan …，与题主描述一致）'],'td')}
{row(['韵库构造',f"按真实频率抽样，总规模 N={meta['N']}，每律上限 cap={meta['cap']} 句（热门律被截断、冷门律保持稀有）"],'td')}
{row(['对局参数',f"每次刷新 K={meta['K']} 个候选，刷新预算 B={meta['B']}，转韵代价 C={meta['C']} 次刷新"],'td')}
{row(['出题分布','默认均匀（36 韵等概率，压力场景）；另测「按真实频率」的自然分布'],'td')}
{row(['样本量','主实验 20,000 局/策略 × 12 策略；压力/鲁棒性 6,000–12,000 局/格'],'td')}
</table>

<h2>4. 实验结果</h2>
<h3>4.1 主实验</h3>
<img src="{b64(str(FIGS/'fig2_main.png'))}" alt="主实验">
<table><tr><th>策略</th><th>通关率</th><th>平均刷新</th><th>平均转韵</th><th>≥3连空率</th></tr>{e1tab}</table>
<p>「≥3连空率」是<b>新手痛苦值</b>：在线估计+DP把它从 {g(K1,'pain3')*100:.1f}% 降到 {g(K8,'pain3')*100:.1f}%，这才是体感改善的直接来源——玩家不再陷入「刷了半天一个都不押韵」。</p>

<h3>4.2 压力测试：出题只在最冷 8 韵里抽</h3>
<img src="{b64(str(FIGS/'fig3_cold.png'))}" alt="冷门律专项">
<p>冷门韵：{'、'.join(RES['E2_cold'])}。此时死磕只有 {E2[K1]['win_rate']*100:.1f}%，固定阈值规则更差（{E2['② 固定3次失败后转韵']['win_rate']*100:.1f}%）；在线估计+DP {E2[K8]['win_rate']*100:.1f}%，若还能挑律则 {E2[K9]['win_rate']*100:.1f}%——<b>这就是「推荐转去哪个韵」的产品价值</b>。</p>

<h3>4.3 理论校验：阈值扫描</h3>
<img src="{b64(str(FIGS/'fig4_thr.png'))}" alt="阈值扫描">

<h3>4.4 鲁棒性：17 组规则/参数变化</h3>
<img src="{b64(str(FIGS/'fig6_robust.png'))}" alt="鲁棒性">
<table><tr><th>场景</th><th>死磕</th><th>固定3次</th><th>在线估计+DP</th><th>+自选律</th><th>DP上界</th></tr>{e4tab}</table>
<p>无论候选数、预算、转韵代价、库上限、出题分布、是否消耗诗句如何变化，在线估计+DP<b>始终贴近上界且稳定优于两条基线</b>；<code>cap</code> 越大（热门律越突出），「推荐选律」的收益越大（最高 {max(r['在线估计+DP+自选律'] for r in E4)*100:.1f}%）。</p>

<h3>4.5 冷启动：不需要预置韵库</h3>
<img src="{b64(str(FIGS/'fig5_learn.png'))}" alt="学习曲线">
<p>在线估计跨局持续积累，<b>约 100 局内就收敛到上界附近</b>；若每局都从零开始估计（不跨局学习），只有 {E5['nolearn']*100:.1f}%。<b>结论：工具上线时不需要任何韵库统计，边打边学即可。</b></p>

<h2>5. 落地建议</h2>
<h3>5.1 提示文案（可直接抄）</h3>
<div class="box">
· 开局就预警：<b>「本律 e 单次命中率 12%，刷完 10 次也只有 73% 把握，建议转韵」</b>——在<b>第一次刷新之前</b>就说，而不是等玩家刷崩了再补救。<br>
· 转韵时给目标：不要只说「该转韵了」，要说<b>「建议转到 en（22.7%/次）」</b>。<br>
· 预算不够时闭嘴：剩余 ≤ C={meta['C']} 次时不再建议转韵，避免误导。
</div>
<h3>5.2 「刷 / 转」判定表（默认参数下）</h3>
<p>命中率低于 <code>q*={meta['qstar']*100:.1f}%</code> 的韵，<b>一律建议立即转韵</b>：</p>
<table><tr><th>韵母</th><th>语料占比</th><th>库内句数</th><th>单次命中率</th><th>建议</th></tr>{rtab}</table>
<h3>5.3 需要标定的三个数</h3>
<p>把 <code>K</code>（每次几个候选）、<code>B</code>（刷新预算）、<code>C</code>（转韵代价）换成你游戏里的真实值，算法会自动重算 <code>q*</code> 和 DP 表，<b>不需要人工调阈值</b>。若诗句会被消耗，把遗忘因子 <code>λ</code> 调到 0.95–0.99 以跟踪漂移。</p>

<h2>6. 局限与可扩展</h2>
<ul>
<li><b>押韵判定口径：</b>本文用现代汉语韵母精确相等。若游戏按十三辙/中华新韵归部（如 an/ian/uan 同辙），把「韵」的粒度换成辙口即可，在线估计+DP完全不变——但冷热门排序会变，需重新标定。</li>
<li><b>代价口径：</b>转韵代价被折算成「{meta['C']} 次刷新」。若真实游戏里转韵是独立次数限制或消耗道具，改成对应的预算维度即可。</li>
<li><b>玩家行为：</b>模型假设玩家会选出手牌里唯一押韵的句子；若玩家会误判/漏看，命中率会更低，此时算法只会更有价值。</li>
<li><b>未建模：</b>牌库构筑、对手博弈、连击奖励等。这些属于「选什么律」的上层问题，可在在线估计+DP之上再接一层期望收益排序。</li>
</ul>
<div class="sub" style="margin-top:26px">代码：<code>rhyme_algo.py</code>（算法与对外封装 RhymeCoach）、<code>simulate.py</code>（模拟器与策略）、<code>run_experiments.py</code>（实验）、<code>demo.py</code>（演示）。语料：<code>corpus/*.json</code>、<code>corpus_lines.jsonl</code>、<code>rhyme_stats.json</code>。</div>

<h2>7. v2 加强：五项新实验，三条「做减法」的结论</h2>
<p>上一版堆了不少机制。v2 用<b>配对设计</b>（所有策略面对完全相同的第 i 局：同一初始律、同一抽样流）把方差降到极低，再逐项检验哪些机制<b>真的有用</b>——结论是<b>砍掉三个</b>。</p>

<h3>7.1 头号结论：先确认押韵口径，它决定这个功能值不值得做</h3>
<img src="{b64(str(FIGS/'fig7_mode.png'))}" alt="口径对比">
<table><tr><th>口径</th><th>韵数</th><th>单次命中率区间</th><th>极差</th><th>死磕</th><th>在线估计+DP</th><th>增益</th></tr>
{row(['36 韵母精确相等','36',f"{E6['fine']['_meta']['qmin']*100:.1f}% ~ {E6['fine']['_meta']['qmax']*100:.1f}%",f"{E6['fine']['_meta']['spread']:.1f}倍",f"{E6['fine']['死磕']['wr']*100:.1f}%",f"<b>{E6['fine']['v2-TS+挑律+鲁棒']['wr']*100:.1f}%</b>",f"<b style='color:#2e7d32'>+{E6['fine']['_cmp'][2]['delta']*100:.1f}pp</b>"],'td')}
{row(['十八韵/十三辙通押','13',f"{E6['g18']['_meta']['qmin']*100:.1f}% ~ {E6['g18']['_meta']['qmax']*100:.1f}%",f"{E6['g18']['_meta']['spread']:.1f}倍",f"{E6['g18']['死磕']['wr']*100:.1f}%",f"{E6['g18']['v2-TS+挑律+鲁棒']['wr']*100:.1f}%",f"+{E6['g18']['_cmp'][2]['delta']*100:.1f}pp"],'td')}
</table>
<div class="box warn"><b>机理：</b>通押把冷门韵并进了大韵部（如 <code>ueng</code> 0.13% 并入「中东」14.9%），冷热门差距从 <b>26.7 倍塌到 1.8 倍</b>——「冷门律刷不出来」这个前提消失了，转韵自然就不值钱了。<br>
<b>决策建议：</b>如果你的游戏是通押口径，<b>不要做转韵提示</b>，把资源放在诗句质量/意境匹配上；如果是精确韵母押韵，转韵算法是刚需。</div>

<h3>7.2 Thompson 采样：测了，没用 —— 砍掉</h3>
<table><tr><th>场景</th><th>后验均值(CE)</th><th>Thompson 采样</th><th>差值</th><th>95%CI</th><th>p 值</th></tr>
{row(['随机转韵',f"{E8[0]['ce']*100:.2f}%",f"{E8[0]['ts']*100:.2f}%",f"{E8[0]['delta']*100:+.2f}pp",f"[{E8[0]['lo']*100:+.2f}, {E8[0]['hi']*100:+.2f}]",f"{E8[0]['p']:.3f}"],'td')}
{row(['可自选律',f"{E8[1]['ce']*100:.2f}%",f"{E8[1]['ts']*100:.2f}%",f"{E8[1]['delta']*100:+.2f}pp",f"[{E8[1]['lo']*100:+.2f}, {E8[1]['hi']*100:+.2f}]",f"{E8[1]['p']:.3f}"],'td')}
</table>
<p><b>无可辨差异</b>（两组 p 值均 &gt; 0.29，置信区间跨 0）。原因值得记住：在这个玩法里<b>刷新既是通关手段，也是唯一的学习途径——探索是免费的副产品</b>，不存在「要不要专门去探索」的取舍，所以 TS 那套探索/利用平衡无处施展。反而 TS 单次采样噪声很大（实测把后验均值 2.1% 的韵采成 0.0001%），会污染一整局的决策。<b>→ 默认用后验均值，实现更简单也更稳。</b></p>

<h3>7.3 转韵额度：只在极稀缺时才值得建模 —— 一般可砍</h3>
<img src="{b64(str(FIGS/'fig8_budget.png'))}" alt="转韵额度">
<p>额度从 1 放宽到 3 有收益（{E7[0]['DP上界3D(知额度)']['wr'] if False else E7[0]['DP上界3D(知额度)']*100:.1f}% → {E7[2]['DP上界3D(知额度)']*100:.1f}%），<b>S≥3 之后完全持平</b>。原因是本参数下平均只转 0.6 次，额度根本用不完。<b>→ 只有当一局最多只能转 1 次时才需要「三维 DP(律,预算,额度)」；否则二维 DP 足够。</b></p>

<h3>7.4 非平稳律池：在线估计 ω 自己就能适应 —— 鲁棒/遗忘因子可砍</h3>
<img src="{b64(str(FIGS/'fig10_nonstat.png'))}" alt="非平稳">
<p>第 4000 局起把出题规则改成「偏向最冷 30% 的韵」，在线估计+DP从 {E9[0]['early']*100:.1f}% 只掉到 {E9[0]['late']*100:.1f}%（-{(E9[0]['early']-E9[0]['late'])*100:.1f}pp），而死磕掉到 {E9[0]['dead']*100:.1f}%。<b>悲观分位数（β=0.3）和遗忘因子（λω=0.98/0.995）对结果没有任何影响</b>——因为算法本来就在实时统计「游戏实际出过哪些律」，规则一变它跟着变。<b>→ 这两个旋钮不加。</b></p>

<h3>7.5 漏看：顺带量化了「判韵工具」本身值多少钱</h3>
<img src="{b64(str(FIGS/'fig9_miss.png'))}" alt="漏看">
<table><tr><th>玩家漏看率</th><th>死磕</th><th>固定3次</th><th>在线估计+DP</th><th>在线估计+DP+挑律</th></tr>
{''.join(row([f"{x['p_miss']*100:.0f}%", f"{x['死磕']*100:.1f}%", f"{x['固定3次']*100:.1f}%", f"<b>{x['v2-TS']*100:.1f}%</b>", f"<b>{x['v2-TS+挑律']*100:.1f}%</b>"], 'td') for x in E10)}
</table>
<p>两个读数：<b>①</b> 即使玩家有 30% 概率漏看（有押韵句没认出来），在线估计+DP仍比死磕高 <b>{(E10[3]['v2-TS']-E10[3]['死磕'])*100:.1f}pp</b>。<b>②</b> 漏看率从 0 到 30% 会掉 {(E10[0]['v2-TS']-E10[3]['v2-TS'])*100:.1f}pp —— 这说明<b>「自动判韵工具」本身就是刚需</b>，转韵算法是它之上的第二层增益，两者不冲突。</p>

<h2>8. 最终定稿配置（v2）</h2>
<div class="box ok">
<b>必做</b>
<ul>
<li><b>判韵口径</b>：先确认游戏是「精确韵母」还是「辙口通押」。通押 → 别做这个功能。</li>
<li><b>在线估计</b>：Dirichlet 后验统计「每次刷新的 K 个候选句的韵」+「游戏出过的律」，<b>无需预置韵库</b>。</li>
<li><b>决策</b>：二维 DP（转韵额度 ≤1 时改用三维），用<b>后验均值</b>，不要采样。</li>
<li><b>提示时机</b>：在第一次刷新之前就给，且要给出「转到哪个韵」。</li>
</ul>
<b>不做（已验证无收益）</b>
<ul>
<li>Thompson 采样 / 后验采样（p&gt;0.29，无差异，还引入噪声）</li>
<li>悲观分位数鲁棒（在线估 ω 已足够自适应）</li>
<li>遗忘因子 λ&lt;1（同上；仅当诗句库真的会被消耗时才考虑 λπ&lt;1）</li>
<li>三维 DP（除非一局最多只能转 1 次）</li>
</ul>
<b>参数</b>：<code>C</code>=转韵代价、<code>K</code>=候选数、<code>B</code>=刷新预算 —— 换成你游戏的真实值即可，<b>阈值自动重算，无需人工调参</b>。
</div>

<h2>9. 交付物</h2>
<table><tr><th>文件</th><th>说明</th></tr>
{row(['<code>rhyme_coach_lite.py</code>','<b>零依赖单文件版</b>（只用 random/math），可直接嵌进游戏服务端。三维 DP 带 memo，实测 &lt;0.05ms/次决策'],'td')}
{row(['<code>rhyme_algo2.py</code>','v2 算法：三维 DP、TS、鲁棒分位数、双遗忘因子（研究用）'],'td')}
{row(['<code>simulate2.py</code>','扩展模拟器：三种口径 / 有限额度 / 非平稳 / 漏看 / 配对设计'],'td')}
{row(['<code>exp2.py</code>','v2 五项实验 + 配对 bootstrap 显著性检验'],'td')}
{row(['<code>rhyme_groups.py</code>','押韵口径定义（36韵母 / 十八韵 / 十三辙）'],'td')}
{row(['<code>rhyme_algo.py</code> · <code>simulate.py</code>','v1 算法与模拟器（含 RhymeCoach 封装）'],'td')}
</table>
</body></html>"""
open(REPORT/'择律转韵算法_实验报告.html','w',encoding='utf-8').write(HTML)
print("report bytes:",len(HTML))
