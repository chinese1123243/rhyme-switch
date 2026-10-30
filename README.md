# rhyme-switch

择律/押韵类玩法中「何时该转韵」的决策算法，附真实语料上的蒙特卡洛验证。

## 原理

韵库共 N 句，韵 r 有 n_r 句，每次刷新给出 K 个候选（无放回）。单次刷新命中概率：

```
q_r = 1 - Π_{j=0..K-1} (N - n_r - j) / (N - j)
```

每次刷新会暴露 K 个候选句的韵，作为韵库分布 π 的样本；游戏每关出的律记为 ω。
两者各用一个 Dirichlet 后验累计。决策用动态规划，状态为 (当前律, 剩余刷新次数 b, 剩余转韵额度 s)：

```
V[r,b,s] = max( q_r + (1-q_r)·V[r,b-1,s] ,  E_ω[V[:,b-C,s-1]] )
```

C 为转韵代价，折算成刷新次数。转韵时可自选律时，右侧取 max 而非期望。

## 实测

语料：chinese-poetry 全唐诗 / 全宋诗 / 全宋词，251,381 句，逐句取末字韵母。

默认对局 K=4、B=10（刷新预算）、C=2，36 韵母口径，20,000 局/策略：

| 策略 | 通关率 | 平均刷新 | 平均转韵 |
|---|---|---|---|
| 死磕不转韵 | 60.0% | 6.64 | 0.00 |
| 固定 2 次不出就转 | 48.1% | 4.64 | 1.45 |
| 固定 3 次不出就转 | 56.6% | 5.66 | 0.73 |
| 闭式阈值 q* | 70.8% | 5.09 | 0.66 |
| 在线估计 + DP | 70.8% | 5.14 | 0.64 |
| 在线估计 + DP + 自选律 | 80.7% | 4.63 | 0.67 |
| DP 上界（已知真实概率） | 71.1% / 80.7% | — | — |

配对实验（15,000 局，同一初始律与同一抽样流）：

| 场景 | 死磕 | 固定 3 次 | 在线估计+DP | +自选律 |
|---|---|---|---|---|
| 36 韵母精确相等 | 60.4% | 56.9% | 70.9% | 80.9% |
| 十八韵 / 十三辙通押 | 95.2% | 92.3% | 96.0% | 96.4% |
| 漏看率 30% | 49.2% | 44.7% | 58.0% | 67.6% |
| 出题规则中途改为偏向冷门 | 38.6% | — | 79.3% | — |

单项结论：

- 押韵口径决定收益大小。36 韵母口径冷热门命中率相差 26.7 倍，通押归并后降到 1.8 倍，转韵只值 +1.2pp。
- 转韵额度 S=1 → 69.1%，S=2 → 71.4%，S=3 → 71.7%，S≥5 持平。默认参数下平均只转 0.6 次。
- Thompson 采样与后验均值无差异：Δ=-0.05pp（p=0.295）、Δ=+0.28pp（p=0.246）。
- 律池分布被误判时，一步式判据会连环空转，通关率 9.4%，低于不转韵的 20.6%。
- 在线估计约 100 局收敛；每局重置后验的对照组为 50.3%。

数据来源：`data/results.json`、`data/results2.json`，脚本见 `experiments/`。
完整报告：`report/择律转韵算法_实验报告.html`。

## 用法

```python
from rhyme_switch import LiteCoach, RHYMES36

coach = LiteCoach(RHYMES36, C=2, K=4)
coach.new_level()                                  # 每局开头
coach.see_refresh(['an', 'ing', 'i', 'an'])        # 每次刷新后喂 K 个候选的韵
coach.see_law('e')                                 # 本关的律

coach.advise('e', budget_left=10, switches_left=3)
# {'action': 'switch', 'q': 0.0242, 'target': 'i', ...}

coach.recommend_law(10, 3)                         # 转韵时推荐去哪个韵
```

不需要预置韵库统计，观测随刷新累计。决策耗时约 0.006 ms/次。

## 结构

```
rhyme_switch/   算法。coach 零依赖，algo 为 numpy 版，groups 为押韵口径
experiments/    模拟器与实验脚本，产出 data/results*.json
scripts/        语料构建与演示
data/           韵母统计与实验结果
report/         HTML 报告与图表
docs/           注释规范
```

## 复现

```bash
pip install -r requirements.txt
python experiments/run_v1.py     # E1~E5
python experiments/run_v2.py     # E6~E10
python experiments/run_e9.py     # E9 单跑
python experiments/plot_v1.py && python experiments/plot_v2.py
python experiments/build_report.py
```

语料需自行下载 chinese-poetry 到 `corpus/` 后运行 `scripts/build_corpus.py`；
`data/rhyme_stats.json` 已随仓库提供，实验可直接运行。

## License

MIT
