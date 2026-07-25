# Twilight Struggle 智能体环境

[English](README.en.md) | 简体中文

一个《冷战热斗》(*Twilight Struggle*，GMT Games，豪华版) 的规则引擎，专门用于**测试和训练
智能体**——强化学习策略、大模型智能体，以及在这个游戏上做后训练的大模型。

棋盘和卡牌数据库是**从游戏自带的 Lua 文件中提取**的，而不是手工抄录，因此国家稳定值、
战场国标记、邻接关系和每张卡的属性都是权威的。

```
python tools/extract_lua.py     # 从游戏安装目录重新生成 twilight/data/*.json
python -m pytest tests -q       # 197 个测试
python tools/random_play.py --games 200
python tools/demo_views.py      # 查看同一局面下两种智能体视图
```

## 设计

### 一份状态，三种视图

智能体看到的一切都是 `GameState` 的纯函数。信息隐藏只在一个地方发生——`observe(state, player)`
——所以强化学习用的数值视图和大模型用的文本视图**不可能**对事实产生分歧，也不可能泄露不同
数量的信息。

```
GameState  ──►  observe(state, player)  ──►  Observation
                (隐藏对手手牌，                  │
                 隐藏牌堆顺序)                   ├──►  encode(obs)  → numpy 字典 + 动作掩码
                                                └──►  render(obs)  → 文本 + 编号菜单
```

### 因子化的动作空间

这是**最重要的一个设计决策**。冷战热斗的一个回合本质上是一棵微决策的树：打出一张牌 → 选择
用法 → 逐个选择目标。如果一次 `step()` 等于一整个回合，动作空间会组合爆炸（把 4 点行动力
分配到 84 个国家上有数百万种可能）。

引擎改为产出一个**原子决策流**，每个决策的合法集合都很小——通常 5 到 200 项。一个原子动作
≈ 真实游戏里点一下鼠标。

```python
game = Game(seed=0)
while game.decision is not None:
    d = game.decision          # d.player, d.type, d.prompt, d.options
    game.step(pick(d))         # 可以传 Action、它的规范 key，或词表下标
```

动作词表是**封闭且有序的**（237 项），所以下标 *i* 在任何一局里含义都一样：

| 类别 | key 示例 | 数量 |
|---|---|---|
| 选一张牌 | `card:Duck and Cover` | 110 |
| 选择用法 | `use:coup` | 6 |
| 选一个国家 | `country:West Germany` | 84 |
| 选一个地区 | `region:Middle East` | 9 |
| 选一个数量 | `number:3` | 13 |
| 事件专属选项 | `option:1` | 12 |
| `yes` / `no` / `pass` | `pass` | 3 |

这一个设计同时服务三条路线：强化学习拿到固定的输出头加布尔掩码；大模型拿到一份能读得懂的
短菜单，以及一套可用于约束解码的稳定语法。

### 卡牌事件写成生成器

整局游戏就是一个 Python 生成器。卡牌效果 `yield` 出决策、再接收选择结果，所以嵌套很深的
事件也不需要显式状态机：

```python
@register("Socialist Governments", playable_if=_not_while_iron_lady)
def socialist_governments(game, ctx):
    """从西欧移除 3 点美国影响力，每国最多 2 点。"""
    yield from game.remove_influence(
        Side.USA, 3,
        allowed=in_region(Region.WESTERN_EUROPE),
        max_per_country=2,
        chooser=ctx.player,
    )
```

这个选择的代价：运行中的对局持有活跃的生成器栈帧，无法深拷贝。所以 `Game.clone()` 用同一个
随机种子**重放动作历史**——结果精确，但复杂度是 O(对局长度)。如果要在长对局上做树搜索，
需要自己另做快照机制。

## 使用方式

### 大模型智能体

```python
from twilight import TwilightStruggleEnv

env = TwilightStruggleEnv(seed=0, encode_observations=False)
while env.decision is not None:
    prompt = env.text()                    # 棋盘 + 推导事实 + 编号菜单
    key = llm(prompt)                      # 例如 "country:Iran"
    env.step(key)
```

文本视图**特意预先算好了模型容易算错的东西**：各地区的控制等级、每个地区**此刻**结算能得
多少分、每个国家还差几点才能控制、政变成功的概率，以及 DEFCON 目前封锁了哪些地区。

```
=== You are USSR | Turn 1/10, action round 5 | action_round ===
VP +7 (you lead, +20 wins) | DEFCON 3 | space race you 1 vs 0 (attempts left 0)
military ops you 5 vs 0, need 3 by end of turn or opponent scores the shortfall
DEFCON 3 forbids coups and realignment in: Asia, Europe

BOARD (only countries with influence; ctrl = who controls)
  Europe -- you presence (1c/1bg), opponent presence (1c/0bg), 5 bg total; scoring now: +1 VP to you
    East Germany           BG stab3  USSR  4 / US  0  ctrl USSR
    UK                        stab5  USSR  0 / US  5  ctrl US
    ...

IF SCORED NOW (net VP to you)
    Europe                 +1   (you presence)
    Central America        +2   (you presence)
```

### 强化学习

```python
env = TwilightStruggleEnv(seed=0, reward_mode="sparse")
obs, info = env.reset()
while True:
    logits = policy(obs["countries"], obs["global"], obs["hand"])
    action = sample(logits, mask=obs["action_mask"])
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

棋盘以 `(84, 27)` 矩阵给出——每个国家一行，顺序固定——这样网络可以在国家之间共享权重、
对它们做注意力，而不是被塞一个扁平向量。需要 MLP 基线时用 `flatten()` 拼成 2967 维。

| 组成部分 | 形状 |
|---|---|
| `countries` | `(84, 27)` |
| `global` | `(149,)` |
| `hand` / `discard` / `removed` / `effects` / `deck_possible` | 各 `(110,)` |
| `action_mask` | `(237,)` |

`deck_possible` 是刻意提供的：真实游戏里弃牌堆和移除堆都是公开信息，所以牌堆构成是可以推断
的。**数牌是这个游戏里真实存在的技巧**，观测直接把这个推断结果交给智能体，而不是让它自己
重新推导。

**双人对局约定。** 决策交替是不规则的，同一方经常连续行动多次，所以这里采用 PettingZoo 的
回合制约定，而不是假装自己是单智能体环境：`info["to_move"]` 指出该谁行动，`step()` 返回
**站在该方视角**的奖励，`info["rewards"]` 两方相加为零。

**奖励模式。** `sparse` 是终局 ±1、其余为 0——这是诚实的目标函数。`vp_delta` 额外加上每步
胜利点数变化（除以 20 缩放）。注意这里的 VP 塑形**不是策略不变的**：控制欧洲、DEFCON 降到 1、
以及回合结束时手里还捏着计分牌，这三种结局都会在 VP 没有相应变动的情况下直接结束游戏，所以
只用 `vp_delta` 训出来的策略会系统性地处理不好这些情况。

### 大模型后训练

用 `encode_observations=False` 配合 `render()`。文本是确定性的——规范排序、用整数而非浮点、
表头稳定——所以同一个局面每次的 token 化结果完全一致。动作是简短的规范字符串，因此可以把
解码约束到一套语法上，并在 token 级别分配信用。`Decision.resolve()` 接受动作 key、`Action`
对象或词表下标，非法时抛出 `IllegalAction` 并在消息里带上合法集合——可以直接当作环境反馈
喂回给模型。

## 基线

```bash
python examples/baselines.py --games 40 --ussr greedy --usa safe_random
python examples/llm_agent.py --show-prompt      # 查看模型会看到什么
python examples/llm_agent.py --games 3          # 用桩模型跑通整个流程
```

三个参考智能体：`random`（下限）、`safe_random`（仍然随机，但拒绝两种立即输棋的动作）、
`greedy`（位置启发式）。

每种配对跑 40 局的实测结果：`greedy` 对 `random` 胜率 57%，但**并不能稳定战胜 `safe_random`**
（40–55%）。**这是一个真实结论，不是 bug。** 短对局由 DEFCON 边缘博弈主导，一个单纯"拒绝输棋"
的过滤器，表现优于一个必须被逐条教会所有输法的评分器。请把 `safe_random` 当作要超越的基线，
并且**始终在双方阵营都做评测**——存在先手优势。

两个 `safe_random` 对局平均能打到第 6 回合并进入最终结算；只要有一方是纯 `random`，对局
往往在第 1–2 回合就结束，因此它不是个好标尺。

如果你要自己写大模型循环，有一个解析细节值得记住：**动作 key 里含空格**
（`country:West Germany`）。用空白切分模型回复会把 key 截断，把本来正确的答案变成非法答案
——在修好之前，这**静默拒绝了 27% 的有效回复**。请改成对 `decision.legal_keys` 做匹配，
参见 `examples/llm_agent.py::extract_key`。

## 目录结构

```
twilight/
  data.py         不可变的 Country / Card 记录与索引，从 data/*.json 加载
  state.py        GameState：唯一的事实来源
  rules.py        控制判定、地区计分、政变、重整、influence 放置合法性
  spacerace.py    8 格太空竞赛轨道（豪华版数值）及其四项能力
  decisions.py    封闭动作词表与 Decision 对象
  engine.py       整局游戏作为一个生成器；卡牌事件所依赖的 API
  events/         每张卡一个处理函数，按卡名注册
  observe.py      state -> 单个玩家可以知道的信息
  encode.py       Observation -> 供学习型策略使用的 numpy 数组
  render.py       Observation -> 供大模型使用的文本
  env.py          reset/step 封装与奖励模式
tools/
  extract_lua.py  从游戏安装目录重新生成数据库
  dump_card_spec.py   生成 docs/card_spec.md：每张卡的文本 + 内部效果函数名
  random_play.py  带不变量检查的随机对局压力测试
  demo_views.py   并排展示两种视图
examples/
  baselines.py    random / safe_random / greedy 智能体与对局运行器
  llm_agent.py    提示词循环、非法输出重试、key 提取
tests/            197 个测试
docs/
  card_spec.md    自动生成：每张卡的规则文本与内部效果函数名
  known_gaps.md   哪些地方**没有**忠实实现，以及原因
```

## 规则忠实度

全部 110 张卡都实现了事件。各项常量都对照 **2015 版 GMT 豪华版规则书**和**官方 FAQ v5**
做了核对，并与四个独立的开源实现做了交叉验证。以下这些细节是实现中常见的错误点，这里是
正确的：

- **太空竞赛**用的是豪华版数值，不是一/二版（一二版把 6–8 格的名称和能力调换了，且
  Lunar Orbit 给 4/2 分）。能力只归**第一个**到达该格的玩家，对手到达同格时立即取消。
- **DEFCON 限制同时适用于重整，不只是政变**，并且东南亚计入亚洲。旧版棋盘上印的轨道漏掉了
  重整，那是错的。
- **军事行动是取净差**：双方都未达标时只结算差额，而不是各自的缺口分别结算。
- **优势 (Domination) 需要国家数**和**战场国数都多于对手**，还必须至少控制一个战场国和一个
  非战场国。棋盘附带的计分速查卡印得不完整。
- **influence 放置的可达范围是快照**，在行动轮开始时确定，所以本轮放下的 influence 不能
  向外接力去够到更远的国家。
- **重整失败会让进攻方自己损失** influence——骰子对抗是对称的。
- **免费政变**不受 DEFCON 地理限制、也不计入军事行动要求，但在战场国仍然会降低 DEFCON。
- **最终结算**结算所有地区，排除东南亚（已包含在亚洲内），并且控制欧洲依然直接获胜。
- 美国初始布置是 25 点 influence，**包含加拿大 2 点**——这一点很容易漏。

### 已知缺口

还有 3 条条款没有做到完全忠实，都列在 `docs/known_gaps.md` 里，同时记录了在资料互相矛盾时
我做出的裁定。**没有任何东西是被静默近似掉的**——`Game(strict_events=True)` 会对任何没有
处理函数的卡直接抛异常，测试套件也断言了完整覆盖。

卡面写着"在对手的下一个行动轮……"的条款走的是通用**延迟触发**机制
（`GameState.defer` / `Game._fire_deferred`），而不是逐卡打补丁；We Will Bury You 可被取消的
胜利点数、以及 Missile Envy 的强制出牌都用的这套机制。

## 数据来源与授权

代码采用 MIT 许可（见 `LICENSE`）。

`twilight/data/*.json` 与 `docs/card_spec.md` 是从本地正版游戏安装目录中的 Lua 数据文件
提取的，其中包含《冷战热斗》的卡牌规则文本与棋盘数据，**其版权归 GMT Games 及原作者所有**，
不在本仓库的 MIT 许可范围内。这些文件仅用于实现规则引擎与研究用途。使用者需自行持有游戏
正版；如需去除这部分数据，可删除这两处并用 `tools/extract_lua.py` 从自己的安装目录重新生成。
