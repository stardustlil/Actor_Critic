# Actor-Critic 实验框架（A2C / QAC）

这是一个面向离散动作空间的 Actor-Critic 强化学习项目，目前支持：

- `A2C`（Advantage Actor-Critic）
- `QAC`（Q-based Actor-Critic）

项目已经改造成**可配置切换算法**的结构，并预留了统一接口，便于后续扩展更多算法。

## 核心特性

- 通过配置项 `algorithm` 在 `a2c` / `qac` 间自由切换。
- 统一的 Agent 接口（`BaseAgent`）与算法注册机制（`AGENT_REGISTRY`），方便新增算法。
- 训练过程内置可视化输出（奖励曲线、Actor/Critic loss 曲线）。
- 新增测试模块，覆盖算法工厂与可视化模块基础功能。

## 项目结构

```text
Actor_Critic/
├── agents/
│   ├── __init__.py          # 算法注册与工厂
│   ├── base_agent.py        # 统一智能体接口
│   ├── a2c_agent.py         # A2C 实现
│   └── qac_agent.py         # QAC 实现
├── config/
│   └── default.yaml         # 默认配置（含 algorithm/可视化配置）
├── env/
│   └── make_env.py          # 环境构建
├── models/
│   ├── actor.py             # 策略网络
│   └── critic.py            # V/Q 价值网络
├── tests/
│   ├── test_agent_factory.py
│   └── test_visualization.py
├── utils/
│   ├── config.py            # 配置加载
│   └── visualization.py     # 训练曲线可视化
├── main.py                  # 入口（创建环境、构建算法、开始训练）
└── trainer.py               # 训练循环
```

## 快速开始

1. 安装依赖（示例）

```bash
pip install torch gymnasium pyyaml matplotlib
```

2. 使用默认配置训练（默认 `a2c`）

```bash
python main.py
```

3. 切换到 QAC

```bash
python main.py --algorithm qac
```

4. 使用自定义配置

```bash
python main.py --config config/default.yaml --algorithm a2c --num_episodes 300
```

## 配置说明（节选）

`config/default.yaml`：

- `algorithm`: `a2c` 或 `qac`
- `num_episodes`: 训练回合数
- `gamma`: 折扣因子
- `actor_lr`, `critic_lr`: 学习率
- `enable_visualization`: 是否保存训练曲线
- `plot_path`: 曲线保存路径

## 扩展新算法（建议步骤）

1. 在 `agents/` 新增算法文件（如 `ppo_agent.py`），并继承 `BaseAgent`。
2. 实现统一方法：
   - `select_action(state, deterministic=False)`
   - `update(transition)`
3. 在 `agents/__init__.py` 的 `AGENT_REGISTRY` 注册新算法。
4. 在配置文件中将 `algorithm` 设为新键名即可接入主流程。

## 测试

运行：

```bash
python -m unittest discover -s tests
```

测试内容：

- 算法工厂是否正确创建 A2C / QAC。
- 无效算法名是否正确抛错。
- 可视化模块是否能成功输出图片文件。
