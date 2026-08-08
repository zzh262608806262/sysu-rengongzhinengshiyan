# 深度强化学习实验报告
### 朱周昊 24319201
## 一、实验原理

### 1.1 深度Q网络（DQN）

深度Q网络（Deep Q-Network, DQN）是将深度学习与强化学习相结合的经典算法。它使用神经网络来近似Q函数，解决传统Q-learning在高维状态空间中的"维度灾难"问题。

**核心公式 - Bellman方程：**

```
Q(s, a) = r + γ * max_a' Q(s', a')
```

其中：
- `Q(s, a)`：状态s下采取动作a的预期累积奖励
- `r`：即时奖励
- `γ`：折扣因子（0≤γ≤1），决定未来奖励的重要性
- `s'`：下一状态
- `a'`：下一状态的最优动作

### 1.2 DQN关键技术

#### （1）经验回放（Experience Replay）
- 将智能体与环境的交互经验`(s, a, r, s', done)`存储在缓冲区中
- 训练时随机采样小批量数据，打破数据间的时间相关性
- 提高数据利用率和训练稳定性

#### （2）目标网络（Target Network）
- 使用两个结构相同但参数不同的Q网络：
  - **在线网络（Q Network）**：每步更新，用于选择动作和计算当前Q值
  - **目标网络（Target Network）**：定期同步，用于计算目标Q值
- 目标Q值计算公式：
  ```
  Target Q = r + γ * Q_target(s') * (1 - done)
  ```
- 定期将在线网络参数复制到目标网络，减少Q值估计的波动

#### （3）ε-greedy探索策略
- 以概率ε随机选择动作（探索）
- 以概率1-ε选择Q值最大的动作（利用）
- ε随训练逐渐衰减，从探索过渡到利用

### 1.3 CartPole环境

CartPole-v1是OpenAI Gym中的经典控制问题：
- **状态空间**：4维连续向量 [小车位置, 小车速度, 杆子角度, 杆子角速度]
- **动作空间**：2个离散动作 [向左推, 向右推]
- **奖励**：每存活一步获得+1奖励
- **终止条件**：
  - 杆子角度超过±12°
  - 小车位置超出±2.4
  - 达到最大步数500
- **求解标准**：连续100轮平均奖励≥195（本实验设为最近20轮平均≥180）

---

## 二、实现思路

### 2.1 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                        DQN Agent                            │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Q Network   │    │ Target Net   │    │ Replay Buffer│  │
│  │  (在线网络)   │    │ (目标网络)    │    │  (经验回放)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └────────┬──────────┘                   │          │
│                  │ 参数复制(每10轮)              │          │
│                  ▼                              │          │
│         ┌──────────────┐                        │          │
│         │ ε-greedy策略 │◄───────────────────────┘          │
│         └──────────────┘                                   │
└─────────────────────────────────────────────────────────────┘
                           ▲
                           │ 状态/奖励
┌─────────────────────────────────────────────────────────────┐
│                    CartPole Environment                     │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 训练流程

```
初始化环境、智能体、网络参数
        │
        ▼
┌─── For episode = 1 to 500 ───┐
│                               │
│  重置环境，获取初始状态s        │
│                               │
│  ┌── While not done ───┐     │
│  │                      │     │
│  │  ε-greedy选择动作a    │     │
│  │                      │     │
│  │  执行动作，获得(s',r,done) │
│  │                      │     │
│  │  存储经验到ReplayBuffer   │
│  │                      │     │
│  │  从Buffer采样训练网络    │
│  │                      │     │
│  │  s ← s'              │     │
│  │                      │     │
│  └──────────────────────┘     │
│                               │
│  每10轮同步目标网络            │
│                               │
│  检查是否达到求解标准           │
│                               │
└───────────────────────────────┘
```

### 2.3 网络结构设计

```
输入层 (4维状态)
    │
    ▼
全连接层 (128神经元) + ReLU
    │
    ▼
全连接层 (128神经元) + ReLU
    │
    ▼
输出层 (2维Q值)
```

---

## 三、关键代码

### 3.1 Q网络定义

```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)
```

### 3.2 ε-greedy动作选择（TODO 1）

```python
def select_action(self, state):
    # 探索：随机选择动作
    if random.random() < self.epsilon:
        return random.randrange(self.action_dim)
    
    # 利用：选择Q值最大的动作
    state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    with torch.no_grad():
        q_values = self.q_net(state)
    return q_values.argmax().item()
```

### 3.3 Bellman方程计算目标Q值（TODO 2）

```python
# 计算目标Q值
with torch.no_grad():
    # 目标网络计算下一状态的最大Q值
    next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
    # Bellman方程：Target = r + γ * max Q_target(s') * (1 - done)
    target_q = rewards + self.gamma * next_q * (1 - dones)

# 计算MSE损失
loss = nn.MSELoss()(current_q, target_q)
```

### 3.4 目标网络更新（TODO 3）

```python
def update_target(self):
    # 将在线网络的参数复制到目标网络
    self.target_net.load_state_dict(self.q_net.state_dict())
```

### 3.5 训练循环

```python
for episode in range(episodes):
    state, _ = env.reset(seed=42+episode)
    episode_reward = 0
    
    while True:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        agent.memory.push(state, action, reward, next_state, done)
        # 训练网络
        agent.train()
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # 每10轮同步目标网络
    if episode % 10 == 0:
        agent.update_target()
    
    # 检查求解条件
    avg_reward = np.mean(rewards_history[-20:])
    if avg_reward >= 180:
        print("\nSolved!")
        break
```

---

## 四、运行结果

### 4.1 训练日志

```
Episode:   0 Reward:  14.0 Avg:  14.0
Episode:   1 Reward:  17.0 Avg:  15.5
Episode:   2 Reward:  54.0 Avg:  28.3
...
Episode:  39 Reward: 182.0 Avg:  43.5
Episode:  40 Reward:  36.0 Avg:  44.8
...
Episode:  46 Reward: 292.0 Avg:  83.5
Episode:  47 Reward: 242.0 Avg:  94.7
...
Episode:  58 Reward: 252.0 Avg: 177.3
Episode:  59 Reward: 183.0 Avg: 177.4
Episode:  60 Reward: 179.0 Avg: 184.6

Solved!
```

### 4.2 关键指标

| 指标 | 数值 |
|------|------|
| 求解轮次 | 第61轮 |
| 最终平均奖励 | 184.6 |
| 单轮最高奖励 | 292（第46轮） |
| 训练总轮次 | 61轮 |

### 4.3 训练曲线分析

```
平均奖励变化趋势：

200 ┤                                              ╭──
    │                                         ╭────╯
150 ┤                                    ╭────╯
    │                               ╭────╯
100 ┤                          ╭────╯
    │                     ╭────╯
 50 ┤                ╭────╯
    │           ╭────╯
  0 ┼───────────╯
    0    10    20    30    40    50    60
              训练轮次 (Episode)
```

---

## 五、结果分析

### 5.1 训练阶段分析

#### 阶段一：探索期（Episode 0-30）
- **平均奖励**：17-28
- **特征**：ε值较高（接近1.0），智能体主要进行随机探索
- **表现**：杆子很快倒下，回合长度短
- **原因**：网络尚未学到有效策略，动作选择以随机为主

#### 阶段二：学习期（Episode 30-50）
- **平均奖励**：17→110
- **特征**：ε逐渐衰减，智能体开始利用学到的Q值
- **表现**：出现较长回合（如Episode 39达到182）
- **原因**：经验回放积累了足够样本，网络开始学习到平衡策略

#### 阶段三：提升期（Episode 50-60）
- **平均奖励**：110→185
- **特征**：策略快速优化，高奖励回合频繁出现
- **表现**：Episode 46达到最高奖励292
- **原因**：目标网络稳定更新，Q值估计更准确

#### 阶段四：求解（Episode 61）
- **平均奖励**：184.6 ≥ 180
- **结果**：满足求解条件，训练成功终止

### 5.2 算法有效性分析

1. **经验回放的作用**
   - 打破样本时间相关性，使训练数据更接近独立同分布
   - 重复利用历史经验，提高样本效率
   - 缓冲区容量10000，确保有足够的多样性样本

2. **目标网络的作用**
   - 每10轮同步一次，避免Q值目标频繁变化
   - 减少训练过程中的震荡，提高收敛稳定性
   - 从训练曲线可见，约30轮后开始稳定上升

3. **ε-greedy策略的作用**
   - 初期高ε保证充分探索状态空间
   - ε衰减因子0.995，确保逐步过渡到利用
   - 最小ε=0.01，保留少量探索避免陷入局部最优

### 5.3 超参数影响

| 超参数 | 设定值 | 影响分析 |
|--------|--------|----------|
| 学习率 | 1e-3 | 适中，保证稳定收敛 |
| 折扣因子γ | 0.99 | 重视长期奖励，适合持续性任务 |
| 批量大小 | 64 | 平衡训练效率和稳定性 |
| ε衰减率 | 0.995 | 约460轮衰减到0.1，本实验61轮时ε≈0.74 |
| 缓冲区容量 | 10000 | 足够存储多样化经验 |

### 5.4 改进方向

1. **Double DQN**：使用在线网络选择动作，目标网络评估Q值，减少Q值高估
2. **Prioritized Experience Replay**：优先采样TD误差大的经验，提高学习效率
3. **Dueling Network**：分离状态价值和优势函数，提升策略评估准确性
4. **调整超参数**：增大学习率或调整ε衰减策略可能加速收敛

---

## 六、实验总结

本次实验成功实现了基于DQN的CartPole平衡控制算法。通过经验回放、目标网络和ε-greedy探索策略三大核心技术，智能体在61轮训练后达到求解标准（平均奖励≥180）。

实验验证了深度强化学习在连续状态空间、离散动作空间控制任务中的有效性，为后续学习更复杂的强化学习算法（如Policy Gradient、PPO、SAC等）奠定了基础。

---

