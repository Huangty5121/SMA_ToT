# 基于 SMA-ToT 的多路径推理优化方法  
# Multi-Path Reasoning Optimization via SMA-ToT

本项目提出了一种全新设计的 SMA-ToT 框架，其构造完全基于黏菌算法（SMA）的随机“位移”更新机制，同时嵌入树状思维（ToT）的多路径探索结构。本文详细描述了该方法的理论推导、公式构建以及参数自适应机制，旨在为复杂推理任务提供一种鲁棒且高效的优化路径。

This project proposes a novel SMA-ToT framework, which is entirely based on the original random displacement update mechanism of the Slime Mould Algorithm (SMA) and integrated into the multi-path exploration structure of Tree-of-Thoughts (ToT). This document details the theoretical derivation, formulation, and adaptive parameter mechanism of the method, aiming to provide a robust and efficient optimization strategy for complex reasoning tasks.

---

## 1. 从 SMA 原始机制出发 / 1. Theoretical Basis from the Original SMA Mechanism

在经典的黏菌算法中，对于候选解 $X_i$（例如搜索空间中的一个位置），更新公式通常采用如下形式（示意性版本）：  
In the classic SMA, for a candidate solution $X_i$ (e.g., a position in the search space), the update equation is typically given as (a representative version):

$$
X_i^{t+1} =
\begin{cases}
X_{\text{best}}^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( X_i^t - X_{\text{best}}^t \right), & \text{if } r_3 < p, \\
X_i^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( X_i^t - X_{\text{best}}^t \right), & \text{otherwise,}
\end{cases}
$$

其中：  
Where:  
- $X_{\text{best}}^t$ 表示当前迭代中表现最优的解；  
  $X_{\text{best}}^t$ denotes the best solution at iteration $t$.  
- $v$ 为步长参数（可设计为随迭代逐渐衰减，如 $v_t = v_0\exp(-\lambda t)$）；  
  $v$ is the step size parameter (which can be designed to decay over iterations, e.g., $v_t = v_0\exp(-\lambda t)$).  
- $r_2$ 与 $r_3$ 均为服从均匀分布 $U(0,1)$ 的随机变量；  
  $r_2$ and $r_3$ are uniformly distributed random variables in $U(0,1)$.  
- $p$ 为切换阈值，用于决定是向当前最优解靠拢还是保持随机性更新。  
  $p$ is a switching threshold to decide whether to move toward the current best solution or to maintain a random update.

该公式体现了黏菌在寻找食物时的动态适应性：在部分时刻，黏菌倾向于向食物丰富区域靠拢，而在另一些时刻则保持一定随机性以探索更广泛的区域。  
This equation reflects the dynamic adaptability of slime moulds in foraging: at certain moments, they tend to move toward food-rich regions, while at other times, they maintain randomness to explore wider areas.

---

## 2. 融入 ToT 架构的思想 / 2. Incorporating the ToT Framework

在 ToT 框架中，我们构建一棵由起始节点 $s_0$ 到结束节点 $s_f$ 的推理树，每条边代表一个推理步骤。为每条边 $(i,j)$ 定义一个“质量评分” $S_{ij}$（初始时令 $S_{ij}^0 = S_0$），用于衡量该推理步骤的优劣，并指导后续路径选择。  
In the ToT framework, we construct a reasoning tree from a starting node $s_0$ to an ending node $s_f$, where each edge represents a reasoning step. We assign a "quality score" $S_{ij}$ (initialized as $S_{ij}^0 = S_0$) for each edge $(i,j)$ to measure its quality and guide subsequent path selection.

---

## 3. SMA-ToT 中的各个公式 / 3. Formulations in SMA-ToT

### 3.1 边选择概率 / 3.1 Edge Selection Probability

对于某个节点 $i$ 下的候选下一状态集合 $N(i)$，采用 softmax 确定选边概率：

$$
P_{ij}^t = \frac{\exp(S_{ij}^t/\tau)}{\sum_{l\in N(i)}\exp(S_{il}^t/\tau)}
$$

其中：  
Where:  
- $\tau$ 是温度参数，用于平衡探索与开发。  
  $\tau$ is the temperature parameter to balance exploration and exploitation.

---

### 3.2 基于 SMA 机制的评分更新 / 3.2 Score Update Based on SMA Mechanism

令当前节点 $i$ 下所有候选边的最高评分为

$$
S_{\text{best}}^t = \max_{l\in N(i)} S_{il}^t.
$$

对于边 $(i,j)$ 的更新公式为：

$$
S_{ij}^{t+1} =
\begin{cases}
S_{\text{best}}^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( S_{ij}^t - S_{\text{best}}^t \right), & \text{if } r_3 < p, \\
S_{ij}^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( S_{ij}^t - S_{\text{best}}^t \right), & \text{otherwise,}
\end{cases}
$$

其中：  
Where:  
- $v$ 为步长参数（可随迭代调整，如 $v_t = v_0\exp(-\lambda t)$）；  
  $v$ is the step size parameter (which may be adjusted over iterations, e.g., $v_t = v_0\exp(-\lambda t)$).  
- $r_2, r_3 \sim U(0,1)$ 是随机变量；  
  $r_2, r_3 \sim U(0,1)$ are random variables.  
- $p$ 为切换阈值。  
  $p$ is the switching threshold.

该更新机制的意义在于：  
- 当随机数 $r_3 < p$ 时，评分向当前最优评分靠拢（模仿黏菌向食物丰富区域移动）；  
- 否则，评分以随机方式更新，保留探索新路径的可能性。  
The significance of this update mechanism is that:  
- If $r_3 < p$, the score moves toward the current best score (mimicking slime mould moving towards food-rich areas);  
- Otherwise, the score is updated randomly, preserving the possibility of exploring new paths.

---

### 3.3 质量反馈更新 / 3.3 Quality Feedback Update

生成一条完整推理路径 $P$（从 $s_0$ 到 $s_f$）后，定义其质量评价函数为：

$$
Q(P) = w_1 C(P) + w_2 L(P) + w_3 E(P),
$$

其中：  
Where:  
- $C(P)$ 表示语义连贯性（如相邻状态的嵌入相似度）；  
  $C(P)$ represents semantic coherence (e.g., similarity between embeddings of adjacent states).  
- $L(P)$ 表示路径长度惩罚（如 $-\ln(|P|)$）；  
  $L(P)$ is a length penalty (e.g., $-\ln(|P|)$).  
- $E(P)$ 表示专家共识评分；  
  $E(P)$ is the expert consensus score.  
- $w_1, w_2, w_3$ 为权重。  
  $w_1, w_2, w_3$ are the corresponding weights.

对于路径 $P$ 中的每条边 $(i,j)$，在更新后的评分上加上质量反馈项：

$$
S_{ij}^{t+1} \leftarrow S_{ij}^{t+1} + \gamma \cdot Q(P),
$$

其中 $\gamma$ 为反馈权重参数。  
To prevent unbounded accumulation, a decay is applied:

$$
S_{ij}^{t+1} \leftarrow (1-\delta) \, S_{ij}^{t+1},
$$

其中 $\delta$ 是衰减率。  
where $\delta$ is the decay rate.

---

### 3.4 可选的衰减项 / 3.4 Optional Decay Term

为防止评分无限累积，更新后引入衰减项：

$$
S_{ij}^{t+1} \leftarrow (1-\delta) \, S_{ij}^{t+1}.
$$

---

## 4. SMA-ToT 算法流程 / 4. SMA-ToT Algorithm Flow

1. **初始化 / Initialization**  
   - 利用中心 LLM 生成初始 ToT 树，为每条边 $(i,j)$ 设定初始评分 $S_{ij}^0 = S_0$.

2. **路径生成 / Path Generation**  
   - 对于每个节点 $i$，根据边选择概率

     $$
     P_{ij}^t = \frac{\exp(S_{ij}^t/\tau)}{\sum_{l\in N(i)}\exp(S_{il}^t/\tau)}
     $$

     采样得到下一状态，直至生成完整推理路径 $P_k$（由多个专家独立生成）。

3. **路径评价 / Path Evaluation**  
   - 对每条生成的推理路径 $P_k$ 计算质量评价 $Q(P_k)$.

4. **评分更新 / Score Update**  
   - 对于路径 $P_k$ 中的每条边 $(i,j)$，使用公式更新评分：

     $$
     S_{ij}^{t+1} =
     \begin{cases}
     S_{\text{best}}^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( S_{ij}^t - S_{\text{best}}^t \right), & r_3 < p, \\
     S_{ij}^t + v \cdot \ln\left(\frac{1}{r_2}\right) \cdot \left( S_{ij}^t - S_{\text{best}}^t \right), & \text{otherwise,}
     \end{cases}
     $$

     并加上反馈：

     $$
     S_{ij}^{t+1} \leftarrow S_{ij}^{t+1} + \gamma \cdot Q(P_k),
     $$

     最后对评分进行衰减：

     $$
     S_{ij}^{t+1} \leftarrow (1-\delta) \, S_{ij}^{t+1}.
     $$

5. **迭代更新与最优路径提取 / Iterative Update and Optimal Path Extraction**  
   - 重复路径生成与评分更新，直至达到最大迭代次数或满足收敛条件。  
   - 最后，提取累计评分最高的推理链作为最终解答。

6. **参数自动调整（基于 MAML） / Adaptive Parameter Adjustment (via MAML)**  
   - 可调参数包括：步长 $v$ 及其衰减率 $\lambda$、切换阈值 $p$、温度参数 $\tau$、反馈权重 $\gamma$、衰减率 $\delta$，以及质量函数中的权重 $w_1, w_2, w_3$.  
   - 通过模型无关元学习（MAML），在面对新任务时，通过少量梯度更新自动优化这些参数，从而提高系统的适应性和鲁棒性.

---

## 5. 方法优势与特性 / 5. Advantages and Characteristics

该 SMA-ToT 框架充分体现了 SMA 的四个核心特性，同时与 ToT 架构目标紧密结合：

1. **自组织能力 / Self-Organization**  
   - 评分更新机制使整个推理树自发调整边的连接结构，高质量路径经过正反馈强化，自然脱颖而出.  
   The score update mechanism enables the reasoning tree to self-organize by dynamically adjusting the connection structure; high-quality paths are reinforced through positive feedback.

2. **动态路径调整 / Dynamic Path Adjustment**  
   - 采用随机“位移”更新机制，在一定概率下向当前最优靠拢，同时保留随机性，平衡局部搜索与全局探索.  
   The random displacement update mechanism allows paths to adjust dynamically—moving toward the current best solution with a certain probability while retaining randomness to balance local and global search.

3. **鲁棒性和灵活性 / Robustness and Flexibility**  
   - 由于不依赖于问题的梯度信息，该方法在处理非凸、非连续或噪声较大的问题时具有良好的适应性.  
   Not relying on gradient information makes the method robust and flexible for handling non-convex, discontinuous, or noisy problems.

4. **信息共享与反馈机制 / Information Sharing and Feedback Mechanism**  
   - 通过对完整推理路径的质量评价，将反馈分散到各边，强化优秀路径，抑制低效路径，实现全局最优解的自组织搜索.  
   By evaluating the quality of complete reasoning paths and distributing the feedback across edges, the method reinforces superior paths and suppresses inefficient ones, achieving self-organized global optimization.

---

## 总结 / Conclusion

本项目提出的 SMA-ToT 框架完全基于 SMA 的随机“位移”更新机制，并融合 ToT 的多路径探索结构。该方法通过对每条边赋予动态质量评分 $S_{ij}$，并结合完整路径的质量反馈与可选衰减机制，实现了推理路径的自组织与动态调整。同时，通过 MAML 框架对关键参数进行自动调整，SMA-ToT 能够在面对复杂推理任务时快速适应并输出高质量解答。  
  
This project introduces the SMA-ToT framework, which is entirely based on the random displacement update mechanism of SMA and integrates the multi-path exploration structure of ToT. By assigning dynamic quality scores $S_{ij}$ to each edge and combining quality feedback with optional decay, the method enables self-organization and dynamic adjustment of reasoning paths. Moreover, adaptive parameter tuning via MAML allows SMA-ToT to quickly adapt to complex reasoning tasks and produce high-quality solutions.
