# QwenVL-RFT 的 PPO 训练流程

本文结合数学公式和当前代码，说明一批视觉问答数据如何经过 Qwen2.5-VL 的 PPO 训练，最终产生 LoRA adapter、value head、指标和测试报告。

主要代码位置：

- 训练入口：[`src/qwen_vl_rl/cli/train_ppo.py`](src/qwen_vl_rl/cli/train_ppo.py)
- 公共 RL 循环：[`src/qwen_vl_rl/training/rl.py`](src/qwen_vl_rl/training/rl.py)
- PPO rollout、GAE 和 loss：[`src/qwen_vl_rl/algorithms/ppo.py`](src/qwen_vl_rl/algorithms/ppo.py)
- Actor-Critic 模型：[`src/qwen_vl_rl/models/policy.py`](src/qwen_vl_rl/models/policy.py)
- 数据 collator：[`src/qwen_vl_rl/data/collators.py`](src/qwen_vl_rl/data/collators.py)
- 答案 reward：[`src/qwen_vl_rl/algorithms/reward.py`](src/qwen_vl_rl/algorithms/reward.py)
- 默认配置：[`configs/ppo_qwen_vl_lora.yaml`](configs/ppo_qwen_vl_lora.yaml)

## 1. PPO 在本项目中优化什么

对一条视觉问答样本，输入是图片和多选题 prompt，策略模型生成一段回答，例如：

```text
<answer>B</answer>
```

将 prompt 记为状态前缀 (x)，生成的 response token 记为：

$$
y=(y_1,y_2,\ldots,y_T)
$$

策略模型给出自回归概率：

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t})
$$

训练目标不是直接对整段文本做普通交叉熵，而是：

1. 根据答案正确性给生成结果一个任务 reward；
2. 根据策略与冻结参考模型的差异加入 KL 惩罚；
3. 用 value head 估计每个 response token 状态的价值；
4. 用 GAE 得到 token 级 advantage；
5. 用 PPO clipped objective 更新策略和 value head。

## 2. 完整调用链

```text
PPO JSONL
  -> Dataset.__getitem__
  -> QwenVLPPOCollator
  -> prompt_inputs + answer_keys
  -> policy.generate() 采样 response
  -> policy/reference teacher-forcing 前向
  -> task reward + KL token reward
  -> GAE advantages + returns
  -> RolloutBatch 暂存到 CPU
  -> 随机划分 PPO minibatch
  -> 当前 policy 再次前向
  -> policy loss + value loss - entropy bonus
  -> backward + gradient clipping + AdamW.step()
  -> 评估、日志、checkpoint
```

入口 `train_ppo.main()` 负责构建数据、模型和 optimizer，并通过 `RLTrainingHooks` 将 PPO 的下列操作交给公共训练循环：

```python
hooks = RLTrainingHooks(
    generate_rollout=...,
    build_minibatch=...,
    compute_losses=...,
    summarize_rollout=...,
    evaluate=...,
    save_checkpoint=...,
)
```

公共循环只管理执行顺序；PPO 数学逻辑仍位于 `algorithms/ppo.py`。

## 3. 输入数据和 collator 输出

### 3.1 单条 JSONL 记录

PPO 数据至少需要包含：

```text
id             样本 ID
messages       Qwen-VL 对话消息，包含一张图片和问题文本
question       问题文本
choice_letter  正确选项 A/B/C/D
ground_truth   原始答案文本
```

### 3.2 DataLoader 的 batch

`QwenVLPPOCollator` 先用 chat template 构造生成 prompt，再由 processor 编码文字和图片。它返回：

```python
{
    'sample_ids': list[int],
    'answer_keys': list[str],
    'questions': list[str],
    'ground_truths': list[str],
    'messages': list,
    'prompt_texts': list[str],
    'prompt_images': list[PIL.Image],
    'prompt_inputs': {
        'input_ids': Tensor[B, P],
        'attention_mask': Tensor[B, P],
        'pixel_values': Tensor[...],
        'image_grid_thw': Tensor[B, 3],
    },
}
```

其中：

- (B) 是 prompt batch size；
- (P) 是左侧 padding 后的 prompt token 长度；
- `pixel_values` 是 processor 生成的视觉 patch 表示，多个样本的 patch 可能连续存放；
- `image_grid_thw` 描述每张图的时间、高度和宽度 patch 网格；
- `answer_keys` 只用于 reward，不会作为监督 token 直接传给模型。

生成任务使用左侧 padding。这样同一 batch 中所有 prompt 的最后一个有效位置对齐，生成 token 都从张量右侧继续追加。

## 4. 三个模型角色

### 4.1 Policy / Actor

`policy.policy_model` 是 Qwen2.5-VL 加可训练 LoRA adapter。它负责：

- 根据 prompt 生成 response；
- 输出每个位置的 vocabulary logits；
- 输出最后一层 hidden states。

基础模型默认以 4-bit 方式加载，optimizer 只接收 `requires_grad=True` 的参数，主要是 LoRA 参数和 value head 参数。

### 4.2 Value head / Critic

`PPOPolicyWithValueHead` 在语言模型最后一层 hidden state 上增加线性层：

$$
V_\phi(s_t)=W_v h_t+b_v
$$

代码为：

```python
self.value_head = nn.Linear(hidden_size, 1)
```

它估计的是状态价值 (V(s_t))，不是动作价值 (Q(s_t,a_t))。在这个自回归任务中，状态 (s_t) 是图片、prompt 和此前已经生成的 token 前缀。

### 4.3 Reference model

参考模型是独立加载、冻结的 SFT 模型：

$$
\pi_{\mathrm{ref}}
$$

它不参与反向传播，用于约束当前策略不要过度偏离 SFT 行为。代码会执行：

```python
parameter.requires_grad_(False)
reference_model.eval()
```

## 5. Rollout 生成

`generate_rollout_batch()` 整体在 `torch.no_grad()` 下执行，因此 rollout 阶段不建立反向传播计算图。

### 5.1 自回归采样

训练时调用：

```python
sequences = policy.generate(**generation_kwargs).sequences
```

默认配置使用：

```text
do_sample=true
temperature=0.7
top_p=0.9
max_new_tokens=16
```

评估时则强制 `do_sample=False`，使用 greedy decoding。

生成后的 `sequences` 形状为：

$$
[B,L],\qquad L=P+T_{\max}
$$

它同时包含 prompt token 和 response token。

### 5.2 Response mask

代码从生成区域开始查找第一个 EOS，并包含该 EOS 位置：

```python
response_attention_mask = build_response_attention_mask(
    sequences[:, prompt_padded_length:],
    eos_token_ids,
)
```

随后构造完整 mask，并向左移一位以对齐 next-token prediction：

```python
response_mask = cat(prompt_zero_mask, response_attention_mask)
shifted_response_mask = response_mask[:, 1:]
```

为什么是 `[:, 1:]`？语言模型在位置 (t-1) 的 logits 用于预测位置 (t) 的 token：

```python
logits = outputs.logits[:, :-1, :]
target_tokens = input_ids[:, 1:]
```

因此 log probability、value、reward、advantage 和 loss 的时间维长度都是 (L-1)。

## 6. 旧策略 log probability 和状态价值

刚完成采样后，用相同策略对固定的完整序列做一次 teacher-forcing 前向：

```python
old_policy_outputs = policy.evaluate_actions(sequences, **model_inputs)
```

得到：

$$
\log\pi_{\mathrm{old}}(a_t\mid s_t)
$$

以及：

$$
V_{\mathrm{old}}(s_t)
$$

这里的 `old` 指“产生当前 rollout 的策略快照”。虽然代码没有复制一份旧 policy 模型，但 rollout 中保存了旧 log probability 和旧 value；后续 optimizer 更新不会改变这些已保存的张量。

`gather_log_probs()` 的计算为：

$$
\log p_t=\log\operatorname{softmax}(z_t)_{y_{t+1}}
$$

也就是只取实际生成 token 对应的 vocabulary log probability，而不是保存整个 vocabulary 分布。

## 7. Reference log probability

冻结参考模型对相同序列进行 teacher-forcing：

```python
ref_outputs = reference_model(input_ids=sequences, ...)
ref_logprobs = gather_log_probs(
    ref_outputs.logits[:, :-1, :],
    sequences[:, 1:],
)
```

得到：

$$
\log\pi_{\mathrm{ref}}(a_t\mid s_t)
$$

它只用于构造 rollout reward，不参与 PPO ratio。

## 8. 任务 reward

response 被解码成字符串后，`score_choice_predictions()` 给每条样本一个序列级 reward：

$$
R_{\mathrm{task}}=
\begin{cases}
1.0, & \text{选项正确}\\
-0.25, & \text{能解析出 A/B/C/D，但答案错误}\\
-0.5, & \text{输出无法解析}
\end{cases}
$$

reward 使用宽松答案解析，以缓解早期训练中的稀疏奖励。例如裸 `A` 仍可能被解析；训练准确率和最终报告另外使用严格的 `<answer>A</answer>` 解析口径。

`scores` 的形状为 `[B]`，每条 response 只有一个任务 reward。

## 9. Token 级 KL reward

PPO 需要 token 级 reward 来计算 GAE。对每个有效 response token，代码定义：

$$
r_t^{\mathrm{KL}}
=-\beta\left(
\log\pi_{\mathrm{old}}(a_t\mid s_t)
-\log\pi_{\mathrm{ref}}(a_t\mid s_t)
\right)
$$

其中默认：

$$
\beta=\texttt{kl\_coef}=0.02
$$

这是一条已采样轨迹上的 log-ratio 惩罚，可视为 KL 的 Monte Carlo 样本估计，不是对整个 vocabulary 精确求和得到的 KL。

任务 reward 只加到最后一个有效 response token：

$$
r_t=
\begin{cases}
r_t^{\mathrm{KL}}, & t<T\\
r_T^{\mathrm{KL}}+R_{\mathrm{task}}, & t=T
\end{cases}
$$

对应代码：

```python
token_rewards = -kl_coef * (old_logprobs - ref_logprobs)
token_rewards[last_valid_position] += score
```

注意，本项目的 PPO 总 loss 中没有再次加入 reference KL loss，因为这项约束已经进入 token reward，随后会通过 advantage 影响策略梯度。

## 10. GAE：从 reward 得到 advantage 和 return

对一条 response，从最后一个有效 token 向前递推。TD residual 为：

$$
\delta_t=r_t+\gamma V_{\mathrm{old}}(s_{t+1})-V_{\mathrm{old}}(s_t)
$$

GAE 为：

$$
A_t=\delta_t+\gamma\lambda A_{t+1}
$$

最后一个 token 的边界条件为：

$$
V(s_{T+1})=0,\qquad A_{T+1}=0
$$

用于训练 value head 的 return 为：

$$
G_t=A_t+V_{\mathrm{old}}(s_t)
$$

默认配置：

```text
gamma=1.0
lam=0.95
```

代码只遍历 `response_mask=True` 的位置，prompt 和 padding 不参与 GAE。

## 11. RolloutBatch 的输入和输出

一次 `generate_rollout_batch()` 的输入是：

```text
policy              当前 Actor-Critic
reference_model     冻结 SFT 模型
processor           Qwen-VL processor
batch               collator 输出
generation_config   采样配置
ppo_config          KL、gamma、lambda 等 PPO 配置
accelerator         设备和分布式运行时
```

输出 `RolloutBatch` 的核心张量为：

| 字段                    | 典型形状     | 含义                                        |
| ----------------------- | ------------ | ------------------------------------------- |
| `sequences`           | `[B, L]`   | prompt + response token                     |
| `full_attention_mask` | `[B, L]`   | 完整 padding mask                           |
| `response_mask`       | `[B, L-1]` | 与 next-token 输出对齐的有效 response mask  |
| `old_logprobs`        | `[B, L-1]` | rollout 策略对实际 token 的 log probability |
| `ref_logprobs`        | `[B, L-1]` | 参考模型 log probability                    |
| `old_values`          | `[B, L-1]` | rollout 时的 value 预测                     |
| `rewards`             | `[B, L-1]` | KL shaping + 最终任务 reward                |
| `advantages`          | `[B, L-1]` | GAE advantage                               |
| `returns`             | `[B, L-1]` | value 回归目标                              |
| `scores`              | `[B]`      | 每条 response 的任务 reward                 |

这些大张量在 rollout 结束时被移动到 CPU。训练每个 minibatch 时，再按需将选中的样本和对应视觉 patch 搬回 GPU，降低 rollout 常驻显存。

## 12. PPO minibatch 和多轮更新

对一个 rollout batch，公共训练循环执行：

```python
for _ in range(ppo_epochs):
    permutation = torch.randperm(rollout_size)
    for indices in minibatches:
        minibatch = build_minibatch(rollout, indices)
        loss_dict = compute_ppo_losses(policy, minibatch, ...)
        backward(loss_dict['loss'])
        optimizer.step()
```

同一批固定 rollout 会被使用 `ppo_epochs` 轮。默认配置为：

```text
per_device_prompt_batch_size=1
per_device_minibatch_size=1
ppo_epochs=2
```

因此默认情况下，每生成一个 response，会对它执行两次 optimizer update。

如果 `whiten_advantages=true`，选中 minibatch 内所有有效 response token 的 advantage 会标准化：

$$
\hat A_t=\frac{A_t-\mu_A}{\sqrt{\sigma_A^2+\epsilon}}
$$

padding 和 prompt 位置保持为 0。当前默认配置为 `false`。

## 13. 当前策略再次前向

在每个 minibatch 中，当前 policy 对固定序列再次 teacher-forcing 前向：

```python
model_outputs = policy(
    input_ids=sequences,
    attention_mask=full_attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
    output_hidden_states=True,
)
```

得到当前参数下的：

$$
\log\pi_\theta(a_t\mid s_t),\quad
V_\phi(s_t),\quad
H(\pi_\theta(\cdot\mid s_t))
$$

随着前面 minibatch 完成 `optimizer.step()`，这里的当前策略会变化；rollout 中保存的 `old_logprobs` 和 `old_values` 始终保持不变。

## 14. PPO 策略损失

重要性采样比率：

$$
\rho_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}
=\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\mathrm{old}}(a_t\mid s_t)
\right)
$$

裁剪后的 PPO 最大化目标是：

$$
J_{\mathrm{clip}}(\theta)
=\mathbb E_t\left[
\min\left(
\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

代码以最小化 loss 的形式实现：

$$
L_{\mathrm{policy}}
=\mathbb E_t\left[
\max\left(
-\rho_t A_t,
-\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t
\right)
\right]
$$

两者数学等价。默认：

$$
\epsilon=\texttt{cliprange}=0.2
$$

`masked_mean()` 保证均值只覆盖有效 response token。

## 15. Value loss

先对 value 的变化做裁剪：

$$
V_t^{\mathrm{clip}}
=V_{\mathrm{old},t}
+\operatorname{clip}\left(
V_{\phi,t}-V_{\mathrm{old},t},
-\epsilon_v,
\epsilon_v
\right)
$$

未裁剪和裁剪误差分别为：

$$
e_{1,t}=(V_{\phi,t}-G_t)^2
$$

$$
e_{2,t}=(V_t^{\mathrm{clip}}-G_t)^2
$$

最终 value loss：

$$
L_{\mathrm{value}}
=\frac12\mathbb E_t[\max(e_{1,t},e_{2,t})]
$$

默认：

$$
\epsilon_v=\texttt{value\_cliprange}=0.2
$$

取两种误差的最大值，可以阻止 value head 通过一次过大的更新轻易绕过裁剪限制。

## 16. Entropy bonus 和总损失

每个位置的策略熵：

$$
H_t=-\sum_{a}\pi_\theta(a\mid s_t)\log\pi_\theta(a\mid s_t)
$$

总损失为：

$$
L
=L_{\mathrm{policy}}
+c_v L_{\mathrm{value}}
-c_H\mathbb E_t[H_t]
$$

代码对应：

```python
total_loss = (
    policy_loss
    + vf_coef * value_loss
    - entropy_coef * entropy
)
```

默认配置：

```text
vf_coef=0.5
entropy_coef=0.0
```

因此当前默认训练实际使用：

$$
L=L_{\mathrm{policy}}+0.5L_{\mathrm{value}}
$$

## 17. 反向传播和参数更新

每个 minibatch 执行：

```python
optimizer.zero_grad(set_to_none=True)
accelerator.backward(loss)
accelerator.clip_grad_norm_(policy.parameters(), max_grad_norm)
optimizer.step()
```

optimizer 是 AdamW。默认主要参数为：

```text
learning_rate=5e-6
adam_beta1=0.9
adam_beta2=0.95
max_grad_norm=1.0
```

参考模型没有进入 optimizer；policy 中只有可训练 LoRA 和 value head 等 `requires_grad=True` 参数进入 optimizer。

## 18. Step、epoch 和 optimizer step 的区别

本项目的一个 `global_step` 表示“完成一个 rollout batch 的所有 PPO 更新”，不是一次 `optimizer.step()`。

设：

- rollout 数量为 (B)；
- minibatch size 为 (M)；
- PPO 更新轮数为 (K)。

那么一个 `global_step` 内的 optimizer step 数约为：

$$
K\left\lceil\frac{B}{M}\right\rceil
$$

默认 (B=1,M=1,K=2)，所以：

$$
1\ \text{global step}=2\ \text{optimizer steps}
$$

配置中的 `logging_steps`、`eval_steps`、`save_steps` 都以这个 `global_step` 为单位。

## 19. 训练指标

### 19.1 Rollout 指标

- `reward_mean`：宽松 reward 的均值；
- `accuracy`：严格 `<answer>...</answer>` 格式下的正确率；
- `valid_option_rate`：严格格式下能解析出合法选项的比例；
- `response_length_mean`：有效 response token 平均长度；
- `kl_mean`：rollout 策略与参考模型在已采样 token 上的平均 log-ratio。

### 19.2 PPO update 指标

- `policy_loss`：PPO clipped policy loss；
- `value_loss`：clipped value loss；
- `entropy`：有效 response token 的平均策略熵；
- `clipfrac`：满足 $|\rho_t-1|>\epsilon$ 的 token 比例；
- `approx_kl`：当前策略相对 rollout 旧策略的近似更新幅度。

代码中的：

$$
\mathrm{approx\_kl}
=\frac12\mathbb E_t\left[
(\log\pi_\theta-\log\pi_{\mathrm{old}})^2
\right]
$$

要特别区分两个概念：

1. rollout 的 `kl_mean` 比较 `old policy` 与 `reference policy`，用于观察偏离 SFT 的程度；
2. loss 的 `approx_kl` 比较 `current policy` 与产生 rollout 的 `old policy`，用于观察本轮 PPO 更新幅度。

## 20. 评估流程

验证集评估复用 rollout 代码，但有以下区别：

- `do_sample=False`，使用 greedy decoding；
- 使用 `eval_max_new_tokens`；
- 不执行 minibatch loss、反向传播或 optimizer step；
- 多进程指标通过 `accelerator.reduce(..., reduction='sum')` 汇总。

最终测试还会逐样本生成 JSONL 和 HTML 报告。

## 21. Checkpoint 输出

一次 PPO checkpoint 包括：

```text
checkpoint-N/
  adapter/             LoRA adapter
  value_head.pt        Critic value head
  optimizer.pt         AdamW 状态
  training_state.json  global step 等信息
  rng_state.pt         Python/NumPy/PyTorch 随机状态
  metadata.json        模型元数据
```

恢复训练时：

1. policy 从 checkpoint adapter 加载；
2. value head 从 `value_head.pt` 加载；
3. optimizer 和随机状态恢复；
4. reference model仍从配置的 SFT adapter 加载并保持冻结；
5. `global_step` 从 checkpoint 继续累计。

`save_total_limit` 控制最多保留多少个最新 checkpoint。

## 22. 一次训练迭代的公式汇总

对 rollout 中的每个有效 response token：

$$
\begin{aligned}
r_t^{\mathrm{KL}}
&=-\beta(\log\pi_{\mathrm{old},t}-\log\pi_{\mathrm{ref},t})\\
r_T
&=r_T^{\mathrm{KL}}+R_{\mathrm{task}}\\
\delta_t
&=r_t+\gamma V_{\mathrm{old},t+1}-V_{\mathrm{old},t}\\
A_t
&=\delta_t+\gamma\lambda A_{t+1}\\
G_t
&=A_t+V_{\mathrm{old},t}\\
\rho_t
&=\exp(\log\pi_{\theta,t}-\log\pi_{\mathrm{old},t})\\
L_{\mathrm{policy}}
&=-\mathbb E_t[\min(\rho_tA_t,\operatorname{clip}(\rho_t)A_t)]\\
L_{\mathrm{value}}
&=\frac12\mathbb E_t[\max((V_t-G_t)^2,(V_t^{\mathrm{clip}}-G_t)^2)]\\
L
&=L_{\mathrm{policy}}+c_vL_{\mathrm{value}}-c_HH
\end{aligned}
$$

然后对 (L) 反向传播，裁剪梯度并更新可训练的 policy LoRA 参数和 value head 参数。

## 23. 最小化理解

如果只记住主线，可以概括为：

```text
当前策略生成答案
  -> 答对得正奖励，答错/无效得负奖励
  -> 偏离冻结 SFT 参考模型会受到 token 级惩罚
  -> value head 估计每个生成位置未来还能得到多少奖励
  -> GAE 判断每个实际生成 token 比预期好还是差
  -> PPO ratio clipping 限制策略一次不要改太多
  -> value clipping 限制 critic 一次不要改太多
  -> 同一批 rollout 做若干轮 minibatch 更新后，再重新生成下一批 rollout
```
