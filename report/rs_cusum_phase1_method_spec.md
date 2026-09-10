# RS-CUSUM-RL Phase 1 冻结方法规范

冻结日期：2026-08-28  
状态：`phase1_method_frozen`，当前版本 `1.0.1-phase1`  
上游 5 分钟 MDP：`configs/mdp_5min_protocol.yaml`，SHA-256 `D0D4C9D19DEE704E4DF81A51F8B20BEC272494D6825E2DD16C0E1772F04F6B2C`。

本规范写于任何正式 multiplier bootstrap、p-value、changepoint 或 OhioT1DM RS 结果之前。Phase 1 只允许模块实现、确定性单元测试和小型合成 degeneration diagnostic。`1.0.1-phase1` 在 outcome-blind 代码自审中把 influence matrix 从单步 Ridge Hessian 修正为 Bellman 固定点 Jacobian；依据是原代码 `W_mat` 中的 next-greedy 项和对带 Ridge estimating equation 的直接求导，不依据任何显著性或结果优劣。

## 1. 风险集面板与时钟

对患者 `i` 的 clock-specific recording start 记为 `c_i`，真实决策边界为 `tau_it`。定义

`e_it = round((tau_it - c_i) / 5 min)`。

非负恰好半格采用向上舍入。必须保存舍入前后的 residual；绝对 residual 超过 0.5 分钟或同一患者出现重复 `(clock_type, e_it)` 时硬失败。时间轴绝不因缺失而压缩。每条记录保存原始 `patient_id`，segment 不是独立受试者，也不是 bootstrap cluster。

三种时钟分别分析：

- `training`：`c_i` 是该患者 training 首时间；
- `testing_reset`：`c_i` 是 testing 首时间；
- `testing_continuous`：`c_i` 是 training 首时间，且 state 必须由合并后的 training+testing 事件流重建；
- 三类时钟不得混在同一风险集分析中。

主 re-entry 规则要求最初记录已有 240 分钟真实历史；每次真实 gap 后，至少先出现连续 120 分钟、24 条合法 5 分钟 CGM edges，才重新纳入 transition。状态仍可使用此前 240 分钟内真实记录，但不插值。敏感性规则 `allow_sparse_real_history_after_gap` 允许在 gap 后首条合法 local edge 重新进入，但仍只用真实既往记录。两种规则都保持同一个 patient cluster。

## 2. Support screen（严格 outcome-blind）

候选切点 `u` 将分析窗口 `[a,b)` 分为

- 左侧 `L(u)={it: a <= e_it < u}`；
- 右侧 `R(u)={it: u <= e_it < b}`。

Support 函数只接收 `(patient_id, elapsed_index, action)` 以及冻结的参数维度，不能访问 reward、Q、Bellman residual、statistic、p-value 或 propensity/IPW。每侧报告 active patients、A=0/A=1 counts、各 action 的 unique patients、两侧均出现的 patients、每患者 action-specific/total shares、最大 dominance 与 count ESS。count ESS 仅为描述：`ESS=1/sum_i share_i^2`。

每个 action-specific linear Q block 有 `d_a=21+1=22` 个参数。每侧每 action 的 transition 下限为

`m = max(absolute_floor, multiplier * d_a)`。

冻结三档为：lenient `m=44`，primary `m=66`，strict `m=110`。其它 patient/support/dominance 条件以 `configs/rs_cusum_sensitivity.yaml` 为准，所有边界均按包含等号判定。若没有 admissible `u`，唯一结果是 `NOT_TESTABLE_UNDER_SUPPORT_RULE`；不得给 `p=1`，不得选 fallback 切点。

## 3. Patient-balanced FQI

令 `I_h` 为侧 `h in {L,R}` 有 transition 的患者集合，`n_ih` 为患者 `i` 在该侧 transition 数。action-specific feature map 为

`x_it = phi(S_it,A_it) = [(1,S_it) I(A=0), (1,S_it) I(A=1)]`。

给定上一步系数 `beta^(k-1)`，Bellman target 为

`Y_it^(k) = R_it + gamma max_a phi(S_i,t+1,a)' beta^(k-1)`，

其中 `gamma=0.9825931938526898`。更新严格为

`beta_h^(k) = argmin_beta (1/|I_h|) sum_i (1/n_ih) sum_t (Y_it^(k)-x_it'beta)^2 + alpha ||beta||_2^2`，

`alpha=1`，截距也被惩罚。等价 transition weight 为 `1/(|I_h| n_ih)`。使用全部合法 transitions，不对 A=0 下采样。最大 800 次，连续系数 L2 差不超过 `1e-5` 时收敛；线性系统失败必须显式报错，不允许 broad-except fallback。

注意：这里的 Ridge 目标按总权重 1 归一化；degeneration benchmark 也必须使用相同归一化和相同 `alpha`，不能直接把未归一化 SSE 的旧 sklearn Ridge 数值混为一谈。

## 4. Patient-level score、influence 与左右对比

FQI 收敛后定义 TD residual

`delta_it,h = R_it + gamma max_a Q_h(S_i,t+1,a) - x_it'beta_h`。

患者平均 Bellman score：

`g_ih = (1/n_ih) sum_t x_it delta_it,h`，

侧内中心化：

`gtilde_ih = g_ih - |I_h|^{-1} sum_j g_jh`。

令 `a*_i,t+1=argmax_a Q_h(S_i,t+1,a)`（动作并列时按固定的最小 action index），并记 `x*_i,t+1=phi(S_i,t+1,a*_i,t+1)`。FQI 固定点 estimating equation 对 `beta` 的负 Jacobian（influence `W` matrix）为：

`W_h = |I_h|^{-1} sum_i n_ih^{-1} sum_t x_it (x_it - gamma x*_i,t+1)' + alpha I`，

患者 influence direction：`eta_ih = W_h^{-1} gtilde_ih`。这里 `+alpha I` 来自归一化 Ridge estimating equation中的 `-alpha beta`；省略 next-greedy 项只对应每一次固定 target 回归，不对应最终 FQI fixed point，因此不采用。

对固定 evaluation point `z=(s,a)`，观察到的 Q contrast 是

`D_u(z)=phi(z)'(beta_L-beta_R)`。

所有在任一侧出现的原始患者都参与 contrast。患者只在一侧出现时，另一侧 contribution 定义为零；segment 不单列。患者 contrast influence 为

`Delta_iu(z) = phi(z)'[ I(i in I_L) eta_iL/|I_L| - I(i in I_R) eta_iR/|I_R| ]`。

这样同一患者跨左右两侧的 covariance 保留在同一个 cluster contribution 中。令 `G_u=|I_L union I_R|`，cluster variance 为

`V_u(z) = [G_u/(G_u-1)] sum_i Delta_iu(z)^2`，`G_u<=1` 或 `V=0` 时该点不可 studentize，禁止除零。

## 5. 三种 boundary scaling

令左、右 elapsed 长度为 `ell_L=u-a`、`ell_R=b-u`，原矩形时间因子为

`b0(u)=sqrt(ell_L ell_R/(ell_L+ell_R))`。

令 `G` 为整个分析窗口原始患者数，`G_L,G_R` 为两侧 active patient 数，`G_B` 为两侧均出现的患者数。令 `E_L,E_R` 为按患者 raw transition count share 算出的 count ESS。冻结候选为：

- A `both_side_patients`：`b_A=b0 sqrt(G_B/G)`；
- B `harmonic_active_patients`：`G_H=2G_LG_R/(G_L+G_R)`，`b_B=b0 sqrt(G_H/G)`；
- C `count_effective_risk`：`E_H=2E_LE_R/(E_L+E_R)`，`b_C=b0 sqrt(E_H/G)`。

主分析冻结为 B。A 与 C 只作敏感性。理由是 B 在矩形完整面板上还原 `b0`，并对左右 active patients 不对称作 outcome-blind 折减，同时不会把 transition dominance 重复当作患者信息；dominance 已由 support screen 单独控制。Phase 1 只在 toy 数据比较三者数值行为。

## 6. 三类统计量

evaluation grid `Z` 必须在 outcome analysis 前另行冻结；Phase 1 只接受显式 toy grid。对 admissible `u`：

- normalized maximum：`T_N(u)=b(u) max_z |D_u(z)|/sqrt(V_u(z))`；
- unnormalized maximum：`T_U(u)=b(u) max_z |D_u(z)|`；
- L1/integral：`T_I(u)=b(u) |Z|^{-1} sum_z |D_u(z)|`。

整体统计量是固定 admissible candidate set 上相应 `T(u)` 的 maximum。空 candidate set 不计算上述量。

## 7. Patient-cluster multiplier null

只实现 null replicate，不在 Phase 1 计算 p-value。对每个原始患者一次抽取 `xi_i ~ N(0,1)`；该 multiplier 在患者全部时间、全部 re-entry segments 和全部候选 `u` 上复用。固定 support screen 得到的 `U_adm`，不得在 bootstrap draw 内重筛。

固定 `u,z` 的 multiplier contrast 为

`D_u^*(z)=sum_i xi_i Delta_iu(z)`。

bootstrap 的 normalized、unnormalized、integral 版本分别以 `D_u^*` 替换 `D_u`，使用与 observed statistic 相同的 `b(u)`、固定 observed `V_u(z)`、固定 evaluation grid，并在完全相同的 `U_adm` 上取 maximum。Phase 1 不声称该 adaptation 已有原论文的现成理论保证。

## 8. 矩形 degeneration 要求

当所有患者完整、相同 `T`、每侧每患者 transition 数相同、feature map/gamma/alpha/evaluation grid 相同时：

1. risk-set panel 等于矩形面板；
2. patient-balanced weights 全部等于 `1/(NT)`，故与归一化 pooled Ridge FQI 完全一致；
3. `G_B=G_L=G_R=E_L=E_R=G`，三种 boundary scaling 都等于 `b0`；
4. 对应 unnormalized Q-contrast statistic 应在数值容差内等于矩形 benchmark。

若这四项任一失败，Phase 1 gate 必须为 FAIL。

## 9. 当前证据边界

本规范不证明 patient-cluster bootstrap 的渐近有效性，也不解决 N=12 的弱 cluster inference、观察性 action confounding、rare A=1 extrapolation 或 evaluation-grid 选择。它仅冻结一个可审计实现，并要求这些风险在 Phase 2 simulation 和后续 gate 中被显式检验。
