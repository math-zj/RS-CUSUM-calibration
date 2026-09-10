# RS-CUSUM-RL Phase 2R-A：Null Failure Localization 预冻结方案

冻结日期：2026-08-29  
诊断标签：`phase2r-a-forensics`  
父方法：`1.0.1-phase1`（只读）  
起始 gate：`PHASE_2_METHOD_REVISION_REQUIRED`  
OhioT1DM gate：`CLOSED`

## 1. 目的与边界

本阶段只定位 complete balanced stationary null（N0）中 Type-I error=0.70 的来源，不寻找“表现最好”的修复。禁止运行 missingness、re-entry、rare-action、alternative、power、changepoint accuracy 或真实 OhioT1DM。任何 diagnostic variant 都不构成新 primary method；Phase 1/2 的代码、配置和结果保持只读。

唯一数据生成机制是冻结的 Phase 2 N0：N=12、T=240、前48步作为历史、21维 state、完整等长轨迹、充分的二元 action support、stationary reward/transition law。主 FQI 使用原 feature map、`gamma=0.9825931938526898`、normalized Ridge `alpha=1`。

## 2. 固定 point、grid 与 candidate

单候选固定为分析区间 `[48,240)` 的中点 `u=144`。固定 evaluation grid 由独立 N0 reference dataset（seed 2026082902）的 state-only panel，按 Phase 2 的患者平衡 deterministic rule 和 seed 9173 选32个 state，再为 action 0/1 各复制一次，形成64点。构造过程不读取 reward、Q、statistic、p-value 或 changepoint；实际数组在首个 multi-repetition run 前保存并哈希。D0/D1/D3 的单点固定为该 grid 的第0点（action=0）。

多候选沿用 outcome-blind 20%–80% lattice：`86,106,125,144,163,182,202`。D4 使用 primary support 和 harmonic-active-patients scaling；N0 中全部患者完整，三种 Phase 1 scaling 均退化为原 time factor。

## 3. D0→D4 factorial decomposition

- D0：单候选、单点、`|D_u(z)|`，不 studentize、不乘 boundary、不取任何 maximum。
- D1：在 D0 上除以固定 observed `sqrt(V_hat)`；bootstrap 同样使用该 observed denominator。
- D2：单候选、64点固定 grid 上取 studentized `max_z`。
- D3：单固定点、七候选，使用 harmonic boundary 后取 `max_u`。
- D4：恢复 Phase 1 normalized maximum：64点、七候选、primary support、harmonic boundary、`max_z,max_u`。

先用1000个独立 N0 dataset 建立 D0 sampling truth。D0/D1 calibration 使用这1000次；D2–D4 与 Original-compatible decomposition 使用预定的400次和 B=399。不能因中间结果改变 grid、candidate 或重复数。

## 4. Sampling truth、variance 与 studentization

每个 truth replicate 保存左右 beta、单点 D、V、sqrt(V)、patient scores、eta、W、左右 design matrices及 greedy action frequency。设计矩阵以 float32 二进制矩阵保存，标量和低维数组另存；不以摘要代替这些 forensic 对象。

报告 `Var_MC(D)`、`SD_MC(D)`、mean/median V_hat、`sqrt(mean(V_hat))/SD_MC` 及 Monte Carlo 95%区间。D1 另报告 Z 的中心、SD、MAD、偏度、峰度和2.5/5/50/95/97.5分位数。Oracle q90/q95/q99 只用于 synthetic forensic benchmark，绝不用于真实数据。

## 5. Linearization 与 W audit

用独立 N=200、T=960 的 stationary reference sample拟合 alpha=1 的 numerical beta0。对每个小样本左右侧，在 beta0 计算 patient estimating equation，并以 population-reference active-set W0 构造 first-order error：

`beta_hat-beta0 ≈ W0^{-1}[mean_i g_i(beta0)-alpha*beta0]`。

左右向量及左右 contrast 均报告 correlation、slope、RMSE、relative RMSE、variance ratio和sign agreement。

对前100个 truth replicates直接比较 analytical Bellman fixed-point W 与 estimating equation `F(beta)` 的 central finite-difference `-J_numeric`，步长固定 `1e-6`。报告最大绝对差、Frobenius相对误差和奇异值差；不把 Ridge Hessian误称为 fixed-point Jacobian。

## 6. Greedy non-smoothness

在每侧记录 next-state `|Q1-Q0|` 小于 `1e-6...1e-2` 的比例。用预定 seed 产生8个固定单位 Rademacher方向，在 epsilon=`1e-6,1e-5,1e-4,1e-3` 的正负扰动下记录 greedy action 平均/最大切换比例。该诊断只检验 fixed-active-set linearization 的局部稳定性。

## 7. Alpha、N 与 T scaling

Alpha diagnostic固定为0、0.01、0.1、1、10，每档R=300、B=399；只定位 penalty 是否系统性改变 bias、variance ratio、studentized SD、bootstrap size、condition number和Q magnitude。

N-scaling固定T=240，N=12/24/50/100，每档R=300、B=399。T-scaling固定N=12，T=120/240/480/960，每档R=300、B=399。N与T分开改变，以区分 few-cluster approximation 与单患者轨迹长度。

## 8. Bootstrap forensic variants

对预定50个 truth replicates，各用B=1999，保存全部 draws。比较 Gaussian、Rademacher和 Webb six-point；Webb六个等概率值固定为 `±sqrt(3/2), ±1, ±sqrt(1/2)`。比较 bootstrap SD、分位数、p-value离散性和对 Monte Carlo truth 的距离，不能仅比较 rejection rate。

Variance variants 均为 `DIAGNOSTIC_ONLY`：

- B0：Phase 1 fixed observed HC1 cluster variance；
- B1：每个 draw 对 multiplier-weighted patient contributions 重新中心化并重算 HC1 variance；
- B2：逐患者 leave-one-out refit 构造 jackknife pseudo-contributions和jackknife variance；
- B3：标量 cluster-leverage `h_i=Delta_i²/sum Delta²`，使用 `Delta_i/sqrt(1-h_i)` 后重中心化的 adapted CR2-inspired analogue。B3没有继承线性回归标准CR2定理。

## 9. Restricted-null full-refit prototype

只有前序诊断完成后才运行30个dataset、每个B=199。先在完整分析窗拟合 common stationary Q。固定原 states/actions，用同一 patient multiplier 扰动该患者整条轨迹的 centered TD residual，并定义：

先在完整窗口的 patient-balanced Gram matrix 上解

`(X'WX) v_ridge = alpha beta_common`，

再将每名患者的 common-model TD residual 减去该患者自己的 scalar mean，并定义

`R*_it = x_it beta_common - gamma max_a Q_common(S_next,a) + x_it v_ridge + xi_i delta_centered_it`。

`x_it v_ridge` 使未扰动的完整窗口 estimating equation包含与 frozen Ridge 相同的 `alpha beta_common` 项；它不保证每个有限左右子样本恰好得到同一 beta，而是保证生成机制在左右使用同一 common-Q null 基线。

每个 draw 在 imposed common-Q null 下重新拟合左右 FQI，重新计算 beta contrast和HC1 variance。它是固定设计、patient-cluster residual wild、restricted-null common-Q prototype，只用于判断“不重估FQI”是否是主要失效层；无论结果如何都不能直接成为 primary。

## 10. Original-compatible benchmark

完整矩形 N0 另用 normalized pooled FQI及 transition-level Gaussian multiplier，R=400、B=399，执行同一 D0–D4 decomposition。它只回答 Phase 0/Original-compatible inference 在此 synthetic DGP 是否也失控，不用于宣布原论文一般无效。

## 11. 输出与 gate

所有预定结果同时保存 replicate-level和summary。最终 gate 只能是 `ROOT_CAUSE_LOCALIZED`、`ROOT_CAUSE_PARTIALLY_LOCALIZED` 或 `FORENSIC_IMPLEMENTATION_FAILURE`。即使某个 prototype接近oracle，也必须另写 Phase 2R-B protocol并作为新版本独立确认。Phase 3和正式OhioT1DM始终禁止。
