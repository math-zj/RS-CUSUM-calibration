# RS-CUSUM-RL Phase 2 预冻结仿真方案

冻结日期：2026-08-28  
Phase 1 方法版本：`1.0.1-phase1`  
Phase 2 protocol：`1.0.0-phase2-pilot`  
真实 OhioT1DM gate：`CLOSED`

本文件和 `configs/rs_cusum_phase2_pilot.yaml` 在任何 Phase 2 multi-repetition simulation 之前写成。Phase 2 只使用 synthetic trajectories；不读取真实 OhioT1DM reward、Q、statistic、p-value 或 changepoint。若仿真显示 Phase 1 的 frozen estimator/statistic/support/scaling/bootstrap 存在结构性失败，立即标记 `PHASE_2_METHOD_REVISION_REQUIRED`，不在原版本下静默修改。

## 1. 不可变的父方法

以下全部继承 Phase 1，不因仿真结果调整：5 分钟 elapsed clock、240 分钟初始历史、120 分钟 post-gap burn-in、原 patient cluster、patient-balanced weight、Bellman fixed-point `W`、`Delta_i`、primary support、harmonic-active scaling、Gaussian patient-cluster multiplier。父配置 SHA-256 已写入 Phase 2 YAML。

## 2. Synthetic MDP

主设计为 N=12、state dimension 21、240 个 nominal transitions；不等长设计最大 360。state 是稳定线性 Markov system：第 1 维为 glucose-like AR state，第 2 维为观测到的、患者固定 phenotype，其余为稳定 AR covariates。行为策略 `P(A=1|S)` 是 logistic state-dependent policy。null 下 reward 和 transition conditional law 不随 elapsed time 改变；observation process 可以改变，但不允许依赖未来 state/reward。

初始 48 条 transitions 用于 240 分钟历史，不进入分析。RS gap 后必须先有 24 条连续 observed transitions 才 re-enter；elapsed index 不压缩。Strict benchmark 在初始 history 后遇首个 invalid transition即永久结束。

Alternative 的真实 change 位于 post-history analysis window 的固定 50%；candidate rule 先于 alternative 定义冻结，不因 true CP 增密。一般 effect levels 为 0.25/0.60/1.20；rare-A=1 interaction 为 0.50/1.00/2.00。

## 3. Outcome-blind evaluation grid

每个 replicate 只从 `(patient_id, elapsed_index, current state)` 构造 grid。患者按 ID 排序；患者内先按 elapsed 排序，再用固定 seed 9173 的 patient-specific deterministic permutation；随后 round-robin 每患者取一个，直到 K states 或无可用 state。每个 state 同时配 A=0 与 A=1。

- small：16 states / 32 evaluation points；
- primary：32 states / 64 evaluation points；
- large：64 states / 128 evaluation points。

主规则在任何 power 结果以前固定为 32。不得读取 reward、Q contrast、statistic、p-value 或 changepoint。未来真实数据若采用该规则，实际 state grid 还必须在访问真实 reward/Q 前另行 freeze/hash。

## 4. Outcome-blind candidate grid

分析窗口 `[a,b)` 的 raw candidates 固定为 post-history elapsed length 的 20%、30%、40%、50%、60%、70%、80%，使用 nonnegative round-half-up 映射为整数并去重。support screen 只在该 raw set 上产生 `U_adm`；不根据峰值增密，不根据 true CP 或 rejection 调整。

## 5. 方法与 statistic

主方法 `RS-full`：dynamic re-entry + primary support + patient-balanced FQI + harmonic-active scaling + Gaussian patient-cluster multiplier。主 statistic 为 fixed evaluation grid 和 fixed `U_adm` 上的 normalized maximum。

有限 bootstrap p-value：

`p=(1+#{T*_b >= T_obs})/(B+1)`。

不允许 p=0。空 support、zero variance、FQI failure、bootstrap failure 分别记录显式 status，不能转换为 non-rejection。

比较方法：

- `original_rectangular_compatible`：只用于 complete data；normalized pooled FQI、original `b0`、transition-level Gaussian multiplier。它是 Original-compatible estimator/statistic benchmark，不冒充未经修改的旧 sklearn API；
- `strict_first_gap_termination`：首 gap 后 permanent termination，同 patient identity，不称为官方 Appendix A.2 implementation；
- `RS-with-pooled-weight`：dynamic panel/support 不变，但 FQI 用 pooled transition weight，bootstrap 仍按患者 cluster；
- `RS-no-support-screen`：所有至少左右各有数据的 raw candidates，仅为 simulation diagnostic，永不用于真实分析。

## 6. Null scenarios

冻结 N0 complete、N1 monotone dropout、N2 intermittent gaps/re-entry、N3 40%/67%/100% unequal lengths、N4 action prevalence 0.15/0.08/0.035、N5 transition/action patient dominance、N6 missing+rare+unequal、N7 phenotype-selective composition shift、N8 current-state/action-dependent mild/strong informative missingness。所有 null 的 reward/transition conditional law stationary。

N7 的目的不是宣称 missingness ignorable，而是直接压力测试：即使 conditional MDP 不变，penalized approximate FQI 是否会因 active composition/state support 改变而产生伪 Q contrast。N8 missing-start hazard 只能访问当前 state/action，不能访问未来。

## 7. Alternatives

冻结 A0 complete general change、A1 intermittent general change、A2 low-prevalence A=1 interaction change、A3 unequal/dominance general change。每个使用 weak/moderate/strong 三档。报告全部 testable replicates 的 CP error，并另报 detection-conditioned error，不能只保留成功定位。

## 8. Sensitivity 与 ablation

Primary 永远保持 primary support、harmonic scaling、primary grid。lenient/strict support，both-side/count-ESS scaling，small/large grid 和 B=199/999 都只作标记清楚的 sensitivity，不得按 size/power 最好者更换 primary。

预指定 ablation：N2/N3/N5/N6/A3 比较 patient-balanced 与 pooled；N4/N5/N6/A2 比较 support 与 no-support；N1/N2/N6/A1 比较 strict termination 与 RS re-entry。Original 仅在 N0/A0 complete rectangular 中比较。

## 9. 两阶段执行

Stage A：所有核心 null/alternative families 每项 R=20、B=199，验证无 crash/NaN silent propagation、status 完整、p-value 范围、CP 仅属于 `U_adm`、seed deterministic 和 cluster identity。Stage A 未通过时禁止 Stage B。

Stage B：每个 null variant R=100、B=499；A0–A3 每个 effect level R=100、B=499。Phase 2 只作 small pilot，不把 R=100 解释成发表级精度。

## 10. Summary denominators 与 gate

alpha 取 0.01/0.05/0.10，主 gate 看 0.05。每个 null 同时报 all-generated rejection（NOT_TESTABLE 计入总数但明确列出）和 conditional-on-testable rejection，另报 Monte Carlo SE、Wilson interval、`U_adm`、support reason、active patients、A=1、dominance 和 numerical failures。

任一核心 null 在 0.05 conditional size >0.15、numerical failures >5%、complete null 失控、composition/informative missingness 极端 inflation、不可复现、CP 不在 `U_adm` 或 cluster identity 破坏，均不得 PASS。0.08–0.15 是 yellow flag，需要 Phase 3 加大重复。即使 Phase 2 通过，也只允许 `PHASE_3_FULL_SIMULATION`，不允许正式 OhioT1DM analysis。
