# RS-CUSUM-RL Phase 2R-C Protocol

冻结日期：2026-08-29  
父方法：`1.0.1-phase1`  
进入状态：`INFERENCE_REVISION_PARTIAL`  
1.0.2：未建立；Phase 3与OhioT1DM：`CLOSED`

## 1. 目标和顺序

本阶段首先定位M2的fixed-design conditional mean、covariance和tail。只有C0 gate支持design-variation机制，才实现M3 whole-patient exchangeably weighted full-refit。C0之前不实现M3；C0通过后再作第二次implementation freeze。任何阶段均不运行alternative、power、changepoint、transfer-null或OhioT1DM。

父方法的risk set、120 min re-entry burn-in、primary support、patient-balanced FQI、gamma、alpha=1、Bellman W/Delta、observed V、harmonic scaling、64点grid、七候选与observed statistic全部只读。

所有C0/C1/C2/C3及nested inner seeds由独立master在64-bit空间确定性生成，在C0前一次冻结，并显式检查Phase 2R-A/B零重叠与本阶段内部零重复。

## 2. C0 conditional-mean分解

Fresh N0使用R=300、B=999，比较M0/M1/M2。对raw Q contrast和studentized process保存每个dataset、candidate与point的conditional mean。令`X*`为signed bootstrap process、`mu*=mean_b X*_b`，同时分析：

- uncentered root second moment `sqrt(E_b X*²)`；
- centered conditional covariance `Cov_b(X*-mu*)`；
- uncentered与forensic-centered D2/D4 q90/q95/q99和p-value；
- candidate/state coordinate mean pattern。

Empirical mean centering只作forensic，不自动成为方法。

## 3. M2 zero-perturbation center

每个dataset在同一common-Q、同一observed design、multiplier恒零下，对七个candidate分别重新拟合左右FQI，得到

`D_null0(u,z)=Q_L^{null,zero}(z)-Q_R^{null,zero}(z)`。

比较其与M2 raw conditional mean的correlation、through-origin slope、RMSE、relative RMSE、candidate/state pattern。Evidence A预冻结为：M2 conditional-mean RMS/truth SD≥0.25、绝对correlation≥0.75、slope 0.60–1.40且relative RMSE≤0.70。

## 4. Nested design/reward decomposition

代码审计确认N0 state/action过程不依赖reward；reward innovation是在给定state/action后独立生成的`N(0,0.6²)`。因此固定outer dataset的states/actions并只重生reward innovation，严格等于冻结DGP下的conditional reward law。

使用100 outer designs×100 inner rewards。对七候选×64点signed studentized process估计

`Cov_total = E Cov(X|design) + Cov(E[X|design])`，

并报告pointwise variance、leading eigenvalue、Frobenius identity error及total/within-centered/design-mean的D2/D4 oracle quantiles。Evidence B要求within pointwise variance fraction≤0.80或within leading-eigen fraction≤0.75，且分解Frobenius relative error≤0.05。

## 5. Patient-level decomposition

M1使用observed whole-patient influence process。M2使用Gaussian multiplier到full-refit process的patient-level least-squares projection。比较每名患者process norm与phenotype、state-support spread、action frequency、candidate-specific action composition及patient empirical Gram差异。Evidence C要求M2/M1 mean norm≤0.80，且至少两个design association相对衰减≥0.15。

任一A/B/C成立，C0 gate为`DESIGN_VARIATION_HYPOTHESIS_SUPPORTED`并允许实现M3；否则最终为`DESIGN_VARIATION_HYPOTHESIS_NOT_SUPPORTED`。

## 6. M3预定义（仅C0通过后实现）

主weights为`Y_i~Exp(1)`、`w_i=Y_i/mean(Y)`。side内patient objective weight为`omega_ih=w_i/sum_{j in I_h}w_j`，transition weight为`omega_ih/n_ih`。原始patient trajectory的state/action/reward/next-state整体共同重加权，不生成pseudo reward、不重筛support，每draw完整迭代左右FQI并重算W、patient influence和V。

Bootstrap process严格使用centered estimator increment：

`D_inc*(u,z)=[Q_L*(z)-Q_R*(z)]-[Q_L(z)-Q_R(z)]`。

全部weights=1时必须数值复现observed FQI且`D_inc*=0`到machine tolerance。Observed statistic仍使用父方法observed V。

M3主law为EXP；若EXP通过C2，才在相同C2 datasets运行patient multinomial counts。`GAMMA4`在本protocol中预先选择不运行。Gaussian/Rademacher/Webb不得作为非负objective weights。

## 7. M3 gates

C1：R=20、B=199，仅工程gate。C2：fresh R=300、B=499；M3继续要求D0≤0.10、D2≤0.12、D4≤0.12、D4 q95 ratio 0.88–1.12、conditional mean RMS/truth SD≤0.20、pointwise SD ratio 0.85–1.15、leading eigen ratio 0.75–1.25、failure≤1%。

C3 seeds在C2以前生成并哈希；若C2通过，locked R=500、B=999。C3要求D4≤0.10、q95 ratio 0.90–1.10、conditional mean ratio≤0.20、无near-zero pile-up、failure≤1%，并优先要求SD ratio 0.90–1.10、leading eigen ratio 0.80–1.20。

即使C3通过，本阶段也只给`WHOLE_CLUSTER_INFERENCE_CANDIDATE_CONFIRMED_ON_N0`，不建立1.0.2；transfer-null必须在独立Phase 2R-D重新预注册。
