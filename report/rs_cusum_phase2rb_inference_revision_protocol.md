# RS-CUSUM-RL Phase 2R-B：Inference Revision Confirmatory Protocol

冻结日期：2026-08-29  
父方法：`1.0.1-phase1`  
父gate：`ROOT_CAUSE_LOCALIZED`  
候选范围：`1.0.2 inference-only revision`，尚未建立  
Phase 3：`CLOSED`；OhioT1DM：`CLOSED`

## 1. 目的与严格边界

本阶段只确认能否修复 Phase 2R-A 已定位的 joint evaluation-state × candidate multiplier inference failure。Observed risk set、re-entry、primary support、patient-balanced FQI、gamma、alpha=1、Bellman fixed-point W、Delta、harmonic scaling、64点evaluation grid、七候选和observed statistic全部保持不变。

仅运行fresh stationary null；在N0 holdout通过后才允许预注册transfer null。禁止alternative、power、changepoint performance和OhioT1DM。Phase 2R-A结果只用于冻结候选和negative-control常数，不进入本轮任何rejection-rate denominator。

RB0、RB1、RB2和transfer的seed lists在RB1以前一次生成并哈希；master seed为2026083001，与Phase 2R-A不同。RB2结果在RB1 gate前不会读取。

## 2. 三个正式比较臂与negative control

### M0 frozen reference

原Phase 1 Gaussian patient multiplier、fixed influence contributions、fixed observed HC1 variance、不refit。M0只能作为失败参照，不能成为1.0.2。

### M1 influence + draw-specific studentization

Numerator保持 `D*_b(u,z)=sum_i xi_bi Delta_i(u,z)`。对每个draw、candidate和point，将`xi_bi Delta_i`跨该candidate的患者中心化，仅用中心化后的值重算

`V*_b(u,z)=G/(G-1) sum_i [xi_bi Delta_i(u,z)-mean_j xi_bj Delta_j(u,z)]²`。

Bootstrap process使用`D*_b/sqrt(V*_b)`；observed process仍使用原observed V。一个患者的同一multiplier跨全部时间、candidate和point复用。M1不refit FQI。主multiplier Gaussian；Rademacher/Webb只在Gaussian结构通过RB1后做paired sensitivity。

### M2 restricted-null common-Q full-refit

完整分析窗拟合一个patient-balanced common beta `beta_c`。令full-window patient-balanced Gram为`H0=X'WX`，解

`H0 v_ridge = alpha beta_c`。

Common-model TD residual为

`delta_it=R_it+gamma max_a Q_c(S_next,a)-x_it'beta_c`，

并在每名患者内作scalar mean centering得到`delta_tilde_it`。第b个draw的pseudo reward严格为

`R*_bit=x_it'beta_c-gamma max_a Q_c(S_next,a)+x_it'v_ridge+xi_bi delta_tilde_it`。

该公式只含同一个common beta、固定states/actions、ridge-consistent offset和patient residual perturbation；不含observed left-right beta/Q difference。相同patient multiplier跨全轨迹、所有candidate复用。

每个draw、每个admissible candidate从零开始执行左右完整frozen Bellman FQI至收敛，随后重新计算beta、greedy action、TD residual、patient score、W、eta、Delta、V和64点process，最后形成D0–D4。Zero perturbation必须数值恢复common-null construction。主multiplier Gaussian；Rademacher/Webb仅为预注册sensitivity。

### M0-global-scale negative control

固定将M0所有bootstrap statistic乘以

`c=1/0.6012249192947142=1.66327104534037`。

该常数来自Phase 2R-A保存的RS D4 q95比，在本轮数据生成前冻结；它对D0–D4使用同一常数，不能成为正式方法。用途是判断结构候选是否只是把临界值整体变大。

## 3. D0–D4与observed invariance

- D0：midpoint `u=144`、grid第0点、absolute unnormalized contrast；
- D1：同一点studentized；
- D2：midpoint的64点maximum；
- D3：七候选、单point、harmonic boundary maximum；
- D4：七候选×64点的完整normalized maximum。

所有臂的observed D0–D4必须逐数组等于冻结Phase 1 calculation。Bootstrap内不得重筛support或改变`U_adm`。

## 4. Stages与不可更改gate

RB0：fresh R=20、B=199，M0/M1/M2 Gaussian；只检查工程有效性，M2 draw failure必须≤1%。不因size停止。

RB1：另一组fresh N0，R=200、B=499，比较M0/M1/M2 Gaussian及M0-global-scale。候选继续要求同时满足D0≤0.10、D2≤0.15、D4≤0.15、D4 q95/oracle≥0.85、failure≤1%。M1/M2均失败则直接`PHASE_2RB_REVISION_FAILED`。

只有Gaussian结构通过RB1，才在完全相同RB1 datasets上运行对应Rademacher/Webb paired sensitivity。

RB2：其seed list已在RB1前冻结且不读取结果。仅通过候选进入，R=500、B=999。接受范围：D0约0.03–0.08，D2/D3/D4≤0.10，D4 q95比0.90–1.10，无near-zero p pile-up，failure≤1%；D4>0.15硬失败。

另外，D4的0.05 nominal必须落在95% Wilson区间内，或与empirical size相差不超过一个MCSE。协方差确认不是“看图决定”：midpoint与joint两种scope在RB2都必须满足预先冻结的宽反病理范围（pointwise SD比0.75–1.25、leading eigenvalue比0.60–1.40、effective-rank比0.60–1.67、correlation Frobenius error≤0.75、off-diagonal MAE≤0.20、conditional-mean RMS/truth mean SD≤0.20）；同时在同一RB1数据上，相对M0的五项joint-process指标至少三项更接近truth，且任一指标不得相对恶化超过25%。这些是防止“只放大critical value”的结构gate，不用于选择size最接近0.05的arm。

## 5. Covariance-process confirmatory endpoint

RB1至少200个independent N0 datasets形成truth covariance。对M0/M1/M2分别平均conditional bootstrap covariance，比较midpoint 64-point process及all-candidate×point 448 components：pointwise SD、leading 10 eigenvalues、effective rank、correlation Frobenius error、off-diagonal MAE、D2/D4 q90/q95/q99。

M2若只把critical values放大但covariance仍错误，不能视为解决。M0-global-scale应展示统一scale无法同时匹配D0、D2、D4及covariance shape。

## 6. Transfer-null与正式版本gate

只有RB2通过才运行N1/N2/N3/N5/N7/N8 mild/strong，每scenario R=100、B=499，只比较M0和通过候选，使用原primary support/scaling。任何testable conditional size>0.15即阻止1.0.2；0.08–0.15为yellow flag。

只有RB2、joint covariance和transfer全部通过，才允许建立`1.0.2`，且唯一修改是inference calibration；observed RS方法不变。成功后的状态也只能是`INFERENCE_REVISION_CONFIRMED`，然后从头重新开始新方法的完整Phase 2 calibration。Phase 3与Ohio仍为NO。

## 7. 双候选预冻结选择规则

若M1/M2都通过RB2和transfer，仅当M1 q95比在0.90–1.10、all-candidate correlation error相对不比M2差超过10%、且任一transfer size不比M2绝对差超过0.03时，才因定义简单优先M1；否则M2通过而M1不满足时选M2。绝不以0.049比0.061“更好看”选择。Multiplier sensitivity不会自动成为primary。
