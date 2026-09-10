"""Build the Phase 2R-B report, explicit stopped-stage artifacts, tests, and manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .protocol import CONFIG, OUTPUT, PROTOCOL, WORKSPACE, load_config, sha256


REQUIRED_RESULTS = (
    "pre_run_hashes.json", "rb0_smoke.json", "rb1_summary.csv",
    "rb2_holdout_summary.csv", "decomposition.csv", "covariance_process.csv",
    "quantile_comparison.csv", "multiplier_sensitivity.csv",
    "transfer_null_summary.csv", "replicate_results.csv", "test_results.txt",
)


def _read_json(name: str) -> dict[str, Any] | None:
    path=OUTPUT/name
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _placeholder(name: str, reason: str) -> None:
    path=OUTPUT/name
    if not path.exists():
        pd.DataFrame([{"status":"NOT_RUN_BY_GATE","reason":reason}]).to_csv(path,index=False)


def _combine_replicates() -> None:
    paths=sorted(OUTPUT.glob("*_replicate_results.csv"))
    if paths:
        pd.concat([pd.read_csv(path) for path in paths],ignore_index=True).to_csv(
            OUTPUT/"replicate_results.csv",index=False
        )


def _run_tests() -> tuple[bool,str]:
    command=[sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py","-v"]
    result=subprocess.run(command,cwd=WORKSPACE,text=True,capture_output=True,encoding="utf-8")
    text=(result.stdout+result.stderr).replace("\r\n","\n")
    (OUTPUT/"test_results.txt").write_text(text,encoding="utf-8")
    return result.returncode==0,text


def _table(frame: pd.DataFrame, columns: list[str], digits: int = 4) -> str:
    if frame.empty or not set(columns).issubset(frame.columns): return "（无：该stage被gate阻止。）"
    shown=frame[columns].copy()
    for column in shown.select_dtypes(include="number"):
        shown[column]=shown[column].map(lambda value:"" if pd.isna(value) else f"{value:.{digits}f}")
    widths={column:max(len(str(column)),*(len(str(value)) for value in shown[column])) for column in columns}
    header="| "+" | ".join(str(column).ljust(widths[column]) for column in columns)+" |"
    rule="| "+" | ".join("-"*widths[column] for column in columns)+" |"
    body=["| "+" | ".join(str(row[column]).ljust(widths[column]) for column in columns)+" |" for _,row in shown.iterrows()]
    return "\n".join([header,rule,*body])


def _final_gate() -> dict[str, Any]:
    existing=_read_json("phase2rb_gate.json")
    if existing is not None: return existing
    rb2=_read_json("rb2_gate.json"); rb1=_read_json("rb1_gate.json"); rb0=_read_json("rb0_smoke.json")
    if rb1 and rb1.get("gate")=="PHASE_2RB_REVISION_FAILED": gate="PHASE_2RB_REVISION_FAILED"
    elif rb2 and rb2.get("gate")!="RB2_HOLDOUT_PASS": gate="INFERENCE_REVISION_PARTIAL"
    elif rb0 and rb0.get("gate")!="RB0_ENGINEERING_PASS": gate="PHASE_2RB_REVISION_FAILED"
    else: gate="INFERENCE_REVISION_PARTIAL"
    payload={
        "gate":gate,"selected_candidate":None,"establish_1_0_2":False,
        "restart_new_phase2_calibration_allowed":False,"phase3_allowed":False,
        "ohiot1dm_allowed":False,"reason":"A required downstream gate was not reached or passed.",
    }
    (OUTPUT/"phase2rb_gate.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return payload


def _arm_answer(arm: str, rb1: pd.DataFrame, rb2: pd.DataFrame) -> str:
    if not rb2.empty and "arm" in rb2 and arm in set(rb2.arm):
        row=rb2[(rb2.arm==arm)&(rb2.layer=="D4")].iloc[0]
        return f"holdout size={row.size_005:.3f}, q95 ratio={row.mean_bootstrap_q95_over_oracle:.3f}"
    if not rb1.empty and "arm" in rb1 and arm in set(rb1.arm):
        row=rb1[(rb1.arm==arm)&(rb1.layer=="D4")].iloc[0]
        return f"RB1 size={row.size_005:.3f}, q95 ratio={row.mean_bootstrap_q95_over_oracle:.3f}; 未获holdout确认"
    return "未运行"


def build_report(tests_passed: bool) -> Path:
    cfg=load_config(); gate=_final_gate(); rb0=_read_json("rb0_smoke.json") or {}
    def csv(name: str) -> pd.DataFrame:
        path=OUTPUT/name
        if not path.is_file(): return pd.DataFrame()
        frame=pd.read_csv(path)
        return pd.DataFrame() if "status" in frame and set(frame.status)=={"NOT_RUN_BY_GATE"} else frame
    rb1=csv("rb1_summary.csv"); rb2=csv("rb2_holdout_summary.csv")
    sens=csv("multiplier_sensitivity.csv"); cov=csv("covariance_process.csv")
    transfer=csv("transfer_null_summary.csv"); audit=csv("restricted_null_audit.csv")
    rb1_show=rb1[(rb1.distribution=="gaussian")][
        ["arm","layer","size_001","size_005","size_010","mean_bootstrap_q95_over_oracle","draw_failure_rate"]
    ] if not rb1.empty else pd.DataFrame()
    rb2_show=rb2[["arm","layer","size_001","size_005","size_010","mean_bootstrap_q95_over_oracle","draw_failure_rate"]] if not rb2.empty else pd.DataFrame()
    sens_show=sens[sens.layer=="D4"][["arm","distribution","size_005","mean_bootstrap_q95_over_oracle","draw_failure_rate"]] if not sens.empty else pd.DataFrame()
    cov_show=cov[cov.scope=="joint_448"][["stage","arm","distribution","mean_pointwise_SD_ratio","leading_eigenvalue_ratio","effective_rank_ratio","correlation_frobenius_relative_error","offdiagonal_correlation_MAE"]] if not cov.empty else pd.DataFrame()
    transfer_show=transfer[["scenario","arm","size_005","testable_replicates","mean_bootstrap_q95_over_oracle"]] if not transfer.empty else pd.DataFrame()
    selected=gate.get("selected_candidate")
    version_statement=(
        f"建立1.0.2；唯一修改为以{selected}替换fixed-contribution multiplier calibration，observed RS statistic及其所有组成不变。"
        if gate.get("establish_1_0_2") else "不建立1.0.2；候选尚未同时通过holdout、joint-process和transfer gate。"
    )
    lines=[
        "# RS-CUSUM-RL Phase 2R-B Inference Revision Confirmatory Report", "",
        f"最终gate：`{gate['gate']}`  ", f"父方法：`{cfg['protocol']['parent_method']}`  ",
        f"测试：{'PASS' if tests_passed else 'FAIL'}  ", "Phase 3：`NO`；OhioT1DM：`NO`", "",
        "## 1. Protocol、实现与审计", "",
        "本轮仅运行null inference calibration；未运行alternative、power、changepoint或OhioT1DM。observed estimator、risk set、support、gamma、ridge、W/Delta、harmonic scaling、64点grid及七候选均冻结。RB2 seeds在RB1前生成并哈希。",
        "", f"RB0：`{rb0.get('gate','NOT_RUN')}`；M2有效draw比例={rb0.get('checks',{}).get('M2_draw_valid_fraction','NA')}。",
        "", "## 2. RB1 fresh N0 screening：D0–D4", "",
        _table(rb1_show,list(rb1_show.columns) if not rb1_show.empty else []), "",
        "## 3. RB2 locked holdout", "", _table(rb2_show,list(rb2_show.columns) if not rb2_show.empty else []), "",
        "## 4. Multiplier sensitivity", "", _table(sens_show,list(sens_show.columns) if not sens_show.empty else []), "",
        "Sensitivity只在Gaussian结构通过RB1后、使用相同RB1 datasets运行；不按size最接近0.05自动选primary。", "",
        "## 5. Joint covariance-process comparison", "", _table(cov_show,list(cov_show.columns) if not cov_show.empty else []), "",
        "M0-global-scale只改变scale而不改变correlation shape，始终是negative control，不能成为方法。", "",
        "## 6. Restricted-null construction audit", "",
        _table(audit,list(audit.columns) if not audit.empty else []), "",
        "## 7. Transfer-null", "", _table(transfer_show,list(transfer_show.columns) if not transfer_show.empty else []), "",
        "## 8. 对17个必须问题的回答", "",
        f"1. M1是否修复D4：{_arm_answer('M1',rb1,rb2)}。", "",
        f"2. M2是否修复D4：{_arm_answer('M2',rb1,rb2)}。", "",
        "3. M1/M2 D0–D4 size：见第2节及第3节逐层表。", "",
        "4. D4 bootstrap/oracle q95 ratio：见同表的`mean_bootstrap_q95_over_oracle`。", "",
        "5. joint covariance eigenvalue ratio：见第5节`leading_eigenvalue_ratio`。", "",
        "6. correlation error：见第5节，且RB2 gate同时检查绝对范围及RB1相对M0改善。", "",
        "7. full-refit是否只是critical-value放大：由M0-global-scale的D0–D4、eigen/SD与不变的correlation error对照判断；结构gate不允许仅凭q95通过。", "",
        "8. Gaussian/Rademacher/Webb差异：见第4节；未达到sensitivity gate时明确不运行。", "",
        "9. fresh holdout是否复现RB1：见第3节及最终gate。", "",
        f"10. M2 draw failure：见逐层表及restricted-null audit；RB0有效比例={rb0.get('checks',{}).get('M2_draw_valid_fraction','NA')}。", "",
        "11. N1/N2/N3/N5/N7/N8 transfer：见第7节。", "",
        f"12. 是否有scenario >0.15：{('见transfer表；final gate已按>0.15阻断' if not transfer.empty else 'transfer未获准运行，不能判断')}。", "",
        f"13. 是否足以建立1.0.2：{'YES' if gate.get('establish_1_0_2') else 'NO'}。", "",
        f"14. 1.0.2唯一修改：{version_statement}", "",
        f"15. 是否允许重启新的Phase 2 calibration：{'YES' if gate.get('restart_new_phase2_calibration_allowed') else 'NO'}。", "",
        "16. 是否允许Phase 3：NO。即使本轮确认，也必须先从头完成新1.0.2的完整Phase 2。", "",
        "17. 是否允许OhioT1DM：NO。", "",
        "## 9. 结论", "", f"`{gate['gate']}`。{version_statement}", "",
        "所有stage-level raw rows、quantiles、p-value ECDF/histogram、covariance、restricted-null audit、hash与manifest均保存在`results_rs_cusum/phase2rb/`。",
    ]
    target=WORKSPACE/"report"/"rs_cusum_phase2rb_inference_revision_report.md"
    target.write_text("\n".join(lines)+"\n",encoding="utf-8")
    return target


def build_manifest(report: Path, tests_passed: bool) -> None:
    files=[]
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name!="manifest.json":
            files.append({"path":str(path.relative_to(WORKSPACE)).replace("\\","/"),
                          "bytes":path.stat().st_size,"sha256":sha256(path)})
    for path in (CONFIG,PROTOCOL,report):
        files.append({"path":str(path.relative_to(WORKSPACE)).replace("\\","/"),
                      "bytes":path.stat().st_size,"sha256":sha256(path)})
    payload={"phase":"Phase 2R-B","tests_passed":tests_passed,"gate":_final_gate(),
             "files":files,"hash_algorithm":"SHA-256"}
    (OUTPUT/"manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True,exist_ok=True); load_config()
    _combine_replicates()
    rb1=_read_json("rb1_gate.json"); rb2=_read_json("rb2_gate.json")
    if not (OUTPUT/"rb1_summary.csv").exists(): _placeholder("rb1_summary.csv","RB1 was not reached")
    if not (OUTPUT/"rb2_holdout_summary.csv").exists(): _placeholder("rb2_holdout_summary.csv","RB2 blocked by RB1 gate")
    if not (OUTPUT/"multiplier_sensitivity.csv").exists(): _placeholder("multiplier_sensitivity.csv","No Gaussian candidate passed RB1")
    if not (OUTPUT/"transfer_null_summary.csv").exists(): _placeholder("transfer_null_summary.csv","Transfer blocked by RB2 gate")
    for name in ("decomposition.csv","covariance_process.csv","quantile_comparison.csv","replicate_results.csv"):
        if not (OUTPUT/name).exists(): _placeholder(name,"Upstream confirmatory stage was not reached")
    tests_passed,_=_run_tests()
    report=build_report(tests_passed); build_manifest(report,tests_passed)
    print(json.dumps({"gate":_final_gate()["gate"],"tests_passed":tests_passed,"report":str(report)},sort_keys=True))


if __name__=="__main__": main()
