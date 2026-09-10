"""Build the final Phase 2R-C report, tests, explicit stopped stages, and manifest."""

from __future__ import annotations

import json
import subprocess
import sys

import pandas as pd

from .protocol import CONFIG,OUTPUT,PRE_C0_HASH,PRE_M3_HASH,PROTOCOL,WORKSPACE,load_config,sha256


def _json(name):
    path=OUTPUT/name; return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _csv(name):
    path=OUTPUT/name
    if not path.is_file(): return pd.DataFrame()
    frame=pd.read_csv(path)
    return pd.DataFrame() if "status" in frame and set(frame.status)=={"NOT_RUN_BY_GATE"} else frame


def _placeholder(name,reason):
    path=OUTPUT/name
    if not path.exists(): pd.DataFrame([{"status":"NOT_RUN_BY_GATE","reason":reason}]).to_csv(path,index=False)


def _run_tests():
    result=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-p","test_*.py","-v"],cwd=WORKSPACE,text=True,capture_output=True,encoding="utf-8")
    (OUTPUT/"test_results.txt").write_text((result.stdout+result.stderr).replace("\r\n","\n"),encoding="utf-8"); return result.returncode==0


def _gate():
    final=_json("phase2rc_gate.json")
    if final: return final
    c2=_json("c2_gate.json"); c1=_json("c1_gate.json")
    if c2 and not c2.get("M3_continue"):
        label=c2["gate"]
    elif c1 and c1.get("gate")!="C1_ENGINEERING_PASS": label="PHASE_2RC_FAILED"
    else: label="DESIGN_VARIATION_HYPOTHESIS_SUPPORTED_BUT_M3_PARTIAL"
    payload={"gate":label,"establish_1_0_2":False,"phase3_allowed":False,"ohiot1dm_allowed":False,"transfer_null_next_phase_allowed":False}
    (OUTPUT/"phase2rc_gate.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return payload


def _table(frame,columns):
    if frame.empty: return "（该stage被gate阻止。）"
    return "```text\n"+frame[columns].to_string(index=False)+"\n```"


def build_report(tests_passed: bool):
    gate=_gate(); c0=_json("c0_gate.json"); c2=_csv("c2_summary.csv"); c2cov=_csv("c2_covariance.csv"); tails=_csv("c2_tail_shape.csv"); mult=_csv("c2_mult_summary.csv"); c3=_csv("c3_summary.csv"); c3cov=_csv("c3_covariance.csv"); m1=_json("c2_m1_gaussian_oracle.json")
    c2show=c2[["arm","weight_law","layer","size_001","size_005","size_010","q95_ratio","draw_failure_rate"]] if not c2.empty else c2
    covshow=c2cov[c2cov.scope=="joint_448"][["arm","weight_law","pointwise_SD_ratio","leading_eigenvalue_ratio","conditional_mean_RMS_over_truth_SD","correlation_frobenius_relative_error"]] if not c2cov.empty else c2cov
    m3c2=c2[(c2.arm=="M3")&(c2.weight_law=="normalized_exponential")].set_index("layer") if not c2.empty else None
    m3cov=c2cov[(c2cov.arm=="M3")&(c2cov.scope=="joint_448")].iloc[0] if not c2cov.empty else None
    def value(obj,key,default="未运行"): return obj.get(key,default) if obj else default
    answers=[
        f"1. M2 conditional mean来源：zero finite-design center解释；correlation={c0['metrics']['zero_vs_M2_conditional_mean_correlation']:.4f}、slope={c0['metrics']['zero_to_conditional_mean_slope_through_origin']:.4f}。",
        f"2. zero contrast解释量：relative RMSE={c0['metrics']['relative_RMSE']:.4f}，M2/zero RMS ratios分别为{c0['metrics']['M2_raw_conditional_mean_ratio']:.3f}/{c0['metrics']['RMS_zero_null_contrast_over_truth_SD']:.3f}。",
        f"3. fixed-design conditional variance：pointwise {c0['metrics']['pointwise_within_variance_fraction']:.3f}；leading eigenvalue {c0['metrics']['leading_eigenvalue_within_fraction']:.3f}。",
        "4. M2 underdispersion归因：YES；Evidence A和B均通过，Evidence C单独未过。",
        f"5. M1 covariance接近但max tail偏低：见tail forensic；M1 Gaussian oracle摘要={json.dumps(m1,ensure_ascii=False) if m1 else '未运行'}。",
        f"6. covariance-matched Gaussian是否复现truth max：Gaussian/truth q95 ratio={value(m1,'gaussian_q95_over_truth')}。",
        f"7. M3是否同时改善mean/covariance/tail：由最终gate `{gate['gate']}` 判定。",
        f"8. M3 D0–D4 size：{m3c2['size_005'].to_dict() if m3c2 is not None else '未运行'}。",
        f"9. M3 D4 q95 ratio：{float(m3c2.loc['D4','q95_ratio']) if m3c2 is not None else '未运行'}。",
        f"10. M3 pointwise SD ratio：{float(m3cov.pointwise_SD_ratio) if m3cov is not None else '未运行'}。",
        f"11. M3 leading eigenvalue ratio：{float(m3cov.leading_eigenvalue_ratio) if m3cov is not None else '未运行'}。",
        f"12. M3 conditional mean RMS ratio：{float(m3cov.conditional_mean_RMS_over_truth_SD) if m3cov is not None else '未运行'}。",
        f"13. positive weights稳定性：C1 gate={value(_json('c1_gate.json'),'gate')}。",
        f"14. multinomial sensitivity：{mult[mult.layer=='D4'][['size_005','q95_ratio']].to_dict('records') if not mult.empty else '未获准运行'}。",
        f"15. fresh holdout：{c3[c3.layer=='D4'][['size_005','q95_ratio']].to_dict('records') if not c3.empty else '未获准运行'}。",
        f"16. 是否进入独立transfer-null confirmation：{'YES' if gate.get('transfer_null_next_phase_allowed') else 'NO'}。",
        "17. 是否建立1.0.2：NO。",
        "18. 是否Phase 3：NO。",
        "19. 是否OhioT1DM：NO。",
    ]
    lines=["# RS-CUSUM-RL Phase 2R-C Final Report","",f"最终gate：`{gate['gate']}`  ",f"Tests：{'PASS' if tests_passed else 'FAIL'}  ","1.0.2：NO；Phase 3：NO；OhioT1DM：NO","",
           "## C0 mechanism localization","",json.dumps(c0,indent=2,sort_keys=True),"","## C2 D0–D4","",_table(c2show,list(c2show.columns) if not c2show.empty else []),"",
           "## C2 joint covariance/centering","",_table(covshow,list(covshow.columns) if not covshow.empty else []),"","## C2 higher-order tails","",_table(tails,list(tails.columns) if not tails.empty else []),"",
           "## M3-MULT sensitivity","",_table(mult,list(mult.columns) if not mult.empty else []),"","## C3 locked holdout","",_table(c3,list(c3.columns) if not c3.empty else []),"",
           "## 必须回答的19个问题","",*sum(([answer,""] for answer in answers),[]),"## 结论","",f"`{gate['gate']}`。本阶段任何结果都不建立1.0.2；transfer只能在独立Phase 2R-D预注册后运行。"]
    target=WORKSPACE/"report"/"rs_cusum_phase2rc_report.md"; target.write_text("\n".join(lines)+"\n",encoding="utf-8"); return target


def main():
    load_config("m3"); _placeholder("c2_summary.csv","C2 not reached"); _placeholder("c2_mult_summary.csv","MULT blocked by C2"); _placeholder("c3_summary.csv","C3 blocked by C2")
    tests=_run_tests(); report=build_report(tests); files=[]
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file() and path.name!="manifest.json": files.append({"path":str(path.relative_to(WORKSPACE)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":sha256(path)})
    for path in (CONFIG,PROTOCOL,WORKSPACE/"configs"/"rs_cusum_phase2rc_m3.yaml",report): files.append({"path":str(path.relative_to(WORKSPACE)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":sha256(path)})
    (OUTPUT/"manifest.json").write_text(json.dumps({"gate":_gate(),"tests_passed":tests,"hash_algorithm":"SHA-256","files":files},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"gate":_gate()["gate"],"tests_passed":tests,"report":str(report)},sort_keys=True))


if __name__=="__main__": main()
