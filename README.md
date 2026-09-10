# Manuscript experiment archive

Extract BOTH ZIP files into the SAME empty working directory. Original repository-relative paths are retained, including the plotting script's nested submission path. Python source/configuration/result bytes are unchanged. SHA256SUMS.json records every packaged source file. The two archives share this documentation but do not duplicate experimental files.

## Scope and interpretation

EXPERIMENT_MAP.md maps manuscript evidence to code/results. phase2rf, phase2rg, phase2rh, representative_altloc and m4_realdata_bridge_v0 contain final experiments. Earlier stages are retained as development evidence and because the final code imports their engines and reads their fixed grids, seeds, gates and protocol records. Do not treat a historical development gate as the final conclusion. Read stage manifests, protocol files and final gate files together; a marker named SUCCESS can denote completed execution even when a scientific gate fails.

## Software and verification

requirements.txt lists directly imported third-party packages discovered from the packaged Python AST. Versions are not guessed or presented as an original lockfile. Consult archived run_metadata.json files for any recorded environment/version details. A compatible Python with modern type-hint syntax is needed (Python 3.10+). No long Monte Carlo experiments were rerun during packaging. Packaging validation establishes archive integrity and dependency/source coverage, not numerical reproducibility on a new environment.

After extraction: python -m pip install -r requirements.txt
Frozen evidence inspection needs no execution. For summary regeneration, work on a disposable extracted copy because summary commands overwrite outputs:

    python -m src.rs_cusum_phase2rf.summarize
    python -m src.rs_cusum_phase2rg.summarize
    python -m src.rs_cusum_phase2rh.summarize

Runner interfaces (inspect --help first):

    python -m src.rs_cusum_phase2rf.runner --help
    python -m src.rs_cusum_phase2rg.runner --help
    python -m src.rs_cusum_phase2rh.runner --help
    python -m src.rs_cusum_representative_altloc.runner --help
    python -m src.rs_cusum_realdata_bridge_v0.runner --help
    python -m src.ohio_final_empirical_application.runner --help

The first three runners expose oracle-primary, oracle-reference and m4 stages; the latter runners expose formal and freeze-related stages. Existing checkpoints can cause a runner to resume/skip work. Do not invoke freeze against the supplied frozen archive: it can reject existing checkpoints or regenerate registries. A from-scratch replication requires a separate workspace and the documented protocol initialization sequence. Configurations/default.yaml and pilot configurations are retained only as dependencies/provenance; the experiment-specific configuration in EXPERIMENT_MAP governs each final result.

## OhioT1DM data boundary

Raw OhioT1DM XML, results_mdp_gate patient feature arrays, per-patient timestamp tables, and individual preprocessing outputs are NOT distributed. Obtain the dataset through its official access process under its own terms and run the supplied preprocessing code with configs/mdp_5min_protocol.yaml. Final observed FQI expects results_mdp_gate/arrays/training_<patient>_5min.npz. The archived input SHA-256 values allow comparison with the original inputs. Aggregate Ohio statistics, candidate processes, support diagnostics and 199 bootstrap-failure records are included. Full Ohio reproduction is conditional on independently obtaining the required input data; these two ZIPs alone do not provide those data.

Historical provenance files may name withheld data files, dataset subject codes, local paths, or excluded execution logs. Their existence in a registry does not mean those files are distributed. Release hashes are in SHA256SUMS.json; original registries remain untouched. No credential files, .env files, IDE settings, caches, external papers, or raw patient XML are selected. No publication upload has been performed and no copyright license has been invented; the authors must select an appropriate code/results license when depositing.

## Known inherited entry-point limitation

The standalone demonstration at the bottom of src/parse_xml.py imports configs.default_config, which is absent from the original repository. This optional __main__ demonstration is not the final preprocessing entry point. Use python src/run_mdp_gate.py with independently obtained Ohio data and the frozen YAML protocol; it imports parse_xml_file without executing that demonstration. This inherited limitation is reported, not repaired inside the frozen analysis code.

All archived clinical trajectory lengths, support counts and aggregate statistics remain as recorded. Raw measurements and per-patient calendar timestamps are excluded. The result files are original archival bytes, so historical traceback strings can retain local filesystem paths.
