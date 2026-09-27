# Braun (arXiv:2607.21831) baseline on the environments_rescofull benchmark

Braun, "A graph-based control interface for traffic signals on heterogeneous road
networks", arXiv:2607.21831 (July 2026). Code: github.com/BertilBraun/GNN-Traffic-Signal-Control,
**pinned to commit `ea47985ccba2bbca273eb08139645399cf53ef23`** (the snapshot the paper
cites), unpacked from a tarball at

    /home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23

That repository has **no LICENSE file**, so none of its source is copied here. Everything
in this directory either runs his code from that path (imports / subprocess) or is our own
adapter/measurement code.

## Environment (isolated; never our miniconda env)

    curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uv-install.sh
    UV_NO_MODIFY_PATH=1 sh /tmp/uv-install.sh                 # uv 0.12.19 -> ~/.local/bin
    cd /home/deea/external/GNN-Traffic-Signal-Control-ea47985ccba2bbca273eb08139645399cf53ef23
    ~/.local/bin/uv sync --frozen --group dev                 # .venv: torch 2.12.1+cu130, torch-geometric 2.8.0,
                                                              # libsumo/traci/sumolib 1.27.1 (from his uv.lock)
    baselines/braun/bpy.sh -m pytest -q -p no:cacheprovider   # 271 pass; 23 fail only because his
                                                              # OSM city builds (gitignored) are absent

`bpy.sh` runs the venv's python with `PYTHONPATH` cleared (ROS / `$SUMO_HOME/tools` would
otherwise shadow his pinned traci/sumolib) and cwd = his repo.

## Pipeline (all commands from the repo root)

1. Scenarios + feasibility report (his phase synthesis on RESCO nets):

       baselines/braun/bpy.sh $PWD/baselines/braun/build_resco_scenarios.py
       python3 baselines/braun/summarize_feasibility.py      # -> results/braun/feasibility.json
       python3 baselines/braun/check_link_coverage.py        # links green in no phase
       baselines/braun/bpy.sh $PWD/baselines/braun/diagnose_gneJ210.py

   Writes `scenarios/<net>/{synth,synthfb,native}.sumocfg` (+ tll files). Windows are read
   from `environments_rescofull/*/config.yaml`.
   * `synth`   = his synthesized phases everywhere (`scripts/build_network.py::_build_tll`,
                 called verbatim; tlLogic ids remapped junction-id -> TLS-id for the 1
                 cologne3 and 4 ingolstadt7 TLS whose ids differ).
   * `synthfb` = `synth`, except ingolstadt7/`gneJ210`, where his synthesizer leaves links
                 6-9 green in NO phase (atomic group with internal SUMO foes is dropped);
                 that junction keeps RESCO's program, as his `_build_tll` does for every
                 junction it rejects. **Primary "as published" arm.** Identical to `synth` on
                 arterial4x4, cologne3, grid4x4.
   * `native`  = RESCO's own program (arm B: his learner restricted to existing greens).

2. Training (sequential queue, skip-or-resume; ~100 s per PPO iteration with 10 workers):

       nohup baselines/braun/run_training_queue.sh > results/braun/train/queue.log 2>&1 &
       # one run: baselines/braun/train_braun.sh synthfb 3 85 cuda

   Config: `configs/rescofull_{synthfb,native}.yaml` — his
   `city_first_pass_throughput_scratch_32_worker.yaml` with ADAPTED lines marked
   (initial occupancy 0, demand scale 1.0, 360-decision rollouts, 10 workers, no periodic
   eval / checkpoint selection). Architecture flags from his `train.sh` (scratch, lane 29,
   movement 4, hidden 64, 1 hop). Snapshots at iterations 1, 10, 85 in
   `results/braun/train/<arm>_s<seed>/snapshots/`.

3. Evaluation (one measurement pipeline for every row):

       baselines/braun/run_eval_ours.sh holdout|city_4|city_6   # ours + our rule-based
       baselines/braun/run_eval_braun.sh [grid4x4 cologne3 ingolstadt7]   # Braun, skip-or-resume
       python3 baselines/braun/aggregate.py                      # tables + results/braun/summary.json

   `trip_metrics.py` exec's `diagnostics/eval_paper_metrics.py::run_episode` from its own
   source (ast), so our and Braun's rows use the identical polling code, and parses SUMO
   tripinfo for both. `eval_braun.py` wraps Braun's runtime in a 5-s reset()/step() shim;
   control is his code unchanged.
