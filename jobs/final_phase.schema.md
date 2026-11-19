# jobs/final_phase.yaml: manifest format for the final-phase runner

Read by `scripts/fp_runner.py` (code in `mirror/final/run/runner.py`; tests in `tests/test_fp_runner.py`). Written
for FINAL_PHASE_PLAN.md v2, Sections 7 (item 11), 8 and 9. The coordinator fills `jobs/final_phase.yaml`; this file
fixes its format. Check a manifest with `python scripts/fp_runner.py check jobs/final_phase.yaml` (it lists every
problem at once and prints jobs, heal-units and cut ranks per wave).

## 1. Running it on the VM

```
bash scripts/fp_vm_w0.sh clock env gpu prereg          # by hand, before the runner (the venv and the prereg check)
. ~/.mirror_env.sh
setsid nohup python scripts/fp_runner.py run jobs/final_phase.yaml > logs/final/runner.out 2>&1 < /dev/null &
python scripts/fp_runner.py status jobs/final_phase.yaml     # states and reasons, writes nothing
python scripts/fp_runner.py project jobs/final_phase.yaml    # projection and the cuts it would make, writes nothing
```

`run` refuses to start unless `mirror.final.prereg.require_prereg()` passes. Stopping the runner leaves its jobs
running; starting it again resumes (it never starts a job twice). Exit codes: 0 every job settled, 1 a job failed or
was blocked, 2 a stop event (two failures of one job, or a `stop_when` condition such as Gate P), 3 the 15-hour
ceiling, 130 the runner was stopped while jobs run.

## 2. Layout

```yaml
version: 1
settings: {...}          # optional; defaults below
jobs:
  - id: ...
```

Unknown keys are errors (a typo never passes silently). Paths are relative to the repository root.

## 3. Settings

| Key | Default | Meaning |
|---|---|---|
| `ceiling_hours` | 15.0 | VM hours from the start of W0 at which every job the runner started gets SIGTERM, then SIGKILL after `term_grace_s`; the runner writes the status and exits (code 3) |
| `heal_stop_hours` | 13.5 | from here no new heal starts, and only `after_heal_stop_kinds` may start |
| `after_heal_stop_kinds` | [eval, decision, report] | kinds that may still start after `heal_stop_hours` ("finish running jobs, evaluations and the report") |
| `end_reserve_hours` | 0.5 | a heal waits if its expected end (heal_units x measured hours per unit) passes `ceiling_hours - end_reserve_hours` |
| `headroom_gb` | 10 | GPU memory always left free |
| `declared_budget` | true | also require: declared GB of running GPU jobs + this job's GB <= total GPU memory - headroom |
| `warmup_s` | 240 | the declared GB of GPU jobs started less than this long ago count as used (their memory may not be allocated yet) |
| `stagger_s` | 30 | at least this long between two GPU job starts |
| `util_target` | 0.85 | when heals run and the mean utilisation over `util_window_s` is below this, `light_kinds` jobs go first; otherwise heals go first |
| `util_window_s` | 300 | window of the utilisation mean |
| `light_kinds` | [light, eval] | the kinds packed next to heals |
| `max_running`, `max_heals`, `max_cpu` | 16, 3, 6 | caps on running jobs (services excluded from `max_running`), heal jobs, and jobs without GPU memory |
| `host_headroom_gb` | 16 | host memory left free when a job declares `host_mem_gb` (the 15 November OOM) |
| `retry_mem_factor` | 1.5 | a job that failed before asks for this times its GPU memory |
| `max_failures` | 2 | failures of one job that stop all new launches (the plan's stop event) |
| `max_lost` | 3 | restarts of a job whose process vanished without an exit code (machine stopped) |
| `grace_s` | 90 | a launched job counts as running this long before its wrapper must have written its pid |
| `term_grace_s` | 60 | SIGTERM to SIGKILL |
| `poll_s`, `gpu_log_s`, `status_s` | 20, 30, 60 | scheduling interval, nvidia-smi CSV interval, STATUS.md interval (also rewritten after every event) |
| `waves` | [W0, W1, W2, W3, W4, W5, End] | wave names in order |
| `barrier_waves` | [W0] | no job of a later wave starts before every job of these waves is settled |
| `recut_each_wave` | true | project (and cut if needed) before the first start of every wave after W0, not only after W0 |
| `stop_when` | [] | conditions (Section 5) that stop new launches when they hold, for example Gate P |
| `env` | {} | environment for every job, for example thread caps `{MIRROR_THREADS: "4", OMP_NUM_THREADS: "4", MKL_NUM_THREADS: "4"}` |
| `log_dir`, `state_dir` | logs/final, logs/final/state | job logs `<log_dir>/<id>.log`; state files (Section 7) |
| `status_file`, `decisions_file`, `gpu_csv` | results/STATUS.md, notes/decisions.md, logs/final/gpu.csv | outputs |
| `projection` | see below | the projection and cut rule |

`projection`:

| Key | Default | Meaning |
|---|---|---|
| `target_hours` | 13.5 | cut until the projected total is at most this |
| `reserve_hours` | 0.5 | added to the projection (final report, pulls) |
| `source` | auto | `timing_file`, `jobs`, `manual`, or `auto`: the finished heal jobs outside W0 once there are `min_heal_jobs` of them, otherwise the timing file, otherwise any finished heal job |
| `timing_file` | results/final/w0_timing.json | written by `bash scripts/fp_vm_w0.sh timing`: `hours_per_unit` (wall hours of one heal-unit per heal while 3 heals run at once) and `concurrency` (3) |
| `hours_per_unit`, `concurrency` | none | set both to use fixed numbers (`manual`) |
| `min_heal_jobs` | 3 | see `source` |

Projection: `used VM hours + remaining heal-units x hours_per_unit / concurrency + est_hours of jobs with
overlap: false + reserve_hours`. Remaining heal-units count every heal job not settled (a running heal by its
remaining share); jobs waiting for an unresolved `when` file count (conservative); jobs whose condition is false do
not. Measured from jobs: hours_per_unit = sum of heal wall hours / sum of their heal-units, concurrency = sum of
heal wall hours / length of the union of their run intervals.

## 4. Jobs

| Key | Required | Meaning |
|---|---|---|
| `id` | yes | unique; letters, digits, `_`, `.`, `-`. The job's key: editing `command` never starts a second copy |
| `wave` | yes | one of `settings.waves` |
| `kind` | yes | `heal`, `eval`, `light`, `cpu`, `decision` or `report` |
| `command` | yes | run with bash from the repository root, output appended to `logs/final/<id>.log` |
| `done` | yes (not for services) | the done marker: a file that exists when the job succeeded. Success = exit code 0 and the marker exists. A marker present before the first start means "done already" (the job is skipped) |
| `deps` | no | hard dependencies (ids in the same or an earlier wave): the job starts after they finished; if one is cut it is cut, if one is not needed it is not needed, if one failed it is blocked |
| `after` | no | ordering only: ids or earlier wave names that must be settled in any way (the final report uses `after: [W5]`) |
| `gpu_mem_gb` | heal: yes | estimated peak GPU memory; 0 means a CPU job |
| `host_mem_gb` | no | host memory the job needs (checked against MemAvailable) |
| `priority` | no (0) | higher starts first |
| `heal_units` | heal: yes | heal-units of the job for the projection (Section 9: one DINOv2-B heal of 780 steps = 1; DINOv2-L about 3; an A11 extended heal about 3; a 200-step LR check about 0.25; a multi-student process = the sum over its students) |
| `cut_rank` | no (never_cut) | Section 9 cut order, an integer from 1 (cut first), or `never_cut`. A never-cut job must not have a hard dependency on a cuttable job (use `after`) |
| `when` | no | one condition or a list (all must hold), Section 5; false makes the job and its dependents "not needed" |
| `est_hours`, `overlap` | no (0, true) | wall hours of a non-heal job, counted in the projection only with `overlap: false` (work that cannot run beside heals) |
| `locks` | no | at most one running job per lock name |
| `exclusive` | no | starts only when no other GPU job runs, and no GPU job starts while it runs (timing, A12 throughput) |
| `service` | no | a long-running job (the evaluation server): needs no done marker, is stopped with SIGTERM once every other job is settled (then counts as finished) |
| `timeout_hours` | no | SIGTERM after this long (a failure) |
| `env` | no | extra environment for this job |
| `note` | no | free text |

Every job receives `FP_JOB_ID`, `FP_JOB_WAVE`, `FP_JOB_ATTEMPT` and `FP_RUNNER_STATE` in its environment.

Section 9 cut order as ranks (the coordinator assigns them; dependents of a cut job are cut with it):

| `cut_rank` | Jobs |
|---|---|
| 1 | A15b and A15a |
| 2 | DINOv2-L (heals, its structure fit if separate, its evaluations) |
| 3 | A11 extended healing for SVD-LLM |
| 4 | A11 extended healing for Mirror |
| 5 | the extra seeds at 0.50 in A06 |
| 6 | the second SliceGPT seed and the third IO-SVD seed |
| 7 | the DeiT random-structure control and the third DeiT seeds |
| never_cut | Gate P and the A01 decision; A02, A03, A04; the main A05 arms (2 seeds each); A06 stacking at 1 seed; DeiT-III-B with 2 seeds per arm; A09, A12, A13, A10 for the main models; W0; the report |

When a multi-student process mixes cuttable and never-cut students, split it into two jobs, or give it the rank of
its most protected student (never_cut).

## 5. Conditions (`when`, `settings.stop_when`)

```yaml
when: {file: results/A01/decision.json, key: adopted, equals: true}
stop_when:
  - {file: results/A01/gate_p.json, key: pass, equals: false, reason: "Gate P failed (FINAL_PHASE_PLAN.md Section 6)"}
```

`file` is a JSON file; `key` a dotted path (`a.b.c`); one operator: `equals`, `not_equals`, `in` (a list),
`truthy`, `falsy`, or `exists` (no key: the file exists). While the file or key is missing, a `when` condition waits
and a `stop_when` condition does not hold. Decision jobs (Gate P, the A01 rule) write these files; give the decision
job the file as its done marker.

## 6. Job states

`ready`, `waiting` (dependency, barrier, `after`, or a `when` file), `running`, `finished` (exit 0 and done marker),
`done` (marker present before any start), `failed` (failed `max_failures` times: a stop event), `gave_up` (lost
`max_lost` times), `cut`, `skipped` (by hand), `not_needed` (condition false, or a dependency not run), `blocked`
(a dependency failed), `budget` (not started because of the 13.5-hour rule or the ceiling), `terminated` (stopped at
the ceiling). The reason for every waiting or held job is in `fp_runner.py status` and in STATUS.md.

## 7. Files

- `logs/final/state/<id>.start` (attempt, time, command, GPU GB), `<id>.cmd`, `<id>.pid` ("pid starttime", written by
  the job's wrapper), `<id>.rc` (exit code, written by the wrapper last), `<id>.json` (every attempt), `<id>.term`
  (signals sent), `<id>.skip` (created by hand: never start it).
- `budget.json` (VM clock: start of W0, counted seconds, pauses left out), `cuts.json`, `projections.json`,
  `stop.json`, `events.jsonl`, `status.json`, `runner.lock`.
- `logs/final/<id>.log`, `logs/final/runner.log`, `logs/final/gpu.csv` (nvidia-smi every 30 s).
- `results/STATUS.md`: the runner writes only between `<!-- fp_runner:begin ... -->` and `<!-- fp_runner:end -->`
  (inserted after the title line if missing); everything else in the file is kept.
- `notes/decisions.md`: one appended table row per cut.

VM clock: VM time counts from `bash scripts/fp_vm_w0.sh clock` (or the runner's first start). When the VM was paused
(the machine or container starts again, seen from the start time of process 1), the gap is left out; time when the
runner was merely stopped is counted. `fp_runner.py clock --set-start <UTC>` or `--adjust-hours H` corrects it by
hand (for example against the bill); `fp_runner.py heartbeat` keeps it current while no runner runs.

## 8. Recovery

- A job failed twice: read `logs/final/<id>.log`, fix, then `fp_runner.py reset <manifest> <id>` and
  `fp_runner.py clear-stop <manifest>`, and start the runner again.
- Stop one running job for good: `kill -TERM -<pid>` (pid in `logs/final/state/<id>.pid`), then
  `touch logs/final/state/<id>.skip`.
- Run a finished job again: delete its done marker and run `fp_runner.py reset <manifest> <id>`.

## 9. Example

```yaml
version: 1
settings:
  env: {MIRROR_THREADS: "4", OMP_NUM_THREADS: "4", MKL_NUM_THREADS: "4"}
  stop_when:
    - {file: results/A01/gate_p.json, key: pass, equals: false, reason: "Gate P failed"}
jobs:
  - {id: w0_models, wave: W0, kind: cpu, command: bash scripts/fp_vm_w0.sh models, done: logs/final/w0/models.done, priority: 100}
  - {id: w0_data, wave: W0, kind: cpu, command: bash scripts/fp_vm_w0.sh data, done: logs/final/w0/data.done, priority: 100}
  - {id: w0_roles, wave: W0, kind: cpu, command: bash scripts/fp_vm_w0.sh roles, deps: [w0_data], done: logs/final/w0/roles.done}
  - {id: w0_tests, wave: W0, kind: cpu, command: bash scripts/fp_vm_w0.sh tests, done: logs/final/w0/tests.done}
  - {id: w0_smoke, wave: W0, kind: light, gpu_mem_gb: 20, command: bash scripts/fp_vm_w0.sh smoke, deps: [w0_models], done: logs/final/w0/smoke.done}
  - {id: w0_timing, wave: W0, kind: light, exclusive: true, gpu_mem_gb: 100, command: bash scripts/fp_vm_w0.sh timing,
     deps: [w0_smoke, w0_roles], done: results/final/w0_timing.json}
  - {id: w0_roles_ext, wave: W1, kind: cpu, host_mem_gb: 8, command: bash scripts/fp_vm_w0.sh roles_ext, done: logs/final/w0/roles_ext.done}
  - {id: eval_server, wave: W1, kind: eval, service: true, gpu_mem_gb: 25, priority: 90, command: python scripts/fp_eval_server.py}
  - id: gate_dinov2b_s0
    wave: W1
    kind: heal
    command: python scripts/fp_heal.py ...           # the heal agent's flags
    gpu_mem_gb: 40
    heal_units: 2                                     # Mirror and SVD-LLM students in one process
    priority: 80
    done: runs/final/A01/dinov2_vitb14_reg/s0/DONE   # whatever the heal script writes last
  - {id: gate_p, wave: W1, kind: decision, command: python scripts/fp_gate_p.py, deps: [gate_dinov2b_s0], priority: 95,
     done: results/A01/gate_p.json}                  # the coordinator's decision scripts write these files
  - {id: a01_decision, wave: W1, kind: decision, command: python scripts/fp_a01_decide.py, deps: [gate_p], priority: 95,
     done: results/A01/decision.json}
  - id: a01_aw_s2
    wave: W2
    kind: heal
    command: python scripts/fp_heal.py ...
    deps: [a01_decision]
    when: {file: results/A01/decision.json, key: adopted, equals: true}
    gpu_mem_gb: 35
    heal_units: 1
    done: runs/final/A01/dinov2_vitb14_reg/mirror_aw/s2/DONE
  - {id: a15b_probe, wave: W5, kind: heal, cut_rank: 1, gpu_mem_gb: 35, heal_units: 1, command: python scripts/fp_heal.py ...,
     done: runs/final/A15b/DONE}
  - {id: report, wave: End, kind: report, after: [W5], command: python scripts/collect_next_phase.py, done: results/FINAL_PHASE_REPORT.md}
```

`scripts/fp_vm_w0.sh` steps `clock`, `env` and `prereg` run by hand before the runner (the runner needs the venv and
the pre-registration). The other W0 steps can run by hand or as W0 jobs as above; each writes
`logs/final/w0/<step>.done`, so a step run by hand is "done" for the runner too when the job's done marker is that file.
