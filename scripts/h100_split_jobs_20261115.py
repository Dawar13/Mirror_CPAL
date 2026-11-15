"""Move not-yet-started critical-path jobs from job file A to a new job file A3 with its own slots (run on the VM).

Moved: the MUST-12 gate C measurements, every MUST-07/MUST-08 line (lock must07, still one at a time), and the MUST-09
heals other than Mirror-exact (random structure and plain SVD to lock heal2; gate-B heals to lock heal3). A line is
moved only if its job has never started in job file A (no .start file in A's state folder).
"""
import glob
import hashlib
import os
import sys

root = sys.argv[1]
A = os.path.join(root, "jobs", "h100_A_phase1.txt")
A3 = os.path.join(root, "jobs", "h100_A3_lanes.txt")
state = os.path.join(root, "logs", "h100_A_phase1", "state")

started_cmds = set()
for f in glob.glob(os.path.join(state, "*.start")):
    k = f[:-len(".start")]
    if os.path.exists(k + ".cmd"):
        started_cmds.add(open(k + ".cmd").read().strip())

lines = open(A).read().split("\n")
keep, move = [], []


def wanted(line: str):
    if line.startswith("#") or "|" not in line:
        return None
    cmd = line.split("|", 1)[1].strip()
    if "h100_measure_run.sh runs/MUST-12" in cmd:
        return ("gatec", line)
    if "%lock=must07" in line and "--structure-only" not in cmd:
        return ("must07", line)
    if "%lock=must07" in line and "dinov2_vitl14_reg" in cmd and "--structure-only" in cmd:
        return ("must07", line)
    if "h100_heal_gate.sh" in cmd:
        return ("heal3", line.replace("%lock=heal ", "%lock=heal3 "))
    if "--method random_structure --seed 0" in cmd or "--method svd --seed 0" in cmd:
        return ("heal2", line.replace("%lock=heal ", "%lock=heal2 "))
    return None


for line in lines:
    w = wanted(line)
    if w is None:
        keep.append(line)
        continue
    cmd = line.split("|", 1)[1].strip()
    if cmd in started_cmds:
        print("already started in A, kept there:", cmd[:100])
        keep.append(line)
        continue
    move.append(w)

order = {"gatec": 0, "must07": 1, "heal2": 2, "heal3": 3}
move.sort(key=lambda x: order[x[0]])
header = [
    "# H200 job file A3 (15 November): critical-path jobs moved out of job file A, whose PAR=5 slots were held by",
    "# long CPU-bound measurement jobs. Moved only if never started in A. Heal lanes: heal (Mirror-exact, job file A),",
    "# heal2 (random structure, plain SVD) and heal3 (gate-B baselines): up to three heals at once, approved by Dawar",
    "# on 15 November (the GPU is used by this project only).",
    "#   PAR=6 setsid nohup bash scripts/h100_sched.sh jobs/h100_A3_lanes.txt > logs/q_A3.log 2>&1 < /dev/null &",
]
open(A3, "w").write("\n".join(header + [l for _, l in move]) + "\n")
tmp = A + ".tmp"
open(tmp, "w").write("\n".join(keep))
os.replace(tmp, A)
print(f"moved {len(move)} lines to {A3}:")
for kind, l in move:
    print(" ", kind, l.split("|", 1)[1].strip()[:110])
