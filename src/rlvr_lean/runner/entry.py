"""The stage runner: the box-side entry point for one rlvr_lean stage. Spec §1 (topology and job-runner
integration).

Runs on the interpreter that starts it with the STANDARD LIBRARY ONLY, and installs nothing into it. It:
  1. installs a pinned `uv` into the persistent store (checksum-verified),
  2. builds the project's environments there from the repository's `uv.lock` (`gpu`, `quantize`),
  3. runs each step of the requested stage as its OWN process in the right environment, so GPU memory is
     released between steps, with a GPU guard in between (used memory back to the task's baseline),
  4. writes a TensorBoard heartbeat to `<out>/tb/` every minute (a job runner that stops a silent task, ours
     after 90 minutes, sees that the stage is alive), and samples card memory and the step's process-tree
     memory so peaks can be reported,
  5. writes `<out>/<stage>.json` (the completion artifact) and per-step JSON beside it.

Every step is idempotent: it skips work whose outputs already exist in the store, so a rerun resumes.

    python -m rlvr_lean.runner.entry --stage m2 --kimina-api-key KEY --out OUT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
from pathlib import Path

from rlvr_lean.runner.heartbeat import ScalarEventWriter

UV_VERSION = "0.12.15"
UV_ARCHIVE = "uv-x86_64-unknown-linux-gnu.tar.gz"
REPO_ROOT = Path(__file__).resolve().parents[3]          # the task's repo directory (cwd for every child)
PROJECT_DIR = REPO_ROOT                                  # the `uv` project: pyproject.toml and uv.lock
PACKAGE_DIR = REPO_ROOT / "src" / "rlvr_lean"            # the package: its data fixtures
STORE = Path(os.environ.get("RLVR_LEAN_STORE", "/root/rlvr_lean_store"))

# (environment, step) in order; ("guard", None) waits for the GPU to be idle again.
STAGES: dict[str, list[tuple[str, str | None]]] = {
    "m2": [
        ("sync", "gpu"), ("sync", "quantize"),
        ("gpu", "download"),
        ("guard", None), ("quantize", "quantize_fp8"),
        ("guard", None), ("gpu", "quantize_nf4"),
        ("gpu", "fix_tokenizers"),
        ("guard", None), ("gpu", "sample_smoke"),
        ("guard", None), ("gpu", "qlora_step"),
        ("guard", None), ("gpu", "adapter_check"),
        ("guard", None),
    ],
    # Phase A (spec §2–§8). The first six steps are Milestone 2's and are no-ops once their outputs exist;
    # they stay so a fresh box can run this stage on its own. The profile picks the sizes.
    "phase_a": [
        ("sync", "gpu"), ("sync", "quantize"),
        ("gpu", "download"),
        ("guard", None), ("quantize", "quantize_fp8"),
        ("guard", None), ("gpu", "quantize_nf4"),
        ("gpu", "fix_tokenizers"),
        ("gpu", "prepare_data"),
        ("guard", None), ("gpu", "generate_conjectures"),
        ("guard", None), ("gpu", "sample_proofs"),
        ("guard", None), ("gpu", "score_selection"),
        ("guard", None), ("gpu", "train_adapter"),
        ("guard", None), ("gpu", "evaluate_sampling"),
        ("guard", None), ("gpu", "evaluate_loss"),
        ("guard", None), ("gpu", "report"),
    ],
}
# Phase B (spec §13) runs the same steps over several arms (`--arms`): everything up to scoring is a no-op on
# Phase A's store, and the reused arm's training and evaluation are skipped by their done markers.
STAGES["phase_b"] = list(STAGES["phase_a"])
# Diagnostics (the diagnosis of the held-out loss's saturation and of the learning-progress score): one read-only
# step each against a finished run's store (`--profile full`), which already holds the models, the data and the trained adapters.
STAGES["diagnose_tokens"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_tokens"), ("guard", None)]
STAGES["diagnose_gradients"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_gradients"),
                                ("guard", None), ("gpu", "diagnose_gradients_second"), ("guard", None)]
STAGES["diagnose_native_round"] = [("sync", "gpu"), ("guard", None), ("gpu", "diagnose_native_round"), ("guard", None)]

# The ladder's first rung and what goes with it, as ONE task (spec §13b):
#   part 1  §13b: one pass on the 50% band in the native format, the fresh-sample probes (base and adapter), the
#           standard evaluation, the census of false conjectures, the report with its branch;
#   part 2  the escalation §13a's scout earned: `native_same_picks` seeds 1 and 2, and the read over three seeds;
#   part 3  last and optional: the base model's reach on the workbook problems it never proved.
# A step may carry options: its own `arms` and `seeds` (the task's arguments otherwise), a `label` for its
# result file when a step runs twice, and its `part`. A failure ends its own part and the parts that need it
# (`needs`); every finished step's results are already in the store and stay there.
_LADDER = {"arms": "ladder_half_native", "seeds": "0", "part": 1}
_SCOUT_SEEDS = {"arms": "native_same_picks", "seeds": "0,1,2", "part": 2}
_REACH = {"part": 3, "needs": (1, 2)}
STAGES["ladder_13b"] = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("guard", None, _LADDER), ("gpu", "train_adapter", {**_LADDER, "label": "train_adapter_ladder"}),
    ("guard", None, _LADDER), ("gpu", "ladder_probes", _LADDER),
    ("guard", None, _LADDER), ("gpu", "evaluate_sampling", {**_LADDER, "label": "evaluate_sampling_ladder"}),
    ("guard", None, _LADDER), ("gpu", "evaluate_loss", {**_LADDER, "label": "evaluate_loss_ladder"}),
    ("guard", None, _LADDER), ("gpu", "negation_census", _LADDER),
    ("guard", None, _LADDER), ("gpu", "report", {**_LADDER, "label": "report_ladder"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "train_adapter", {**_SCOUT_SEEDS, "label": "train_adapter_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "evaluate_sampling", {**_SCOUT_SEEDS, "label": "evaluate_sampling_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "evaluate_loss", {**_SCOUT_SEEDS, "label": "evaluate_loss_scout"}),
    ("guard", None, _SCOUT_SEEDS), ("gpu", "report", {**_SCOUT_SEEDS, "label": "report_scout"}),
    ("guard", None, _REACH), ("gpu", "base_reach", _REACH),
    ("guard", None, _REACH),
]


# The ladder loop's L0, the GPU half (docs/spec/ladder-loop.spec.md; `gpu/ladder_loop.py`): the base model's
# episodes on the held-out set H and on the base-map sample of the pool, both read from the package's data
# directory (they travel with the code), at the ladder loop's own Lean pin. No parts: a failure, or a
# soundness alarm (a step's exit code 3), ends the stage; every block that finished is in the box's store.
_LADDER_L0B = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("gpu", "ladder_prepare"),
    ("guard", None), ("gpu", "ladder_episodes_heldout"),
    ("guard", None), ("gpu", "ladder_episodes_base_map"),
    ("guard", None), ("gpu", "ladder_l0_report"),
]
STAGES["ladder_l0b"] = list(_LADDER_L0B)
# The same steps on the ten-statement fixture, in a run directory of its own: the ONE small task that goes to a
# terminal state before the real one is queued. It is the first stage run at Lean v4.27 (the OEIS Open spec, O2a 9g).
_LADDER_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                 "RLVR_LEAN_LADDER_RUN": "ladder_l0_smoke"}}
STAGES["ladder_l0b_smoke"] = [(environment, step, _LADDER_SMOKE) for environment, step in _LADDER_L0B]

# The ladder loop's L1 (spec ladder-loop, "A round" and "L1's read"; `gpu/ladder_round.py`): ONE round, two arms that
# differ only in who chooses the round's problems (the challenger, or a uniform draw of the same candidates), on one
# box, one after the other. Shared first: the data check, the statement embeddings, both arms' proposals, the base's
# fresh episodes on G and on the rungs. Then each arm: its round (n episodes of the base), one training pass, the
# trained model's measurements. One seed per task (`--seeds 0`): its run directory is the seed's. No parts: a
# failure, or a soundness alarm, ends the stage; every finished step and block is in the box's store and a rerun
# resumes there.
_LADDER_L1 = [
    ("sync", "gpu"), ("gpu", "fix_tokenizers"),
    ("gpu", "ladder_l1_prepare"),
    ("guard", None), ("gpu", "ladder_l1_embed"),
    ("guard", None), ("gpu", "ladder_l1_propose"),
    ("guard", None), ("gpu", "ladder_l1_base"),
]
for _arm in ("challenger", "random"):
    _LADDER_L1 += [("guard", None), ("gpu", f"ladder_l1_round_{_arm}"),
                   ("guard", None), ("gpu", f"ladder_l1_train_{_arm}"),
                   ("guard", None), ("gpu", f"ladder_l1_measure_{_arm}")]
_LADDER_L1 += [("guard", None), ("gpu", "ladder_l1_report")]
STAGES["ladder_l1"] = list(_LADDER_L1)
# The same steps on the fixtures (L0's ten statements, L1's twelve candidates and made-up base results), in a run
# directory of its own: the ONE small task that goes to a terminal state before the real one is queued. It is the
# first run of the embedding step, of a native-format adapter served beside the base, and of the report.
_LADDER_L1_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                    "RLVR_LEAN_LADDER_ROUND_DATA": str(PACKAGE_DIR / "data" / "ladder_l1_fixture"),
                                    "RLVR_LEAN_LADDER_ROUND_RUN": "ladder_l1_smoke"}}
STAGES["ladder_l1_smoke"] = [(environment, step, _LADDER_L1_SMOKE) for environment, step in _LADDER_L1]

# The ladder loop's L1b, the dose curve (spec ladder-loop, "L1b: the dose curve"; `gpu/ladder_dose.py`): one seed's
# challenger arm again, from the base, on the training proofs L1 stored, for three passes, with the loss read over
# time and the held-out rungs measured at five checkpoints. It READS the L1 run directory of its seed in the box's
# store and writes a run directory of its own. One seed per task (`--seeds 0`). No parts: a failure, or a soundness
# alarm, ends the stage; a rerun resumes at the first step (and, inside a measurement, the first block) not done.
_LADDER_L1B = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l1b_prepare"), ("guard", None), ("gpu", "ladder_l1b_train")]
for _checkpoint in ("p050", "p100", "p150", "p200", "p300"):
    _LADDER_L1B += [("guard", None), ("gpu", f"ladder_l1b_measure_{_checkpoint}")]
_LADDER_L1B += [("guard", None), ("gpu", "ladder_l1b_report")]
STAGES["ladder_l1b"] = list(_LADDER_L1B)
# The same steps on what the L1 SMOKE run left in the box's store (`ladder_l1_smoke`: a dozen training proofs, five
# rung problems), in a run directory of its own.
_LADDER_L1B_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DOSE_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_DOSE_RUN": "ladder_l1b_smoke"}}
STAGES["ladder_l1b_smoke"] = [(environment, step, _LADDER_L1B_SMOKE) for environment, step in _LADDER_L1B]

# The ladder loop's L2 (spec ladder-loop, "L2: three rounds"; `gpu/ladder_l2.py`): three rounds of the challenger arm at
# one seed, then the equal-compute control. Round r starts from M(r - 1) and trains M(r) from the base on every round's
# training set so far; M(0) is the base. A round's 1,000 proposals are made in four batches inside ONE process (the
# challenger is refitted after each batch's episodes), then one training pass, then M(r) on the held-out rungs and on G.
# It READS the L1 run directory of its seed in the box's store (G, the rungs, the base's fresh results) and writes a run
# directory of its own; the statement embeddings are stored once per box, beside the runs, and another seed's task
# reuses them. One seed per task (`--seeds 0`). No parts: a failure, or a soundness alarm, ends the stage; a rerun resumes
# at the first step, batch and block not done. A round whose below-band interval lies entirely below zero stops the
# loop: the steps after it run nothing and the report says so.
_LADDER_L2 = [("sync", "gpu"), ("gpu", "fix_tokenizers"), ("gpu", "ladder_l2_prepare"), ("guard", None), ("gpu", "ladder_l2_embed")]
for _round in (1, 2, 3):
    _LADDER_L2 += [("guard", None), ("gpu", f"ladder_l2_round_{_round}"),
                   ("guard", None), ("gpu", f"ladder_l2_train_{_round}"),
                   ("guard", None), ("gpu", f"ladder_l2_measure_{_round}")]
# After the control, the LAST model's own extra attempts on G (added 2026-10-05, after seeds 0 and 1): the control's
# number of episodes with the control's sampling seed, so M(3) and the base are compared at equal attempts. A run that
# finished before this step existed gets it alone when its task is queued again: every step before it returns what is
# stored, and the report is built again with the new part.
_LADDER_L2 += [("guard", None), ("gpu", "ladder_l2_control"), ("guard", None), ("gpu", "ladder_l2_control_trained"),
               ("guard", None), ("gpu", "ladder_l2_report")]
STAGES["ladder_l2"] = list(_LADDER_L2)
# The same steps on the fixtures (L0's ten statements; L1's twelve candidates as the whole pool), reading what the L1
# SMOKE run left in the box's store (`ladder_l1_smoke`), in a run directory of its own. Tiny sizes: three rounds of four
# problems, in two batches of two, use the twelve candidates up.
_LADDER_L2_SMOKE = {"environment": {"RLVR_LEAN_LADDER_DATA": str(PACKAGE_DIR / "data" / "ladder_l0_fixture"),
                                    "RLVR_LEAN_LADDER_L2_DATA": str(PACKAGE_DIR / "data" / "ladder_l1_fixture"),
                                    "RLVR_LEAN_LADDER_L2_SOURCE": "ladder_l1_smoke", "RLVR_LEAN_LADDER_L2_RUN": "ladder_l2_smoke",
                                    "RLVR_LEAN_LADDER_L2_PROBLEMS": "4", "RLVR_LEAN_LADDER_L2_BATCHES": "2"}}
STAGES["ladder_l2_smoke"] = [(environment, step, _LADDER_L2_SMOKE) for environment, step in _LADDER_L2]
SOUNDNESS_ALARM_EXIT = 3        # `gpu.__main__`'s code for a proof of both sides of one statement (it cannot be imported here)


def step_fields(entry_step: tuple) -> tuple[str, str | None, dict]:
    """(environment, step, options) of one stage entry; the options are empty for a plain (environment, step)."""
    return entry_step[0], entry_step[1], (dict(entry_step[2]) if len(entry_step) > 2 else {})


def part_is_blocked(options: dict, failed_parts: set[int]) -> bool:
    """A step is skipped when its own part has failed, or a part it needs has."""
    return options.get("part", 0) in failed_parts or any(part in failed_parts for part in options.get("needs", ()))


def log(message: str) -> None:
    print(f"[rlvr_lean.entry {time.strftime('%H:%M:%S')}] {message}", flush=True)


def gpu_memory_used_mib() -> int | None:
    try:
        output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=20, check=True).stdout
        return int(output.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _proportional_set_kib(proc_entry: Path, status_fields: dict[str, str]) -> int:
    """Proportional set size: each shared page is split between the processes that map it, so a tree's total
    is real memory. Plain RSS counts shared pages once per process; vLLM's processes share most of theirs,
    and summed RSS read 33 GB on a 30 GB box (Milestone 3 smoke run)."""
    try:
        for line in (proc_entry / "smaps_rollup").read_text().splitlines():
            if line.startswith("Pss:"):
                return int(line.split()[1])
    except OSError:
        pass
    return int(status_fields.get("VmRSS", "0 kB").split()[0]) if "VmRSS" in status_fields else 0


def process_tree_rss_mib(root_pid: int) -> int:
    """Memory of `root_pid` and all its descendants (vLLM runs its engine in a child process), as PSS."""
    parents: dict[int, int] = {}
    fields_by_pid: dict[int, dict[str, str]] = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue
        fields = dict(line.split(":", 1) for line in status.splitlines() if ":" in line)
        pid = int(entry.name)
        parents[pid] = int(fields.get("PPid", "0").strip() or 0)
        fields_by_pid[pid] = fields
    total, frontier = 0, [root_pid]
    while frontier:                      # only the tree's own processes are measured
        pid = frontier.pop()
        if pid in fields_by_pid:
            total += _proportional_set_kib(Path("/proc") / str(pid), fields_by_pid[pid])
        frontier.extend(child for child, parent in parents.items() if parent == pid)
    return total // 1024


class Monitor:
    """Background sampler: heartbeat events every 60 s, memory samples every 5 s."""

    def __init__(self, out: Path) -> None:
        self.writer = ScalarEventWriter(out / "tb")
        self.started = time.monotonic()
        self.child_pid: int | None = None
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def reset_peaks(self) -> None:
        self.peak_gpu_mib = 0
        self.peak_tree_rss_mib = 0

    def _run(self) -> None:
        last_heartbeat = 0.0
        while not self._stop.is_set():
            used = gpu_memory_used_mib()
            if used is not None:
                self.peak_gpu_mib = max(self.peak_gpu_mib, used)
            if self.child_pid is not None:
                self.peak_tree_rss_mib = max(self.peak_tree_rss_mib, process_tree_rss_mib(self.child_pid))
            now = time.monotonic()
            if now - last_heartbeat >= 60:
                last_heartbeat = now
                self.writer.scalars(int(now - self.started), {
                    "heartbeat/elapsed_minutes": (now - self.started) / 60,
                    "gpu/used_mib": float(used or 0),
                    "memory/step_tree_rss_mib": float(self.peak_tree_rss_mib),
                })
            self._stop.wait(5)


def ensure_uv() -> Path:
    uv = STORE / "bin" / f"uv-{UV_VERSION}" / "uv"
    if uv.exists():
        return uv
    base = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
    archive = STORE / "downloads" / UV_ARCHIVE
    archive.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(base + UV_ARCHIVE, archive)
    expected = urllib.request.urlopen(base + UV_ARCHIVE + ".sha256", timeout=60).read().decode().split()[0]
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"uv archive checksum mismatch: {actual} != {expected}")
    with tarfile.open(archive) as bundle:
        member = next(m for m in bundle.getmembers() if m.name.endswith("/uv"))
        member.name = "uv"
        bundle.extract(member, uv.parent, filter="data")
    uv.chmod(0o755)
    return uv


def environment_python(group: str) -> Path:
    return STORE / "envs" / group / "bin" / "python"


PROFILE = {"value": "smoke", "seeds": "", "arms": ""}      # set from --profile / --seeds / --arms in main(); read by every child step


def child_environment(group: str | None, kimina_api_key: str | None, options: dict | None = None) -> dict[str, str]:
    options = options or {}
    env = dict(os.environ)
    env.update({
        "RLVR_LEAN_PROFILE": PROFILE["value"],
        "RLVR_LEAN_TRAINING_SEEDS": options.get("seeds", PROFILE["seeds"]),      # a step's own, else the task's
        "RLVR_LEAN_ARMS": options.get("arms", PROFILE["arms"]),
        "PYTHONPATH": str(REPO_ROOT / "src"),
        "RLVR_LEAN_STORE": str(STORE),
        "HF_HOME": str(STORE / "hf"),
        "UV_CACHE_DIR": str(STORE / "uv-cache"),
        "UV_PYTHON_INSTALL_DIR": str(STORE / "python"),
        # A job runner may pin every task to one thread (ours does); GPU steps need their own (vLLM tokenizers, data loading).
        "OMP_NUM_THREADS": os.environ.get("RLVR_LEAN_THREADS", "8"),
        "MKL_NUM_THREADS": os.environ.get("RLVR_LEAN_THREADS", "8"),
        "TOKENIZERS_PARALLELISM": "false",
    })
    if group is not None:
        env["UV_PROJECT_ENVIRONMENT"] = str(STORE / "envs" / group)
    if kimina_api_key:
        env["RLVR_LEAN_KIMINA_API_KEY"] = kimina_api_key
    env.update(options.get("environment", {}))          # a stage's own variables for its steps (the smoke run's data)
    return env


def run_child(command: list[str], env: dict[str, str], monitor: Monitor) -> int:
    log("run: " + " ".join(command[:6]) + (" ..." if len(command) > 6 else ""))
    process = subprocess.Popen(command, cwd=REPO_ROOT, env=env)     # cwd inside the repo: a leaked process can be found by it
    monitor.child_pid = process.pid
    try:
        return process.wait()
    finally:
        monitor.child_pid = None


GUARD_RELEASE_MIB = 6000          # a step that held a model gives back at least this much (the engine holds over 11 GiB)


def wait_for_idle_gpu(baseline_mib: int | None, margin_mib: int, timeout_seconds: int = 120,
                      step_peak_mib: int | None = None, release_mib: int = GUARD_RELEASE_MIB) -> dict:
    """The GPU guard: the step before it must have given its memory back. Either card-wide used memory is back
    at the baseline plus a margin, or it has fallen at least `release_mib` below that step's own peak. The
    second rule is for the card's OTHER users: the desktop session may hold more than it did when the task
    started (ladder_l0b_r1 failed here with both attempt steps done: 485 MiB of desktop growth against a 300
    MiB margin), and that is not this task's to wait for. A model that lingers leaves the card near the peak
    and fails both rules, as before."""
    if baseline_mib is None:
        return {"ok": True, "detail": "nvidia-smi unavailable; guard skipped"}

    def passed(used: int | None) -> str | None:
        if used is None:
            return None
        if used <= baseline_mib + margin_mib:
            return "back at the baseline"
        if step_peak_mib is not None and step_peak_mib - used >= release_mib:
            return "the step's memory was released; the card's other users hold more than at the baseline"
        return None

    deadline = time.monotonic() + timeout_seconds
    used = gpu_memory_used_mib()
    while used is not None and passed(used) is None and time.monotonic() < deadline:
        time.sleep(3)
        used = gpu_memory_used_mib()
    why = passed(used)
    result = {"ok": why is not None, "used_mib": used, "baseline_mib": baseline_mib, "margin_mib": margin_mib,
              "step_peak_mib": step_peak_mib}
    if why:
        result["passed_because"] = why
    return result


def run_stage(stage: str, out: Path, kimina_api_key: str | None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(exist_ok=True)
    STORE.mkdir(parents=True, exist_ok=True)
    monitor = Monitor(out)
    monitor.start()
    baseline = gpu_memory_used_mib()
    previous_peak: int | None = None
    report: dict = {"stage": stage, "box": platform.node(), "gpu_baseline_mib": baseline, "steps": []}
    log(f"stage {stage}; store {STORE}; GPU memory in use at start: {baseline} MiB")
    try:
        uv = ensure_uv()
        failed_parts: set[int] = set()
        for entry_step in STAGES[stage]:
            environment, step, options = step_fields(entry_step)
            label = options.get("label", step)
            if part_is_blocked(options, failed_parts):
                report["steps"].append({"step": label or "gpu_guard", "skipped": f"part {options.get('part', 0)}: an earlier step it needs failed"})
                continue
            started = time.monotonic()
            monitor.reset_peaks()
            if environment == "guard":
                result = {"step": "gpu_guard", **wait_for_idle_gpu(baseline, 300, step_peak_mib=previous_peak)}
                returncode = 0 if result["ok"] else 1
                if result["ok"] and result.get("passed_because", "").startswith("the step's memory was released"):
                    baseline = result["used_mib"]       # what the card's other users hold now: the next guard's baseline
            elif environment == "sync":
                command = [str(uv), "sync", "--project", str(PROJECT_DIR), "--group", step, "--no-dev",
                           "--frozen", "--python", "3.12"]
                returncode = run_child(command, child_environment(step, None), monitor)
                result = {"step": f"sync_{step}"}
            else:
                step_out = out / "steps" / f"{label}.json"
                command = [str(environment_python(environment)), "-m", "rlvr_lean.gpu", step, "--out", str(step_out)]
                env = child_environment(environment, kimina_api_key, options)
                env["RLVR_LEAN_STEP_DIR"] = str(out / "steps")      # side files (samples) land in the output directory a job runner collects
                env["RLVR_LEAN_TB_DIR"] = str(out / "tb")           # one TensorBoard run per arm and seed (spec §13)
                returncode = run_child(command, env, monitor)
                result = {"step": label}
                if step_out.exists():
                    result["result"] = json.loads(step_out.read_text())
            result.update({"returncode": returncode, "seconds": round(time.monotonic() - started, 1),
                           "peak_gpu_used_mib": monitor.peak_gpu_mib, "peak_step_rss_mib": monitor.peak_tree_rss_mib})
            if environment not in ("guard", "sync"):
                previous_peak = monitor.peak_gpu_mib or None       # what the next guard holds this step to
            report["steps"].append(result)
            (out / f"{stage}.partial.json").write_text(json.dumps(report, indent=2))
            log(f"{result['step']}: returncode {returncode} in {result['seconds']} s")
            if returncode != 0:
                report.setdefault("failed_step", result["step"])
                report.setdefault("failed_steps", []).append(result["step"])
                if returncode == SOUNDNESS_ALARM_EXIT and environment not in ("guard", "sync"):
                    report["soundness_alarm"] = result["step"]      # not an ordinary failure: nothing may be read or rerun
                if "part" not in options:           # a stage without parts, or the setup before them: stop here
                    break
                failed_parts.add(options["part"])
    finally:
        monitor.stop()
    report["passed"] = "failed_step" not in report
    return report


def store_ca_certificate(encoded: str, store: Path) -> Path:
    """Write the Lean pool authority's certificate into the store and return its path. The OEIS Open spec, O2a item
    9b: it reaches a task the way the key does (an argument at queue time, never committed, never printed)
    and is public. Steps need a FILE (the TLS library reads one), named by its content so two pools' never
    collide. A value that is not base64 of a PEM certificate is refused without being echoed."""
    import base64
    import binascii

    try:
        pem = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("--kimina-ca-certificate must be base64 on one line (`base64 -w0 ca.crt`)") from None
    if b"-----BEGIN CERTIFICATE-----" not in pem:
        raise ValueError("--kimina-ca-certificate does not decode to a PEM certificate")
    target = store / "lean-ca" / f"{hashlib.sha256(pem).hexdigest()[:16]}.crt"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_bytes(pem)
    temporary.replace(target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--profile", default="smoke", choices=["smoke", "full"], help="sizes, from config/experiment.yaml")
    parser.add_argument("--seeds", default="", help="training seeds, e.g. 0 or 0,1,2; default: the config's list")
    parser.add_argument("--arms", default="", help="selections to train and evaluate, e.g. learning_progress_cosine,random; default: Phase A's")
    parser.add_argument("--kimina-api-key", default=None, help="handed to steps as an environment variable")
    parser.add_argument("--kimina-ca-certificate", default=None, metavar="BASE64",
                        help="the Lean pool authority's certificate (its PEM file, base64 on one line: `base64 -w0 ca.crt`); "
                             "passed at queue time beside the key, written into the store, never printed")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", default=None, help="accepted and ignored (a job runner may append one)")
    parser.add_argument("--print-run-identity", action="store_true")
    arguments = parser.parse_args()
    if arguments.print_run_identity:
        # What a job runner may record as this run's identity; the API key is never part of it.
        print(json.dumps({"config": {"tool": "rlvr_lean.runner.entry", "stage": arguments.stage, "profile": arguments.profile,
                                     "training_seeds": arguments.seeds, "arms": arguments.arms, "seed": None,
                                     "run": {"out_dir": str(arguments.out) if arguments.out else None}}}))
        return 0
    if arguments.out is None:
        parser.error("--out is required")
    if arguments.seeds and not all(part.isdigit() for part in arguments.seeds.split(",")):
        parser.error("--seeds must look like 0 or 0,1,2")
    # Format only: the names are checked by the pipeline (`domain.training.arms`), which this stdlib-only
    # shim cannot import without numpy (the package `__init__`s pull in the selection context).
    if arguments.arms and not all(part.replace("_", "").isalpha() for part in arguments.arms.split(",")):
        parser.error("--arms must look like learning_progress_cosine,random")
    PROFILE["value"], PROFILE["seeds"], PROFILE["arms"] = arguments.profile, arguments.seeds, arguments.arms
    if arguments.kimina_ca_certificate:
        try:
            # Every step inherits this process's environment (`child_environment`).
            os.environ["RLVR_LEAN_KIMINA_CA_FILE"] = str(store_ca_certificate(arguments.kimina_ca_certificate, STORE))
        except ValueError as error:
            parser.error(str(error))
    report = run_stage(arguments.stage, arguments.out, arguments.kimina_api_key)
    (arguments.out / f"{arguments.stage}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "failed_step": report.get("failed_step")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
