"""`python -m rlvr_lean.gpu <step> --out <file>`: run one GPU step and write its JSON result."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


SOUNDNESS_ALARM_EXIT = 3        # Lean verified a proof of both sides of one statement: not a failed step, a stop (the entry shim reads it)


def main() -> int:
    from rlvr_lean.domain.problem_pool.selection import SoundnessAlarm
    from rlvr_lean.gpu import (
        diagnostics,
        ladder_ceiling,
        ladder_dose,
        ladder_l2,
        ladder_l3a,
        ladder_l3a2,
        ladder_l3c,
        ladder_l3d1,
        ladder_l3d2,
        ladder_l4,
        ladder_loop,
        ladder_round,
        milestone2,
        pipeline,
    )

    steps = {**milestone2.STEPS, **pipeline.STAGES, **diagnostics.STEPS, **ladder_loop.STEPS, **ladder_round.STEPS, **ladder_dose.STEPS, **ladder_l2.STEPS,
             **ladder_l3a.STEPS, **ladder_l3a2.STEPS, **ladder_l3c.STEPS, **ladder_ceiling.STEPS, **ladder_l3d1.STEPS, **ladder_l3d2.STEPS,
             **ladder_l4.STEPS}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=sorted(steps))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "config" / "experiment.yaml")
    arguments = parser.parse_args()
    started = time.monotonic()
    code = 1
    try:
        config = milestone2.load_config(arguments.config)
        pipeline.use_lean_pin(config)       # the config's `lean.pin`: the run directory follows it (the OEIS Open spec, O2a)
        result = steps[arguments.step](config)
        result["ok"] = bool(result.get("ok", True))
    except SoundnessAlarm as alarm:         # spec ladder-loop, fixture 10: the round stops, with a code of its own
        result = {"ok": False, "soundness_alarm": str(alarm), "error": repr(alarm), "traceback": traceback.format_exc()[-4000:]}
        code = SOUNDNESS_ALARM_EXIT
    except Exception as error:  # noqa: BLE001 - the step's failure IS its result, recorded and returned non-zero
        result = {"ok": False, "error": repr(error), "traceback": traceback.format_exc()[-4000:]}
    result["step_seconds"] = round(time.monotonic() - started, 1)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({key: value for key, value in result.items() if key != "traceback"}, default=str)[:2000], flush=True)
    return 0 if result["ok"] else code


if __name__ == "__main__":
    code = main()
    if code != 0:
        # A failed step ends here. A normal exit would first wait for the background verification thread to
        # finish checks whose results are thrown away, and for a dead engine's shutdown: on 2026-10-03 that
        # kept the box and the Lean server busy for 31 minutes after the failure (task 83efeb13).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    sys.exit(code)
