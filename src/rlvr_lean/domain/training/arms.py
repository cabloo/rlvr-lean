"""Names of everything one trained arm and seed produces (spec §13, "Artifact names"; §6 item 2a; §13a).

The arm Phase A trained keeps Phase A's names exactly, so its adapters, markers and evaluations are reused
without moving a file; every other arm is keyed `{arm}_seed{k}`. Base evaluations are shared by all arms
and are not named here, except the base's held-out loss, which depends on the pair format.

**A name fixes its format.** Phase A's and Phase B's arms were trained and measured in the legacy pair format
(no sequence-start token before the proof), and everything stored under their names is in it. Those arms
stay legacy, so a legacy name never comes to hold a native artifact; an arm in the native format has its own
name, and so does the base's held-out loss in that format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from rlvr_lean.domain.training.target_format import LEGACY, NATIVE, TARGET_FORMATS

LEGACY_ARMS = ("learning_progress_cosine", "learning_progress", "random", "difficulty_heuristic", "half_pass_rate")
# Spec §13a: Phase A's own examples (its learning-progress selection), trained in the native format.
SCOUT_ARM = "native_same_picks"
# Spec §13b: the `half_pass_rate` selection (the 50% band), native format, ONE pass over the data.
LADDER_ARM = "ladder_half_native"
LADDER_SELECTION = "half_pass_rate"
NATIVE_ARMS = (SCOUT_ARM, LADDER_ARM)
KNOWN_ARMS = LEGACY_ARMS + NATIVE_ARMS


@dataclass(frozen=True)
class ArmNames:
    arm: str
    seed: int
    key: str                 # suffix of every marker and file
    variant: str             # the `variant` of its evaluation attempts
    adapter_directory: str   # under `<run>/adapters/`
    tensorboard_run: str     # under `out/tb/`
    target_format: str = LEGACY   # the pair format it is trained and measured in
    selection_arm: str = ""       # the arm whose selection supplies its training examples ("" = its own)

    @property
    def train_marker(self) -> str:
        return f"train_adapter_{self.key}"

    @property
    def sampling_marker(self) -> str:
        return f"evaluate_sampling_{self.key}"

    @property
    def loss_marker(self) -> str:
        return f"evaluate_loss_{self.key}"

    @property
    def eval_attempts_file(self) -> str:
        return f"eval_attempts_{self.key}.jsonl"

    @property
    def eval_verification_file(self) -> str:
        return f"eval_verification_{self.key}.jsonl"

    @property
    def fp8_loss_file(self) -> str:
        return f"heldout_loss_fp8_{self.key}.jsonl"

    @property
    def nf4_loss_file(self) -> str:
        return f"heldout_loss_nf4_{self.key}.jsonl"

    @property
    def examples_arm(self) -> str:
        """The arm whose selection this one trains on."""
        return self.selection_arm or self.arm


def arm_names(arm: str, seed: int, reused_arm: str) -> ArmNames:
    """`reused_arm` is the selection Phase A trained (`selection.phase_a_method`)."""
    if arm not in KNOWN_ARMS:
        raise ValueError(f"unknown arm {arm!r}; known: {', '.join(KNOWN_ARMS)}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    if arm == reused_arm:
        return ArmNames(arm, seed, key=f"seed{seed}", variant=f"eval_adapter_s{seed}",
                        adapter_directory=f"learning_progress_seed{seed}", tensorboard_run=f"{arm}_seed{seed}")
    if arm in NATIVE_ARMS:
        return ArmNames(arm, seed, key=f"{arm}_seed{seed}", variant=f"eval_{arm}_s{seed}",
                        adapter_directory=f"{arm}_seed{seed}", tensorboard_run=f"{arm}_seed{seed}",
                        target_format=NATIVE, selection_arm=reused_arm if arm == SCOUT_ARM else LADDER_SELECTION)
    return ArmNames(arm, seed, key=f"{arm}_seed{seed}", variant=f"eval_{arm}_s{seed}",
                    adapter_directory=f"{arm}_seed{seed}", tensorboard_run=f"{arm}_seed{seed}")


def check_arm_list(arms: Sequence[str]) -> None:
    """One step runs Phase A's arm, Phase B's arms, or ONE native arm on its own: each native arm has its own
    report, and Phase B's is not rebuilt around an arm in another format."""
    unknown = [arm for arm in arms if arm not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; known: {', '.join(KNOWN_ARMS)}")
    if len(set(arms)) != len(arms):
        raise ValueError(f"an arm is listed twice: {list(arms)}")
    if any(arm in NATIVE_ARMS for arm in arms) and len(arms) > 1:
        raise ValueError(f"a native arm ({', '.join(NATIVE_ARMS)}) runs on its own (spec §13a, §13b): its report compares it with "
                         f"the base and stored controls, and Phase B's report stays as it was written; got {list(arms)}")


FRESH_SETS = ("below_band", "workbook_unsolved", "never_proved")     # spec §13b, in the order they are sampled


@dataclass(frozen=True)
class FreshNames:
    """Where one model's FRESH samples on one set of statements are stored (spec §13b). `who` is `base` or an
    arm's key; the adapter and the base are sampled with the same new seed, in the same session."""
    who: str
    set_name: str

    def __post_init__(self) -> None:
        if self.set_name not in FRESH_SETS:
            raise ValueError(f"unknown fresh-sample set {self.set_name!r}; known: {', '.join(FRESH_SETS)}")
        if not self.who:
            raise ValueError("`who` is `base` or an arm's key")

    @property
    def marker(self) -> str:
        return f"fresh_samples_{self.set_name}_{self.who}"

    @property
    def attempts_file(self) -> str:
        return f"fresh_attempts_{self.set_name}_{self.who}.jsonl"

    @property
    def verification_file(self) -> str:
        return f"fresh_verification_{self.set_name}_{self.who}.jsonl"

    @property
    def variant(self) -> str:
        return f"fresh_{self.who}"


@dataclass(frozen=True)
class BaseLossNames:
    """Where the BASE model's held-out loss in one pair format is stored."""
    nf4_marker: str
    nf4_file: str
    fp8_marker: str
    fp8_file: str


def base_loss_names(target_format: str) -> BaseLossNames:
    """Phase A's names for the legacy format (its FP8 loss is written by the stage `evaluate_base` marks);
    names of their own for every other format."""
    if target_format == LEGACY:
        return BaseLossNames("evaluate_loss_base", "heldout_loss_nf4_base.jsonl", "evaluate_base", "heldout_loss_fp8_base.jsonl")
    if target_format not in TARGET_FORMATS:
        raise ValueError(f"unknown target format {target_format!r}; known: {', '.join(TARGET_FORMATS)}")
    return BaseLossNames(f"evaluate_loss_{target_format}_base", f"heldout_loss_nf4_{target_format}_base.jsonl",
                         f"evaluate_fp8_loss_{target_format}_base", f"heldout_loss_fp8_{target_format}_base.jsonl")
