"""How much of the GPU a task may take: all of it except a reserve for the machine's desktop session.

Why it exists (2026-10-03): the sampling engine was given 85% of the card's TOTAL memory. With its own
overhead and the display's 1.1 GB that came to 15,590 of 15,819 MiB, and no window
manager could start on the machine. "This task has priority on the box" is about the queue and
the CPU caps; it was never meant to take the last of the VRAM from the screen.

So every GPU stage budgets against `total − reserve`, where total is the card's USABLE memory:
  - vLLM is given a fraction of that total and then uses about 1 GB more (CUDA graphs and the like;
    measured 1,009 MiB), so its fraction is (total − reserve − overhead) / total. Inside its share the
    engine spends 9.04 GiB before any cache (7.61 weights and non-PyTorch, 1.42 peak activation, its own
    report), and refuses to start unless the rest holds one `max_model_len` sequence (1.88 GiB at 4,096
    tokens). That is why the reserve is 3 GiB and not 4: at 4 the cache would be 1.4 GiB;
  - a PyTorch stage (training, scoring, the loss evaluation) caps its allocator at
    (total − reserve − overhead) / total, so it runs out of memory inside its own process instead of
    starving the desktop. The overhead is what the process holds OUTSIDE the allocator the cap governs (the
    CUDA context and library workspaces): 0.5 GiB is an estimate, not yet measured on this card.

Nothing here imports torch: the vLLM stage must not initialise CUDA in the parent process, so the card's
size comes from nvidia-smi.
"""

from __future__ import annotations

import subprocess

MIB_PER_GIB = 1024


def sampling_memory_fraction(total_mib: int, reserve_gb: float, engine_overhead_gb: float, ceiling: float) -> float:
    """vLLM's `gpu_memory_utilization`: the largest fraction whose whole footprint leaves `reserve_gb` free.

    `ceiling` is the fraction the config would use on a card nobody else needs; the result never exceeds it.
    Rounded DOWN to three places, so rounding can only leave more for the desktop.
    """
    if total_mib <= 0:
        raise ValueError(f"the card's total memory must be positive, got {total_mib} MiB")
    if reserve_gb < 0 or engine_overhead_gb < 0 or not 0 < ceiling <= 1:
        raise ValueError("reserve and overhead must be non-negative and the ceiling must lie in (0, 1]")
    total_gb = total_mib / MIB_PER_GIB
    budget_gb = total_gb - reserve_gb - engine_overhead_gb
    if budget_gb <= 0:
        raise ValueError(f"a {total_gb:.1f} GiB card has nothing left after a {reserve_gb} GiB reserve and "
                         f"{engine_overhead_gb} GiB of engine overhead")
    return min(ceiling, int(budget_gb / total_gb * 1000) / 1000)


def process_memory_fraction(total_mib: int, reserve_gb: float, process_overhead_gb: float) -> float:
    """The cap for one PyTorch process's allocator (`torch.cuda.set_per_process_memory_fraction`), rounded
    down. `process_overhead_gb` is what the process holds outside that allocator, so the allocator's cap plus
    the overhead is the process's whole footprint and still leaves `reserve_gb` free."""
    if total_mib <= 0:
        raise ValueError(f"the card's total memory must be positive, got {total_mib} MiB")
    if reserve_gb < 0 or process_overhead_gb < 0:
        raise ValueError("reserve and overhead must be non-negative")
    total_gb = total_mib / MIB_PER_GIB
    budget_gb = total_gb - reserve_gb - process_overhead_gb
    if budget_gb <= 0:
        raise ValueError(f"a {total_gb:.1f} GiB card has nothing left after a {reserve_gb} GiB reserve and "
                         f"{process_overhead_gb} GiB of process overhead")
    return int(budget_gb / total_gb * 1000) / 1000


def token_budget_batches(token_counts: list[int], budget: int, block_tokens: int = 16) -> list[tuple[int, int]]:
    """Contiguous (start, stop) groups of sequences whose cache need fits `budget` tokens.

    A sequence's need is its tokens plus the one it generates, rounded up to the cache's block size. A
    sequence larger than the budget gets a group of its own.

    Why it exists (2026-10-03, task 83efeb13): scoring 114 held-out prompts in one call needs about 25,000
    tokens of cache. With the desktop's reserve the engine has 5,264, so it evicts sequences that are half
    done, and vLLM asserts when an evicted sequence had asked for prompt log-probabilities
    (`assert not prompt_logprobs_tensors`): the engine died and took the arm's evaluation with it. A group
    that fits the cache is never evicted.
    """
    if budget <= 0 or block_tokens <= 0:
        raise ValueError("the token budget and the block size must be positive")
    groups, start, used = [], 0, 0
    for index, count in enumerate(token_counts):
        need = -(-(count + 1) // block_tokens) * block_tokens
        if index > start and used + need > budget:
            groups.append((start, index))
            start, used = index, 0
        used += need
    if token_counts:
        groups.append((start, len(token_counts)))
    return groups


def usable_mib(nvidia_smi_used_free: str) -> int:
    """The card's usable memory from nvidia-smi's `memory.used, memory.free` line: their sum.

    Not `memory.total`: that includes what the driver reserves for itself (484 MiB on the card this was measured on,
    16,303 against 15,819), which neither a task nor the desktop can have. CUDA, and so vLLM and PyTorch,
    take their fractions of the usable figure.
    """
    used, free = (int(field) for field in nvidia_smi_used_free.strip().splitlines()[0].split(","))
    return used + free


def gpu_total_mib() -> int:
    """The first GPU's usable memory, from nvidia-smi (no CUDA initialisation in this process)."""
    output = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=30, check=True).stdout
    return usable_mib(output)
