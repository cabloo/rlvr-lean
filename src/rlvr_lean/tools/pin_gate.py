"""Pin-validation gate: do proofs KNOWN to verify under DeepSeek-Prover-V1.5's Lean/Mathlib pin verify on
our server? Spec: the first experiment's spec, "Pin validation gate (Milestone 1)".

Why it exists: a wrong Lean or Mathlib version makes valid proofs fail, which looks exactly like a bad
model. So before any model output is judged, the server must accept proofs that are already known good.

Sources (each pinned to a commit or dataset revision):
  quick_start    1    the proof in DeepSeek's quick_start.py, documented as passing in their environment
  few_shot       67   human miniF2F-valid proofs shipped in the DeepSeek-Prover-V1.5 repo
  v2_valid       221  DeepSeek-Prover-V2 miniF2F solutions, `valid/` only (`test/` is never opened)
  goedel         200  Goedel-Prover workbook proofs (its repo pins the same Mathlib commit)

PASS: the quick-start proof verifies AND at least 95% of the rest do. Every non-verified item is listed with
its status and first Lean message. It also reports throughput (proofs/s) for the batch, which is how the
worker count is tuned.

    PYTHONPATH=src python -m rlvr_lean.tools.pin_gate --url http://lean-server.example:18000 \\
        --api-key-file <the server's key file> --cache-dir /tmp/pin_sources --out /tmp/pin_gate_out
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import random
import re
import time
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from rlvr_lean.domain.verification import LEAN_HEADER, VerificationStatus, classify_check_result, theorem_name_of
from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet

DEEPSEEK_V15_COMMIT = "2c4ba9119eef74d0d611f494261b2c5bae98c69a"
DEEPSEEK_V2_COMMIT = "e598a57ea3284997d4a2a168a069fdd5064afbc8"
GOEDEL_REVISION = "b731852af8d8ab11498fda27bce9020738c01c59"
GOEDEL_ROW_COUNT = 29_750
PASS_FRACTION = 0.95


@dataclass(frozen=True)
class KnownGoodProof:
    source: str
    item_id: str
    lean_file: str   # complete file, header included, WITHOUT the trailing `#print axioms`


def _fetch(url: str, cache_path: Path, attempts: int = 5) -> bytes:
    """Download once into the cache. Retries with backoff: the Hugging Face row API answers 502 at times."""
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={"User-Agent": "rlvr-lean-pin-gate"})
        for attempt in range(1, attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    cache_path.write_bytes(response.read())
                break
            except OSError:   # URLError and HTTPError are both OSErrors
                if attempt == attempts:
                    raise
                time.sleep(5 * 2 ** attempt)
    return cache_path.read_bytes()


def load_quick_start(cache: Path) -> list[KnownGoodProof]:
    text = _fetch(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/{DEEPSEEK_V15_COMMIT}/quick_start.py",
                  cache / "quick_start.py").decode()
    prefix = re.search(r"code_prefix = r'''(.*?)'''", text, re.DOTALL).group(1)
    proof = re.search(r"# Expected output:\n'''(.*?)```", text, re.DOTALL).group(1)
    return [KnownGoodProof("quick_start", "amc12b_2003_p6", prefix + proof)]


def load_few_shot(cache: Path) -> list[KnownGoodProof]:
    lines = _fetch(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V1.5/{DEEPSEEK_V15_COMMIT}/datasets/minif2f_valid_few_shot.jsonl",
                   cache / "minif2f_valid_few_shot.jsonl").decode().splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [KnownGoodProof("few_shot", row["name"],
                           LEAN_HEADER + row["informal_prefix"] + row["formal_statement"] + row["formal_proof"])
            for row in rows]


def load_v2_valid(cache: Path) -> list[KnownGoodProof]:
    archive = _fetch(f"https://raw.githubusercontent.com/deepseek-ai/DeepSeek-Prover-V2/{DEEPSEEK_V2_COMMIT}/minif2f-solutions.zip",
                     cache / "minif2f-solutions.zip")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        names = sorted(name for name in bundle.namelist() if name.startswith("valid/") and name.endswith(".lean"))
        return [KnownGoodProof("v2_valid", Path(name).stem, bundle.read(name).decode()) for name in names]


def load_goedel(cache: Path, sample_size: int, seed: int, page_rows: int = 10) -> list[KnownGoodProof]:
    """A seeded sample, fetched as JSON from the Hugging Face datasets-server (no parquet reader needed):
    `sample_size / page_rows` pages at seeded, non-overlapping offsets, paced because the API rate-limits."""
    pages = sample_size // page_rows
    offsets = sorted(random.Random(seed).sample(range(0, GOEDEL_ROW_COUNT - page_rows, page_rows), pages))
    proofs = []
    for offset in offsets:
        url = ("https://datasets-server.huggingface.co/rows?dataset=Goedel-LM/Lean-workbook-proofs"
               f"&config=default&split=train&offset={offset}&length={page_rows}&revision={GOEDEL_REVISION}")
        cache_path = cache / "goedel" / f"{offset}_{page_rows}.json"
        if not cache_path.exists():
            time.sleep(1.0)
        for entry in json.loads(_fetch(url, cache_path))["rows"]:
            row = entry["row"]
            proofs.append(KnownGoodProof("goedel", str(row["problem_id"]), row["full_proof"]))
    return proofs


def with_axiom_report(lean_file: str) -> str:
    return lean_file.rstrip() + f"\n\n#print axioms {theorem_name_of(lean_file)}\n"


async def run_gate(proofs: list[KnownGoodProof], settings: KiminaClientSettings) -> tuple[list[dict], float]:
    """Each proof is its own call, so its completion time is known; the client's semaphore still bounds
    how many are in flight. (With several snippets per request, one slow proof holds its request open
    and idles the workers its neighbours finished on: head-of-line blocking.)"""
    async with KiminaVerifier(settings) as verifier:
        started = time.monotonic()

        async def check_one(proof: KnownGoodProof) -> tuple[dict, float]:
            snippet = LeanSnippet(f"{proof.source}:{proof.item_id}", with_axiom_report(proof.lean_file))
            raw = (await verifier.check([snippet]))[0]
            return raw, time.monotonic() - started

        outcomes = await asyncio.gather(*(check_one(proof) for proof in proofs))
        elapsed = time.monotonic() - started
    rows = []
    for proof, (raw, finished_at) in zip(proofs, outcomes):
        result = classify_check_result(f"{proof.source}:{proof.item_id}", raw)
        first_message = next((m for m in result.messages if m.startswith("error")), result.detail or "")
        rows.append({"source": proof.source, "item_id": proof.item_id, "status": result.status.value,
                     "seconds": result.verification_seconds, "peak_memory_bytes": result.peak_memory_bytes,
                     "finished_at_seconds": round(finished_at, 2),
                     "axioms": list(result.axioms), "first_message": first_message[:400]})
    return rows, elapsed


def summarize(rows: list[dict], elapsed: float) -> dict:
    by_source = {}
    for source in sorted({row["source"] for row in rows}):
        source_rows = [row for row in rows if row["source"] == source]
        by_source[source] = {"total": len(source_rows),
                             "verified": sum(row["status"] == VerificationStatus.VERIFIED.value for row in source_rows),
                             "statuses": dict(Counter(row["status"] for row in source_rows))}
    quick_start_ok = all(row["status"] == "verified" for row in rows if row["source"] == "quick_start")
    rest = [row for row in rows if row["source"] != "quick_start"]
    rest_fraction = sum(row["status"] == "verified" for row in rest) / max(1, len(rest))
    seconds = sorted(row["seconds"] for row in rows if isinstance(row["seconds"], (int, float)))
    memories = sorted(row["peak_memory_bytes"] for row in rows if isinstance(row["peak_memory_bytes"], (int, float)))

    def percentile(values, fraction):
        return values[min(len(values) - 1, int(fraction * len(values)))] if values else None

    finish_times = sorted(row["finished_at_seconds"] for row in rows)
    time_to_95 = percentile(finish_times, 0.95)
    return {
        "passed": quick_start_ok and rest_fraction >= PASS_FRACTION,
        # The batch's wall time is set by its slowest proof (up to the timeout); the rate over the first
        # 95% of completions is the number that says how fast the server works through a bulk load.
        "seconds_to_95_percent_done": time_to_95,
        "proofs_per_second_first_95_percent": round(0.95 * len(rows) / time_to_95, 3) if time_to_95 else None,
        "total_check_seconds": round(sum(seconds), 1),
        "quick_start_verified": quick_start_ok,
        "rest_verified_fraction": round(rest_fraction, 4),
        "by_source": by_source,
        "wall_seconds": round(elapsed, 1),
        "proofs_per_second": round(len(rows) / elapsed, 3) if elapsed else None,
        "check_seconds_median": percentile(seconds, 0.5), "check_seconds_p95": percentile(seconds, 0.95),
        "repl_peak_memory_gb_median": round(percentile(memories, 0.5) / 1e9, 2) if memories else None,
        "repl_peak_memory_gb_p95": round(percentile(memories, 0.95) / 1e9, 2) if memories else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--api-key-file", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--goedel-sample", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--lean-timeout", type=int, default=300, help="generous: a pin check, not a speed test")
    parser.add_argument("--snippets-per-request", type=int, default=1)
    parser.add_argument("--concurrent-requests", type=int, default=12, help="match the server's worker count")
    parser.add_argument("--label", default="gate", help="names the output files, e.g. repls12")
    arguments = parser.parse_args()

    cache = arguments.cache_dir.expanduser()
    proofs = (load_quick_start(cache) + load_few_shot(cache) + load_v2_valid(cache)
              + load_goedel(cache, arguments.goedel_sample, arguments.seed))
    settings = KiminaClientSettings(
        base_url=arguments.url, api_key=arguments.api_key_file.expanduser().read_text().strip(),
        lean_timeout_seconds=arguments.lean_timeout, snippets_per_request=arguments.snippets_per_request,
        concurrent_requests=arguments.concurrent_requests)
    rows, elapsed = asyncio.run(run_gate(proofs, settings))
    summary = summarize(rows, elapsed)

    arguments.out.mkdir(parents=True, exist_ok=True)
    with (arguments.out / f"{arguments.label}_results.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (arguments.out / f"{arguments.label}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    for row in rows:
        if row["status"] != "verified":
            print(f"NOT VERIFIED {row['source']}:{row['item_id']} {row['status']} {row['first_message'][:200]!r}")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
