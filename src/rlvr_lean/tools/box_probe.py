"""Milestone 1 probe, run AS A QUEUED TASK on the GPU box: can work there reach the Lean server, and does
the server classify the six reference cases correctly from there? Standard library only (a bare
interpreter, no environment built yet), using the same domain rules as the real pipeline.

Writes `<out>/probe.json` (the job's completion artifact) and exits 0 only if every check passed.

    python -m rlvr_lean.tools.box_probe --kimina-host lean-server.example --kimina-port 18000 \\
        --api-key KEY --out OUT
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
import urllib.request
from pathlib import Path

from rlvr_lean.domain.verification import LEAN_HEADER, VerificationStatus, build_proof_source, classify_check_result
from rlvr_lean.infrastructure.lan_resolver import DEFAULT_LAN_DNS_SERVER, resolve_lan_host

STATEMENT = "theorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 1 := by\n"
FALSE_STATEMENT = "theorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 5 := by\n"
AXIOM_STATEMENT = "axiom rlvr_cheat : False\n\ntheorem rlvr_check (x : ℝ) (h₀ : x + 1 = 2) : x = 5 := by\n"
CASES = {   # the same six as tests/rlvr_lean/test_kimina_integration.py
    "valid": (build_proof_source(STATEMENT, "  linarith\n"), {"verified"}),
    "invalid": (build_proof_source(FALSE_STATEMENT, "  linarith\n"), {"lean_error"}),
    "sorry": (build_proof_source(FALSE_STATEMENT, "  sorry\n"), {"uses_sorry"}),
    "admit": (build_proof_source(FALSE_STATEMENT, "  admit\n"), {"uses_sorry", "forbidden_axiom"}),
    "axiom": (build_proof_source(AXIOM_STATEMENT, "  exact absurd rlvr_cheat id\n"), {"forbidden_axiom"}),
    "timeout": (LEAN_HEADER + "#eval IO.sleep 30000\n", {"timeout"}),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--kimina-host", default="lean-server.example")
    parser.add_argument("--kimina-port", type=int, default=18000)
    parser.add_argument("--lan-dns-server", default=DEFAULT_LAN_DNS_SERVER)
    key_source = parser.add_mutually_exclusive_group(required=True)
    key_source.add_argument("--api-key", help="as a queued task receives it")
    key_source.add_argument("--api-key-file", type=Path, help="for a local run")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--config", default=None, help="accepted and ignored (a job runner may append one)")
    parser.add_argument("--print-run-identity", action="store_true",
                        help="a job runner's queue-time check: print the run's identity, exit 0")
    arguments = parser.parse_args()
    if arguments.print_run_identity:
        # Written out by hand: this stdlib-only probe must not need any third-party package. The API key is
        # deliberately NOT part of the identity: it must not become part of the recorded config.
        identity = {"tool": "rlvr_lean.box_probe", "kimina_host": arguments.kimina_host,
                    "kimina_port": arguments.kimina_port, "lan_dns_server": arguments.lan_dns_server,
                    "seed": None, "run": {"out_dir": str(arguments.out) if arguments.out else None}}
        print(json.dumps({"config": identity}))
        return 0
    if arguments.out is None:
        parser.error("--out is required")
    if arguments.api_key_file is not None:
        arguments.api_key = arguments.api_key_file.expanduser().read_text().strip()

    report: dict = {"box": platform.node(), "kimina_host": arguments.kimina_host, "checks": {}}
    try:
        address, method = resolve_lan_host(arguments.kimina_host, arguments.lan_dns_server)
        report["resolved"] = {"address": address, "via": method}
        base = f"http://{address}:{arguments.kimina_port}"
        with urllib.request.urlopen(base + "/health", timeout=10) as response:
            report["health"] = json.load(response)
        body = json.dumps({"snippets": [{"id": name, "code": code} for name, (code, _) in CASES.items()],
                           "timeout": 20}).encode()
        request = urllib.request.Request(base + "/api/check", data=body, headers={
            "Content-Type": "application/json", "Authorization": "Bearer " + arguments.api_key})
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=300) as response:
            results = {result["id"]: result for result in json.load(response)["results"]}
        report["check_seconds"] = round(time.monotonic() - started, 2)
        for name, (_, expected) in CASES.items():
            status = classify_check_result(name, results[name]).status.value
            report["checks"][name] = {"status": status, "ok": status in expected}
    except (OSError, KeyError, ValueError) as error:
        report["error"] = repr(error)
    report["passed"] = bool(report["checks"]) and all(check["ok"] for check in report["checks"].values()) \
        and "error" not in report
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "probe.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
