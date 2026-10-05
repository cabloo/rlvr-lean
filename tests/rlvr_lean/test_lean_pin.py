"""The Lean pin (the OEIS Open spec, O2a): ONE setting, `lean.pin`, and what it selects.

9a  the endpoint, the rule for a failed warning, whether imports move above a leading comment, the Lean
    timeout and the requests in flight; the default is v4.9 and v4.9 is exactly what it was.
9b  the v4.27 endpoint is the pool: HTTPS to a resolved address, the certificate checked against the pool's
    name, the pool's authority and no other (the handshake itself: test_kimina_client_tls.py).
9c  a run directory of its own per pin.
9d  the earlier pin's seed and reward statements, filtered again.
"""

import base64
import json
from pathlib import Path

import pytest

from rlvr_lean.data.sources import EarlierStatements, Statement, load_earlier_statements
from rlvr_lean.domain.verification import (
    DEFAULT_LEAN_PIN,
    LEAN_PINS,
    VerificationStatus,
    lean_pin,
    lean_pin_from_config,
)

AXIOM_REPORT = {"severity": "info", "data": "'demo' depends on axioms: [propext]"}
RECOVERED_FAILURE = {"severity": "warning", "data": "aesop: failed to prove the goal after exhaustive search."}
LICENSED_FILE = "/-\nCopyright 2026\n-/\nimport Mathlib\n\ntheorem demo : True := trivial\n"


def response(messages):
    return {"time": 1.0, "response": {"env": 0, "messages": messages}}


def config_at(pin=None):
    config = {"kimina": {"host": "lean-server.example", "port": 18000, "lan_dns_server": "192.0.2.1",
                         "lean_timeout_seconds": 60, "concurrent_requests": 16},
              "lean": {"pool": {"name": "pool.lan", "port": 18100, "lean_timeout_seconds": 120, "concurrent_requests": 8}}}
    if pin is not None:
        config["lean"]["pin"] = pin
    return config


# ------------------------------------------------------------------------------------- the setting (9a)
def test_the_default_pin_is_v4_9_and_a_config_from_before_the_port_reads_as_it():
    assert DEFAULT_LEAN_PIN == "v4.9"
    assert lean_pin_from_config({"kimina": {}}).name == "v4.9"
    assert lean_pin_from_config(config_at()).name == "v4.9"
    assert lean_pin_from_config(config_at("v4.27")).name == "v4.27"


def test_the_shipped_config_selects_v4_9_and_names_the_pool():
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "src/rlvr_lean/config/experiment.yaml").read_text())
    assert config["lean"]["pin"] == "v4.9" and lean_pin_from_config(config) is LEAN_PINS["v4.9"]
    assert config["lean"]["pool"] == {"name": "lean-pool.example", "port": 18100, "lean_timeout_seconds": 120,
                                      "concurrent_requests": 8,
                                      # how long the proxy's queue and a Lean server may hold a request (the ladder loop's client waits that long)
                                      "proxy_queue_seconds": 630, "server_wait_seconds": 120,
                                      # for a client that follows the size the pool states (lean-pool's README, "Background work and the pool's size")
                                      "size_margin": 1.25, "in_flight_ceiling": 256}
    assert config["kimina"]["port"] == 18000 and config["kimina"]["concurrent_requests"] == 16    # v4.9's, untouched


def test_an_unknown_pin_is_refused_by_name():
    with pytest.raises(ValueError, match="v4.27, v4.9"):
        lean_pin("v4.15")
    with pytest.raises(ValueError, match="4.27"):
        lean_pin_from_config({"lean": {"pin": 4.27}})          # YAML without the `v` is a number, not a pin


def test_a_failed_warning_is_an_error_at_v4_9_and_not_at_v4_27():
    recovered = response([RECOVERED_FAILURE, AXIOM_REPORT])
    assert LEAN_PINS["v4.9"].classify("a", recovered).status is VerificationStatus.LEAN_ERROR
    assert LEAN_PINS["v4.27"].classify("a", recovered).status is VerificationStatus.VERIFIED
    broken = response([{"severity": "error", "data": "unsolved goals"}, AXIOM_REPORT])
    assert {LEAN_PINS[name].classify("a", broken).status for name in LEAN_PINS} == {VerificationStatus.LEAN_ERROR}


def test_imports_move_above_a_leading_comment_at_v4_27_only():
    assert LEAN_PINS["v4.9"].source(LICENSED_FILE) == LICENSED_FILE
    assert LEAN_PINS["v4.27"].source(LICENSED_FILE).startswith("import Mathlib\n/-\nCopyright 2026\n-/\n")


def test_a_file_the_pipeline_assembles_is_sent_unchanged_at_both_pins():
    from rlvr_lean.domain.verification import build_proof_source

    source = build_proof_source("theorem demo (x : ℕ) : x = x := by\n", "  rfl\n")
    assert {LEAN_PINS[name].source(source) for name in LEAN_PINS} == {source}


def test_each_pin_has_a_run_directory_of_its_own_and_v4_9_keeps_runs():
    assert LEAN_PINS["v4.9"].runs_directory == "runs"
    directories = [pin.runs_directory for pin in LEAN_PINS.values()]
    assert len(set(directories)) == len(directories)
    assert LEAN_PINS["v4.9"].statements_from_pin is None and LEAN_PINS["v4.27"].statements_from_pin == "v4.9"


def test_nothing_else_in_the_pipeline_asks_which_pin_is_in_use():
    """Item 9a's last sentence, as a guard: outside the pin's own module, no pipeline code names a pin or
    compares a pin's name. It asks the pin object (`settings.pin`) what to do instead."""
    import re

    root = Path(__file__).resolve().parents[2] / "src" / "rlvr_lean"
    names_a_pin = re.compile(r"""["']v4\.\d+["']|\.pin\.name\s*[!=]=|pin\s*[!=]=\s*["']""")
    offenders = [str(path.relative_to(root)) for package in ("gpu", "infrastructure", "runner", "data", "reporting", "domain")
                 for path in sorted((root / package).rglob("*.py"))
                 if path.name != "pin.py" and names_a_pin.search(path.read_text())]
    assert offenders == []


# ------------------------------------------------------------------------------ the endpoint (9a, 9b)
@pytest.fixture
def service(monkeypatch):
    pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")
    from rlvr_lean.infrastructure import verification_service

    resolved = []

    def resolve(host, dns_server):
        resolved.append((host, dns_server))
        return "192.0.2.35", "test"

    monkeypatch.setattr(verification_service, "resolve_lan_host", resolve)
    verification_service.resolved = resolved
    return verification_service


def test_v4_9_is_the_single_server_over_plain_http_as_before(service):
    settings = service.lean_settings(config_at(), api_key="k")
    assert settings.base_url == "http://192.0.2.35:18000" and settings.api_key == "k"
    assert (settings.lean_timeout_seconds, settings.concurrent_requests) == (60, 16)
    assert settings.ca_file is None and settings.tls_server_name is None
    assert settings.pin is LEAN_PINS["v4.9"] and service.resolved == [("lean-server.example", "192.0.2.1")]
    # A certificate file in the environment changes nothing at v4.9.
    assert service.lean_settings(config_at("v4.9"), api_key="k", ca_file="/some/ca.crt") == settings


def test_v4_27_is_the_pool_by_address_with_its_name_its_authority_and_its_own_limits(service):
    settings = service.lean_settings(config_at("v4.27"), api_key="pool-key", ca_file="/store/lean-ca/ca.crt")
    assert settings.base_url == "https://192.0.2.35:18100"          # the ADDRESS the pool's name resolves to
    assert settings.tls_server_name == "pool.lan"                       # the NAME the certificate must carry
    assert settings.ca_file == "/store/lean-ca/ca.crt" and settings.api_key == "pool-key"
    assert (settings.lean_timeout_seconds, settings.concurrent_requests) == (120, 8)
    assert settings.pin is LEAN_PINS["v4.27"] and service.resolved == [("pool.lan", "192.0.2.1")]


def test_v4_27_without_the_authoritys_certificate_is_refused_before_anything_is_resolved_or_sent(service):
    with pytest.raises(RuntimeError, match="RLVR_LEAN_KIMINA_CA_FILE"):
        service.lean_settings(config_at("v4.27"), api_key="pool-key")
    assert service.resolved == []


def test_a_step_reads_the_key_and_the_certificate_file_from_its_environment(service, monkeypatch):
    monkeypatch.setenv("RLVR_LEAN_KIMINA_API_KEY", "from-the-entry")
    monkeypatch.setenv("RLVR_LEAN_KIMINA_CA_FILE", "/store/lean-ca/ca.crt")
    settings = service.kimina_settings_from_config(config_at("v4.27"))
    assert (settings.api_key, settings.ca_file, settings.tls_server_name) == ("from-the-entry", "/store/lean-ca/ca.crt", "pool.lan")
    monkeypatch.delenv("RLVR_LEAN_KIMINA_CA_FILE")
    assert service.kimina_settings_from_config(config_at()).base_url == "http://192.0.2.35:18000"


class RecordingVerifier:
    """Stands in for the client: records what would be sent and answers each snippet with a recovered failure."""
    sent: list = []

    def __init__(self, settings):
        self.settings = settings

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exception_info):
        return None

    async def check(self, snippets):
        type(self).sent.extend(snippets)
        return [{"id": snippet.snippet_id, **response([RECOVERED_FAILURE, AXIOM_REPORT])} for snippet in snippets]


@pytest.mark.parametrize("pin, first_line, status", [("v4.9", "/-", VerificationStatus.LEAN_ERROR),
                                                     ("v4.27", "import Mathlib", VerificationStatus.VERIFIED)])
def test_checks_are_written_and_read_by_the_pins_rules(service, monkeypatch, pin, first_line, status):
    monkeypatch.setattr(service, "KiminaVerifier", RecordingVerifier)
    RecordingVerifier.sent = []
    settings = service.lean_settings(config_at(pin), api_key="k", ca_file="/ca.crt")
    raw = service.check_lean_sources(settings, {"licensed": LICENSED_FILE})
    assert RecordingVerifier.sent[0].code.splitlines()[0] == first_line and set(raw) == {"licensed"}
    verification = service.VerificationService(settings)
    verification.submit([service.ProofAttemptToVerify("a", "theorem demo : True := by\n", "  trivial\n"),
                         service.ProofAttemptToVerify("b", "theorem demo : True := by\n", "  sorry\n")])
    results = verification.results()
    assert results["a"].status is status
    assert results["b"].status is VerificationStatus.REJECTED_LEXICAL          # the lexical filter is the same at every pin


def test_bare_client_settings_are_read_by_the_default_pins_rules(service):
    from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings

    assert service.pin_of(KiminaClientSettings(base_url="http://kimina.test")) is LEAN_PINS["v4.9"]


# ------------------------------------------------------------- how a task is told the certificate (9b)
def test_the_entry_writes_the_certificate_it_is_handed_into_the_store(tmp_path):
    from rlvr_lean.runner.entry import store_ca_certificate

    pem = b"-----BEGIN CERTIFICATE-----\nMIIB\n-----END CERTIFICATE-----\n"
    path = store_ca_certificate(base64.b64encode(pem).decode(), tmp_path)
    assert path.parent == tmp_path / "lean-ca" and path.read_bytes() == pem
    assert store_ca_certificate(base64.b64encode(pem).decode(), tmp_path) == path       # named by content: idempotent
    assert sorted(entry.name for entry in path.parent.iterdir()) == [path.name]         # no temporary file left


@pytest.mark.parametrize("value", ["not base64 !!", base64.b64encode(b"a key, not a certificate").decode()])
def test_the_entry_refuses_a_value_that_is_not_a_certificate_without_echoing_it(tmp_path, value):
    from rlvr_lean.runner.entry import store_ca_certificate

    with pytest.raises(ValueError) as refusal:
        store_ca_certificate(value, tmp_path)
    assert value not in str(refusal.value) and not (tmp_path / "lean-ca").exists()


# --------------------------------------------------------------------- a store of its own (9c), 9d
@pytest.fixture
def pipeline(monkeypatch, tmp_path):
    pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")
    from rlvr_lean.gpu import pipeline as module

    monkeypatch.setattr(module, "STORE", tmp_path)
    monkeypatch.setattr(module, "_LEAN_PIN", {})
    monkeypatch.setenv("RLVR_LEAN_PROFILE", "smoke")
    monkeypatch.delenv("RLVR_LEAN_STEP_DIR", raising=False)
    return module


def test_v4_9_keeps_its_run_directory_and_v4_27_has_another(pipeline, tmp_path):
    from rlvr_lean.domain.training.arms import arm_names

    names = arm_names("random", 0, "learning_progress_cosine")
    pipeline.use_lean_pin(config_at())
    assert pipeline._store().root == tmp_path / "runs" / "smoke"
    assert pipeline._adapter_dir(names) == tmp_path / "runs" / "smoke" / "adapters" / "random_seed0"
    pipeline._store().mark_done("prepare_data", {"pin": "v4.9"})
    pipeline._store().write_rows("statements_seed.jsonl", [{"statement_id": "from v4.9"}])

    pipeline.use_lean_pin(config_at("v4.27"))
    assert pipeline._store().root == tmp_path / "runs-v4.27" / "smoke"
    assert pipeline._adapter_dir(names) == tmp_path / "runs-v4.27" / "smoke" / "adapters" / "random_seed0"
    assert not pipeline._store().is_done("prepare_data")                    # no v4.9 marker is read as a v4.27 one
    with pytest.raises(FileNotFoundError):
        pipeline._store().read_rows("statements_seed.jsonl")                # nor a v4.9 file
    pipeline._store().mark_done("prepare_data", {"pin": "v4.27"})
    pipeline.use_lean_pin(config_at("v4.9"))
    assert pipeline._store().done_summary("prepare_data") == {"pin": "v4.9"}   # and none is written over


def test_a_step_that_selected_no_pin_is_refused_rather_than_given_v4_9s_directory(pipeline):
    with pytest.raises(RuntimeError, match="use_lean_pin"):
        pipeline._store()


def workbook_statement(name, body="True"):
    return Statement(statement_id=name, statement=f"theorem {name} : {body} := by\n", natural_language=f"problem {name}", source="lean_workbook")


def stored_rows(statements):
    return "".join(json.dumps({"statement_id": s.statement_id, "statement": s.statement, "source": s.source}) + "\n" for s in statements)


def test_the_earlier_pins_statements_are_filtered_again_and_only_shrink(tmp_path):
    workbook = [workbook_statement(f"w{index}") for index in range(8)]
    (tmp_path / "statements_seed.jsonl").write_text(stored_rows([workbook[3], workbook[0], workbook[5]]))
    (tmp_path / "statements_reward.jsonl").write_text(stored_rows([workbook[6], workbook[1]]))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    earlier = load_earlier_statements(tmp_path, workbook)
    assert [s.statement_id for s in earlier.candidates] == ["w3", "w0", "w5", "w6", "w1"]
    # w0 and w6 no longer compile; w2 compiles but was in neither set and is not taken in their place.
    seeds, reward = earlier.survivors([workbook[1], workbook[2], workbook[3], workbook[5]])
    assert [s.statement_id for s in seeds] == ["w3", "w5"] and [s.statement_id for s in reward] == ["w1"]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before          # read, never written


def test_missing_or_different_earlier_statements_are_refused(tmp_path):
    workbook = [workbook_statement("w0")]
    with pytest.raises(FileNotFoundError, match="9d"):
        load_earlier_statements(tmp_path / "absent", workbook)
    assert not (tmp_path / "absent").exists()
    (tmp_path / "statements_seed.jsonl").write_text(stored_rows([workbook_statement("w0", body="1 = 1")]))
    (tmp_path / "statements_reward.jsonl").write_text("")
    with pytest.raises(ValueError, match="w0"):
        load_earlier_statements(tmp_path, workbook)
    assert EarlierStatements(seeds=[], reward=[]).survivors(workbook) == ([], [])


def compile_reply(statement_id, compiles):
    messages = [{"severity": "info", "data": "rlvr-type-fingerprint 7"}] if compiles else [{"severity": "error", "data": "unknown identifier"}]
    return {"id": statement_id, "time": 0.5, "response": {"env": 0, "messages": messages}}


@pytest.fixture
def prepared(pipeline, monkeypatch, tmp_path):
    """`prepare_data` with the data and the Lean server faked: ten workbook statements, of which w2 and w7 do
    not compile under the pin being run; two miniF2F statements per split."""
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((Path(__file__).resolve().parents[2] / "src/rlvr_lean/config/experiment.yaml").read_text())
    config["profiles"]["smoke"].update({"seed_statements": 3, "reward_statements": 3, "minif2f_problems": 2})
    workbook = [workbook_statement(f"w{index}") for index in range(10)]
    checked = []

    def check(settings, sources):
        checked.append(sorted(sources))
        return {key: compile_reply(key, key not in ("w2", "w7")) for key in sources}

    monkeypatch.setattr(pipeline, "load_lean_workbook", lambda directory, revision: workbook)
    monkeypatch.setattr(pipeline, "all_minif2f_normalized", lambda directory, commit: set())
    monkeypatch.setattr(pipeline, "load_minif2f", lambda directory, commit, split: [workbook_statement(f"{split}{i}") for i in range(2)])
    monkeypatch.setattr(pipeline, "check_lean_sources", check)

    def run(pin):
        from rlvr_lean.infrastructure.verification_service import LeanCheckSettings

        config["lean"]["pin"] = pin
        monkeypatch.setattr(pipeline, "kimina_settings_from_config",
                            lambda config: LeanCheckSettings(base_url="http://kimina.test", pin=LEAN_PINS[pin]))
        pipeline.use_lean_pin(config)
        return pipeline.prepare_data(config), checked

    return run


def ids(path):
    return [json.loads(line)["statement_id"] for line in path.read_text().splitlines()]


def test_prepare_data_at_v4_27_filters_v4_9s_statements_into_its_own_store(prepared, tmp_path):
    summary_49, _ = prepared("v4.9")
    old = tmp_path / "runs" / "smoke"
    assert summary_49["seed_statements"] == 3 and summary_49["reward_statements"] == 3
    assert "statements_from_pin" not in summary_49 and "lean_pin" not in summary_49        # v4.9's summary is what it was
    seed_49, reward_49 = ids(old / "statements_seed.jsonl"), ids(old / "statements_reward.jsonl")
    # Make two of v4.9's statements the ones that fail under the next pin: one seed, one reward.
    (old / "statements_seed.jsonl").write_text((old / "statements_seed.jsonl").read_text().replace(seed_49[1], "w2"))
    (old / "statements_reward.jsonl").write_text((old / "statements_reward.jsonl").read_text().replace(reward_49[0], "w7"))
    seed_49[1], reward_49[0] = "w2", "w7"
    before = {path.name: path.read_bytes() for path in old.iterdir()}

    summary, checked = prepared("v4.27")
    new = tmp_path / "runs-v4.27" / "smoke"
    assert sorted(seed_49 + reward_49) == checked[-2]                       # exactly v4.9's statements were compiled again
    assert ids(new / "statements_seed.jsonl") == [s for s in seed_49 if s != "w2"]
    assert ids(new / "statements_reward.jsonl") == [s for s in reward_49 if s != "w7"]
    assert summary["lean_pin"] == "v4.27" and summary["candidates_checked"] == 6 and summary["candidates_compiled"] == 4
    assert (summary["seed_statements"], summary["reward_statements"]) == (2, 2)
    assert summary["statements_from_pin"] == {"pin": "v4.9", "seed_statements": 3, "reward_statements": 3}
    assert {path.name: path.read_bytes() for path in old.iterdir()} == before          # v4.9's run is untouched


def test_prepare_data_at_v4_9_chooses_from_the_ranked_candidates_as_before(prepared, tmp_path):
    summary, checked = prepared("v4.9")
    assert len(checked[0]) == 10 and summary["candidates_compiled"] == 8      # every candidate is tried, w2 and w7 fail
    run = tmp_path / "runs" / "smoke"
    chosen = ids(run / "statements_seed.jsonl") + ids(run / "statements_reward.jsonl")
    assert len(chosen) == 6 and not {"w2", "w7"} & set(chosen)
    assert not (tmp_path / "runs-v4.27").exists()
