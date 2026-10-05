"""The Kimina client over HTTPS with a private certificate authority (the lean-pool front door,
lean-pool's README, "TLS"): it trusts the authority it is given and no other.

A real TLS server on loopback; the certificates are made with the `openssl` command."""

import asyncio
import http.server
import json
import shutil
import ssl
import subprocess
import threading

import pytest

httpx = pytest.importorskip("httpx", reason="rlvr_lean's client dependency; see pyproject.toml")

from rlvr_lean.infrastructure.kimina_client import KiminaClientSettings, KiminaVerifier, LeanSnippet  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("openssl") is None, reason="needs the openssl command")

AUTHORITY_CONFIG = """[req]
distinguished_name = dn
x509_extensions = v3_ca
prompt = no
[dn]
CN = {name}
[v3_ca]
basicConstraints = critical,CA:TRUE
keyUsage = critical,keyCertSign,cRLSign
subjectKeyIdentifier = hash
"""
SERVER_EXTENSIONS = """basicConstraints = CA:FALSE
keyUsage = critical,digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:localhost,IP:127.0.0.1
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid
"""


def openssl(directory, *arguments):
    subprocess.run(["openssl", *arguments], cwd=directory, check=True, capture_output=True)


def make_authority(directory, name):
    (directory / f"{name}.cnf").write_text(AUTHORITY_CONFIG.format(name=name))
    openssl(directory, "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", f"{name}.key")
    openssl(directory, "req", "-x509", "-new", "-key", f"{name}.key", "-sha256", "-days", "2",
            "-config", f"{name}.cnf", "-out", f"{name}.crt")
    return directory / f"{name}.crt"


def make_server_certificate(directory, authority):
    (directory / "server.ext").write_text(SERVER_EXTENSIONS)
    openssl(directory, "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "server.key")
    openssl(directory, "req", "-new", "-key", "server.key", "-subj", "/CN=pool", "-out", "server.csr")
    openssl(directory, "x509", "-req", "-in", "server.csr", "-CA", f"{authority}.crt", "-CAkey", f"{authority}.key",
            "-CAcreateserial", "-days", "2", "-sha256", "-extfile", "server.ext", "-out", "server.crt")


class LeanServerStandIn(http.server.BaseHTTPRequestHandler):
    seen_authorization: list = []

    def do_POST(self):  # noqa: N802 - the base class's naming
        type(self).seen_authorization.append(self.headers.get("Authorization"))
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        reply = json.dumps({"results": [{"id": snippet["id"], "time": 0.1,
                                         "response": {"env": 0, "messages": [], "sorries": []}}
                                        for snippet in request["snippets"]]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)

    def log_message(self, *arguments):
        pass


@pytest.fixture
def pool(tmp_path):
    """An HTTPS stand-in for the pool's front door, its authority's certificate, and a stranger's."""
    trusted = make_authority(tmp_path, "pool-authority")
    stranger = make_authority(tmp_path, "another-authority")
    make_server_certificate(tmp_path, "pool-authority")
    LeanServerStandIn.seen_authorization = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), LeanServerStandIn)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "server.crt", tmp_path / "server.key")
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"url": f"https://127.0.0.1:{server.server_address[1]}", "trusted": str(trusted), "stranger": str(stranger)}
    server.shutdown()
    server.server_close()


def check_one(url, ca_file):
    async def run():
        async def no_wait(seconds):
            return None

        settings = KiminaClientSettings(base_url=url, api_key="secret", ca_file=ca_file, max_attempts_per_batch=2)
        async with KiminaVerifier(settings, sleep=no_wait) as verifier:
            return (await verifier.check([LeanSnippet("only", "-- a proof")]))[0]

    return asyncio.run(run())


def test_a_server_signed_by_the_given_authority_is_answered(pool):
    result = check_one(pool["url"], pool["trusted"])
    assert result["id"] == "only" and "error" not in result
    assert LeanServerStandIn.seen_authorization == ["Bearer secret"]


def test_a_server_signed_by_another_authority_gets_no_request_and_no_key(pool):
    result = check_one(pool["url"], pool["stranger"])
    assert result["error"].startswith("server_error: no answer after 2 attempts")
    assert "CERTIFICATE_VERIFY_FAILED" in result["error"]
    assert LeanServerStandIn.seen_authorization == []          # the handshake failed before the key was sent


def test_the_systems_own_authorities_do_not_vouch_for_the_pool(pool):
    result = check_one(pool["url"], None)
    assert "CERTIFICATE_VERIFY_FAILED" in result["error"]
    assert LeanServerStandIn.seen_authorization == []


def test_a_missing_authority_file_is_refused_before_any_request(tmp_path):
    settings = KiminaClientSettings(base_url="https://127.0.0.1:1", ca_file=str(tmp_path / "absent.crt"))
    with pytest.raises(OSError):
        KiminaVerifier(settings)


# --- by address, by name (the OEIS Open spec, O2a item 9b) -------------------------------------
# A GPU task reaches the pool at a resolved LAN address, and the pool's certificate names hosts, not
# addresses. So the client connects to the address and checks the certificate against the pool's NAME, which
# it also sends for SNI. The server below is real TLS on loopback, and its certificate names ONLY a host.

POOL_NAME = "pool.test"


class NamedPoolStandIn(LeanServerStandIn):
    seen_authorization: list = []

    def do_GET(self):  # noqa: N802 - the base class's naming
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def named_pool(tmp_path):
    """An HTTPS stand-in whose certificate carries the name `pool.test` and no address, reached at 127.0.0.1.
    `sni` collects the server name each TLS client asked for."""
    trusted = make_authority(tmp_path, "pool-authority")
    stranger = make_authority(tmp_path, "another-authority")
    (tmp_path / "named.ext").write_text(SERVER_EXTENSIONS.replace("DNS:localhost,IP:127.0.0.1", f"DNS:{POOL_NAME}"))
    openssl(tmp_path, "ecparam", "-name", "prime256v1", "-genkey", "-noout", "-out", "named.key")
    openssl(tmp_path, "req", "-new", "-key", "named.key", "-subj", "/CN=front-door", "-out", "named.csr")
    openssl(tmp_path, "x509", "-req", "-in", "named.csr", "-CA", "pool-authority.crt", "-CAkey", "pool-authority.key",
            "-CAcreateserial", "-days", "2", "-sha256", "-extfile", "named.ext", "-out", "named.crt")
    NamedPoolStandIn.seen_authorization = []
    sni: list = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), NamedPoolStandIn)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(tmp_path / "named.crt", tmp_path / "named.key")
    context.sni_callback = lambda tls_socket, server_name, tls_context: sni.append(server_name)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield {"url": f"https://127.0.0.1:{server.server_address[1]}", "trusted": str(trusted), "stranger": str(stranger), "sni": sni}
    server.shutdown()
    server.server_close()


def check_by_name(url, ca_file, server_name):
    async def run():
        async def no_wait(seconds):
            return None

        settings = KiminaClientSettings(base_url=url, api_key="secret", ca_file=ca_file, tls_server_name=server_name,
                                        max_attempts_per_batch=2)
        async with KiminaVerifier(settings, sleep=no_wait) as verifier:
            healthy = await verifier.is_healthy()
            return healthy, (await verifier.check([LeanSnippet("only", "-- a proof")]))[0]

    return asyncio.run(run())


def test_an_address_is_answered_when_the_certificate_carries_the_expected_name(named_pool):
    healthy, result = check_by_name(named_pool["url"], named_pool["trusted"], POOL_NAME)
    assert healthy and result["id"] == "only" and "error" not in result
    assert NamedPoolStandIn.seen_authorization == ["Bearer secret"]
    assert named_pool["sni"] and set(named_pool["sni"]) == {POOL_NAME}      # the name, not the address, is sent for SNI


def test_an_address_alone_is_refused_when_the_certificate_names_only_a_host(named_pool):
    healthy, result = check_by_name(named_pool["url"], named_pool["trusted"], None)
    assert not healthy and "CERTIFICATE_VERIFY_FAILED" in result["error"]
    assert NamedPoolStandIn.seen_authorization == []          # the handshake failed before the key was sent


def test_a_certificate_for_another_name_is_refused(named_pool):
    """The name is CHECKED, not only sent: the trusted authority's certificate for `pool.test` does not pass
    for another host."""
    healthy, result = check_by_name(named_pool["url"], named_pool["trusted"], "another.test")
    assert not healthy and "CERTIFICATE_VERIFY_FAILED" in result["error"]
    assert "another.test" in result["error"]                  # a host-name mismatch, named
    assert NamedPoolStandIn.seen_authorization == []


def test_the_expected_name_does_not_replace_the_authority(named_pool):
    healthy, result = check_by_name(named_pool["url"], named_pool["stranger"], POOL_NAME)
    assert not healthy and "CERTIFICATE_VERIFY_FAILED" in result["error"]
    assert NamedPoolStandIn.seen_authorization == []


def test_an_expected_name_without_https_is_refused_before_any_request():
    with pytest.raises(ValueError, match="https"):
        KiminaVerifier(KiminaClientSettings(base_url="http://127.0.0.1:1", tls_server_name=POOL_NAME))
