"""Resolve a LAN host name from inside a worker container. Spec §1, item 11.

Docker on a host whose /etc/resolv.conf is a systemd-resolved stub hands containers PUBLIC DNS, which has
never heard of a name that only the LAN knows. So: try the system resolver
first, and if it cannot answer, ask the LAN's own DNS server (the router) directly with one plain DNS
query. Standard library only, so it runs on any interpreter before any environment is built.
"""

from __future__ import annotations

import random
import socket
import struct

DEFAULT_LAN_DNS_SERVER = "192.0.2.1"       # a documentation address: set `kimina.lan_dns_server` to your LAN's


def _query_a_record(host: str, dns_server: str, timeout_seconds: float) -> str | None:
    """One DNS A-record query over UDP (RFC 1035). Returns the first IPv4 address, or None."""
    query_id = random.randrange(1 << 16)
    header = struct.pack(">HHHHHH", query_id, 0x0100, 1, 0, 0, 0)   # recursion desired, one question
    question = b"".join(bytes([len(label)]) + label.encode("ascii") for label in host.rstrip(".").split("."))
    question += b"\x00" + struct.pack(">HH", 1, 1)                   # QTYPE A, QCLASS IN
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(timeout_seconds)
        connection.sendto(header + question, (dns_server, 53))
        reply, _ = connection.recvfrom(4096)
    reply_id, flags, question_count, answer_count = struct.unpack(">HHHH", reply[:8])
    if reply_id != query_id or flags & 0x000F != 0 or answer_count == 0:
        return None
    position = 12
    for _ in range(question_count):            # skip the echoed question
        while reply[position] != 0:
            position += reply[position] + 1
        position += 5
    for _ in range(answer_count):
        if reply[position] & 0xC0 == 0xC0:     # compressed name pointer
            position += 2
        else:
            while reply[position] != 0:
                position += reply[position] + 1
            position += 1
        record_type, _, _, length = struct.unpack(">HHIH", reply[position:position + 10])
        position += 10
        if record_type == 1 and length == 4:
            return socket.inet_ntoa(reply[position:position + 4])
        position += length
    return None


def resolve_lan_host(host: str, dns_server: str = DEFAULT_LAN_DNS_SERVER, timeout_seconds: float = 3.0) -> tuple[str, str]:
    """Return (IPv4 address, how it was resolved). Raises OSError when neither resolver knows the name."""
    try:
        return socket.gethostbyname(host), "system resolver"
    except OSError:
        pass
    address = _query_a_record(host, dns_server, timeout_seconds)
    if address is None:
        raise OSError(f"{host!r} resolves neither through the system resolver nor through {dns_server}")
    return address, f"LAN DNS server {dns_server}"
