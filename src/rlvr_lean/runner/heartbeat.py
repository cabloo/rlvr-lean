"""Write TensorBoard event files with the standard library only.

Why: a job runner may stop a task whose `out/tb/events.out.tfevents*` has not changed for a while (ours: 90
minutes), and the box-side entry runs on a bare interpreter before any environment exists, so it cannot import
TensorBoard. A record is the TFRecord framing (length, masked CRC-32C of the length, payload, masked CRC
of the payload) around a hand-encoded `Event` protobuf carrying scalar summaries.
"""

from __future__ import annotations

import socket
import struct
import time
from pathlib import Path


def _crc32c_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x82F63B78 if crc & 1 else crc >> 1
        table.append(crc)
    return table


_TABLE = _crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for byte in data:
        crc = _TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def _masked_crc(data: bytes) -> int:
    crc = _crc32c(data)
    return ((((crc >> 15) | (crc << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _length_delimited(field_number: int, payload: bytes) -> bytes:
    return _varint((field_number << 3) | 2) + _varint(len(payload)) + payload


def _event(wall_time: float, step: int, *, file_version: str | None = None, scalars: dict[str, float] | None = None) -> bytes:
    body = b"\x09" + struct.pack("<d", wall_time) + b"\x10" + _varint(step)          # wall_time, step
    if file_version is not None:
        body += _length_delimited(3, file_version.encode())
    if scalars:
        values = b"".join(
            _length_delimited(1, _length_delimited(1, tag.encode()) + b"\x15" + struct.pack("<f", float(value)))
            for tag, value in scalars.items())
        body += _length_delimited(5, values)                                           # Event.summary
    return body


class ScalarEventWriter:
    """Appends scalar events to one `events.out.tfevents.*` file under `directory`."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"events.out.tfevents.{int(time.time())}.{socket.gethostname()}"
        self._write(_event(time.time(), 0, file_version="brain.Event:2"))

    def _write(self, payload: bytes) -> None:
        header = struct.pack("<Q", len(payload))
        record = header + struct.pack("<I", _masked_crc(header)) + payload + struct.pack("<I", _masked_crc(payload))
        with self.path.open("ab") as handle:
            handle.write(record)

    def scalars(self, step: int, values: dict[str, float]) -> None:
        self._write(_event(time.time(), step, scalars=values))
