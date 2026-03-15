"""Extract selected members from a remote ZIP using HTTP range requests."""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import urllib.request
import zlib
from dataclasses import dataclass
from pathlib import Path


EOCD_SIGNATURE = b"PK\x05\x06"
CD_SIGNATURE = 0x02014B50
LOCAL_SIGNATURE = 0x04034B50


@dataclass
class ZipEntry:
    name: str
    compression: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def _fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def _content_length(url: str) -> int:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=300) as response:
            size = response.headers.get("Content-Length")
            if size:
                return int(size)
    except Exception:
        pass
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        content_range = response.headers.get("Content-Range")
        if content_range and "/" in content_range:
            return int(content_range.rsplit("/", 1)[1])
        size = response.headers.get("Content-Length")
        if not size:
            raise RuntimeError("Remote server did not provide Content-Length")
        return int(size)


def _find_eocd(url: str, size: int) -> tuple[int, int]:
    window = min(size, 131072)
    tail = _fetch_range(url, size - window, size - 1)
    idx = tail.rfind(EOCD_SIGNATURE)
    if idx < 0:
        raise RuntimeError("EOCD not found in remote ZIP tail")
    absolute = size - window + idx
    cd_size = struct.unpack_from("<I", tail, idx + 12)[0]
    cd_offset = struct.unpack_from("<I", tail, idx + 16)[0]
    return cd_offset, cd_size


def _read_entries(url: str) -> list[ZipEntry]:
    size = _content_length(url)
    cd_offset, cd_size = _find_eocd(url, size)
    directory = _fetch_range(url, cd_offset, cd_offset + cd_size - 1)
    stream = io.BytesIO(directory)
    entries: list[ZipEntry] = []
    while True:
        header = stream.read(46)
        if not header:
            break
        signature = struct.unpack_from("<I", header, 0)[0]
        if signature != CD_SIGNATURE:
            raise RuntimeError(f"Unexpected central-directory signature: {signature:#x}")
        compression = struct.unpack_from("<H", header, 10)[0]
        compressed_size = struct.unpack_from("<I", header, 20)[0]
        uncompressed_size = struct.unpack_from("<I", header, 24)[0]
        name_length = struct.unpack_from("<H", header, 28)[0]
        extra_length = struct.unpack_from("<H", header, 30)[0]
        comment_length = struct.unpack_from("<H", header, 32)[0]
        local_header_offset = struct.unpack_from("<I", header, 42)[0]
        name = stream.read(name_length).decode("utf-8")
        stream.seek(extra_length + comment_length, io.SEEK_CUR)
        entries.append(
            ZipEntry(
                name=name,
                compression=compression,
                compressed_size=compressed_size,
                uncompressed_size=uncompressed_size,
                local_header_offset=local_header_offset,
            )
        )
    return entries


def _extract_entry(url: str, entry: ZipEntry) -> bytes:
    header = _fetch_range(url, entry.local_header_offset, entry.local_header_offset + 30 - 1)
    signature = struct.unpack_from("<I", header, 0)[0]
    if signature != LOCAL_SIGNATURE:
        raise RuntimeError(f"Unexpected local-header signature for {entry.name}: {signature:#x}")
    name_length = struct.unpack_from("<H", header, 26)[0]
    extra_length = struct.unpack_from("<H", header, 28)[0]
    data_start = entry.local_header_offset + 30 + name_length + extra_length
    data_end = data_start + entry.compressed_size - 1
    payload = _fetch_range(url, data_start, data_end)
    if entry.compression == 0:
        return payload
    if entry.compression == 8:
        return zlib.decompress(payload, -zlib.MAX_WBITS)
    raise RuntimeError(f"Unsupported compression method {entry.compression} for {entry.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Remote ZIP URL")
    parser.add_argument("--prefix", default="", help="Optional path prefix filter")
    parser.add_argument("--member", action="append", default=[], help="Exact member path to extract")
    parser.add_argument("--suffix", action="append", default=[], help="Extract members matching suffix")
    parser.add_argument("--output-dir", required=True, help="Directory to write extracted files into")
    parser.add_argument("--list", action="store_true", help="List members and exit")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = _read_entries(args.url)
    if args.prefix:
        entries = [entry for entry in entries if entry.name.startswith(args.prefix)]
    if args.list:
        for entry in entries:
            print(f"{entry.name}\t{entry.compressed_size}\t{entry.uncompressed_size}")
        return 0

    wanted: list[ZipEntry] = []
    exact = set(args.member)
    suffixes = list(args.suffix)
    for entry in entries:
        if entry.name in exact or any(entry.name.endswith(suffix) for suffix in suffixes):
            wanted.append(entry)
    if not wanted:
        raise RuntimeError("No matching members found")

    for entry in wanted:
        data = _extract_entry(args.url, entry)
        destination = output_dir / Path(entry.name).name
        destination.write_bytes(data)
        print(f"{entry.name} -> {destination} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
