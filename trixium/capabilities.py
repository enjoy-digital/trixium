# SPDX-License-Identifier: BSD-2-Clause
"""Versioned device facts, independent of board names, model tensors and placement.

The immutable BNCP blob is little-endian and CRC protected. Physical memory
regions are not compute pipes. Describing a region does NOT authorize region
I/O: current legacy engines have no REGION_IO capability and expose zero region
access flags. Legacy v2/v3 placement remains a separate, unchanged interface.
"""
from dataclasses import asdict, dataclass
from enum import IntEnum, IntFlag
import json
from pathlib import Path
import struct
import zlib

MAGIC = b'BNCP'
ABI_MAJOR, ABI_MINOR = 1, 0
HEADER = struct.Struct('<4sHHIIIIIIQQIIII')  # 64 bytes
REGION = struct.Struct('<IIIIQQII')          # 40 bytes
RESERVATION = struct.Struct('<IIQQ')         # 24 bytes
MAX_BYTES = 65536
CRC_OFFSET = 12


class Feature(IntFlag):
    SOFT_RESET = 1 << 0
    VECTOR = 1 << 1
    SEQUENCER = 1 << 2
    DUAL_Q = 1 << 3
    DELTANET_FP32 = 1 << 4
    COHERENT_OVERLAP = 1 << 5
    AXI_TRAFFIC = 1 << 6
    DDR_TRAFFIC = 1 << 7
    REGION_IO = 1 << 8  # Requires a region-aware command/loader backend; not legacy wbase.


class WeightEncoding(IntEnum):
    # IDs 0..4 preserve the legacy tensor-kind IDs. These bits describe hardware
    # weight decoding, NOT every format the CPU can read as ordinary raw memory.
    TERNARY_ROWS = 0
    Q8_ROWS = 1
    F16 = 2
    F32 = 3
    BFP16 = 4
    PTQ1_0 = 5
    Q8_0 = 6
    Q4_0 = 7


class MemoryKind(IntEnum):
    DDR4 = 1
    HBM = 2
    SIMULATION = 3


class Access(IntFlag):
    READ = 1
    WRITE = 2


class ReservationKind(IntEnum):
    CPU = 1
    UPLOAD_STAGING = 2
    DIAGNOSTIC = 3
    SYSTEM = 4


def _uint(value, bits, label):
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < (1 << bits):
        raise ValueError(f'{label} must be an unsigned {bits}-bit integer')


def _range(offset, size, capacity, label):
    _uint(offset, 64, label + ' offset')
    _uint(size, 64, label + ' size')
    if not size or offset > capacity or size > capacity - offset:
        raise ValueError(label + ' is empty or outside its address space')


@dataclass(frozen=True)
class MemoryRegion:
    id: int
    kind: MemoryKind
    endpoint: int  # Physical address space/controller, NOT a compute-pipe index.
    base: int
    size: int
    alignment: int = 32
    locality: int = 0
    access: Access = Access(0)

    def __post_init__(self):
        for name in ('id', 'endpoint', 'alignment', 'locality'):
            _uint(getattr(self, name), 32, 'region ' + name)
        _uint(self.kind, 32, 'region kind')
        _uint(self.access, 32, 'region access')
        MemoryKind(self.kind)
        if int(self.access) & ~int(Access.READ | Access.WRITE):
            raise ValueError('unknown region access flags')
        _range(self.base, self.size, 1 << 64, 'region')
        if not self.alignment or self.alignment & (self.alignment - 1):
            raise ValueError('region alignment must be a power of two')
        if self.base % self.alignment or self.size % self.alignment:
            raise ValueError('region base/size must be aligned')


@dataclass(frozen=True)
class Reservation:
    region_id: int
    kind: ReservationKind
    offset: int
    size: int

    def __post_init__(self):
        _uint(self.region_id, 32, 'reservation region ID')
        _uint(self.kind, 32, 'reservation kind')
        ReservationKind(self.kind)
        _range(self.offset, self.size, 1 << 64, 'reservation')


@dataclass(frozen=True)
class RegionAddress:
    region_id: int
    offset: int

    def __post_init__(self):
        _uint(self.region_id, 32, 'region ID')
        _uint(self.offset, 64, 'region offset')


@dataclass(frozen=True)
class Capabilities:
    sys_hz: int
    compute_hz: int
    compute_pipes: int
    scratchpad_bytes: int
    features: Feature
    weight_encodings: int
    regions: tuple[MemoryRegion, ...]
    reservations: tuple[Reservation, ...] = ()

    def __post_init__(self):
        # Freeze caller-supplied lists as well as the record itself.
        object.__setattr__(self, 'regions', tuple(self.regions))
        object.__setattr__(self, 'reservations', tuple(self.reservations))
        for name in ('sys_hz', 'compute_hz', 'compute_pipes', 'scratchpad_bytes'):
            _uint(getattr(self, name), 32, name)
            if not getattr(self, name):
                raise ValueError(name + ' must be nonzero')
        _uint(self.features, 64, 'features')
        _uint(self.weight_encodings, 64, 'weight encodings')
        if int(self.features) & ~sum(int(f) for f in Feature):
            raise ValueError('unknown mandatory feature bits for ABI 1.0')
        if self.weight_encodings & ~sum(1 << int(e) for e in WeightEncoding):
            raise ValueError('unknown weight encoding bits for ABI 1.0')
        if not self.regions or len({r.id for r in self.regions}) != len(self.regions):
            raise ValueError('missing/duplicate memory region IDs')
        if sum(r.size for r in self.regions) >= 1 << 64:
            raise ValueError('aggregate physical capacity exceeds 64-bit accounting')
        if HEADER.size + len(self.regions)*REGION.size + len(self.reservations)*RESERVATION.size > MAX_BYTES:
            raise ValueError('capability blob exceeds size limit')
        ordered = sorted(self.regions, key=lambda r: (r.endpoint, r.base))
        for a, b in zip(ordered, ordered[1:]):
            if a.endpoint == b.endpoint and a.base + a.size > b.base:
                raise ValueError('physical memory regions overlap')
        by_id = {r.id: r for r in self.regions}
        for r in self.regions:
            if r.access and not self.features & Feature.REGION_IO:
                raise ValueError('region access requires region-aware I/O capability')
        for r in self.reservations:
            if r.region_id not in by_id:
                raise ValueError('reservation refers to missing region')
            _range(r.offset, r.size, by_id[r.region_id].size, 'reservation')
        reserved = sorted(self.reservations, key=lambda r: (r.region_id, r.offset))
        for a, b in zip(reserved, reserved[1:]):
            if a.region_id == b.region_id and a.offset + a.size > b.offset:
                raise ValueError('memory reservations overlap')

    @property
    def physical_bytes(self):
        return sum(r.size for r in self.regions)

    @property
    def unreserved_bytes(self):
        """Capacity arithmetic only; not proof of reachability or model fit."""
        return self.physical_bytes - sum(r.size for r in self.reservations)

    def resolve(self, address, size, access=Access.READ):
        """Validate a region transfer without truncation; return endpoint/address.

        Pure validation: does not submit commands. Legacy adapters MUST NOT use
        this to truncate a 64-bit offset into their 32-bit wbase register.
        """
        if not self.features & Feature.REGION_IO:
            raise ValueError('device does not support region-addressed I/O')
        if not access or int(access) & ~int(Access.READ | Access.WRITE):
            raise ValueError('invalid requested region access')
        region = next((r for r in self.regions if r.id == address.region_id), None)
        if region is None:
            raise ValueError('unknown memory region')
        if region.access & access != access:
            raise ValueError('region does not allow requested access')
        _range(address.offset, size, region.size, 'transfer')
        if address.offset % region.alignment or size % region.alignment:
            raise ValueError('transfer is not region-aligned')
        for r in self.reservations:
            if r.region_id == region.id and address.offset < r.offset + r.size and r.offset < address.offset + size:
                raise ValueError('transfer overlaps reserved memory')
        return region.endpoint, region.base + address.offset

    def to_bytes(self):
        body = b''.join(REGION.pack(r.id, r.kind, r.endpoint, r.access, r.base, r.size,
                                    r.alignment, r.locality) for r in self.regions)
        body += b''.join(RESERVATION.pack(r.region_id, r.kind, r.offset, r.size)
                         for r in self.reservations)
        header = HEADER.pack(MAGIC, ABI_MAJOR, ABI_MINOR, HEADER.size + len(body), 0,
                             self.sys_hz, self.compute_hz, self.compute_pipes, self.scratchpad_bytes,
                             self.features, self.weight_encodings, len(self.regions),
                             len(self.reservations), REGION.size, RESERVATION.size)
        blob = bytearray(header + body)
        struct.pack_into('<I', blob, CRC_OFFSET, zlib.crc32(blob))
        return bytes(blob)

    def write_artifacts(self, directory):
        """Build-time descriptions, not evidence of the currently programmed card."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'capabilities.bin').write_bytes(self.to_bytes())
        (directory / 'capabilities.json').write_text(json.dumps(dict(
            abi_major=ABI_MAJOR, abi_minor=ABI_MINOR, physical_bytes=self.physical_bytes,
            unreserved_bytes=self.unreserved_bytes, **asdict(self)), indent=2) + '\n')

    @classmethod
    def from_bytes(cls, data):
        data = bytes(data)
        if not HEADER.size <= len(data) <= MAX_BYTES:
            raise ValueError('capability length out of bounds')
        (magic, major, minor, length, crc, sys_hz, compute_hz, pipes, sp_bytes,
         features, encodings, nr, ns, rs, ss) = HEADER.unpack_from(data)
        if magic != MAGIC or (major, minor) != (ABI_MAJOR, ABI_MINOR):
            raise ValueError('unknown capability magic/version')
        if length != len(data) or rs != REGION.size or ss != RESERVATION.size:
            raise ValueError('capability length/record size mismatch')
        if HEADER.size + nr*rs + ns*ss != length:
            raise ValueError('truncated/extra capability records')
        check = bytearray(data)
        struct.pack_into('<I', check, CRC_OFFSET, 0)
        if zlib.crc32(check) != crc:
            raise ValueError('capability CRC mismatch')
        regions = []
        for i in range(nr):
            rid, kind, endpoint, flags, base, size, alignment, locality = REGION.unpack_from(data, HEADER.size + i*rs)
            regions.append(MemoryRegion(rid, MemoryKind(kind), endpoint, base, size,
                                        alignment, locality, Access(flags)))
        reservations = tuple(Reservation(rid, ReservationKind(kind), offset, size)
                             for rid, kind, offset, size in RESERVATION.iter_unpack(data[HEADER.size + nr*rs:]))
        return cls(sys_hz, compute_hz, pipes, sp_bytes, Feature(features), encodings,
                   tuple(regions), reservations)


def read_capabilities(registers):
    """Read from a LiteX RemoteClient-style register collection; no legacy guessing.

    Caller exclusively owns the zzcap index register during this read. The
    physical device need not be running inference or loaded with a model.
    """
    required = ('zzcaps_magic', 'zzcaps_abi', 'zzcaps_length', 'zzcaps_index',
                'zzcaps_word', 'zzcaps_valid')
    if not all(hasattr(registers, name) for name in required):
        raise ValueError('bitstream has no versioned capabilities; use explicit legacy configuration')
    if registers.zzcaps_magic.read() != int.from_bytes(MAGIC, 'little'):
        raise ValueError('unexpected capability CSR magic')
    if registers.zzcaps_abi.read() != ABI_MAJOR | (ABI_MINOR << 16):
        raise ValueError('unsupported capability CSR ABI')
    length = registers.zzcaps_length.read()
    if not HEADER.size <= length <= MAX_BYTES or length % 4:
        raise ValueError('invalid capability CSR length')
    words = []
    for index in range(length//4):
        registers.zzcaps_index.write(index)
        if not registers.zzcaps_valid.read():
            raise ValueError('capability CSR selector rejected')
        words.append(registers.zzcaps_word.read())
    return Capabilities.from_bytes(struct.pack('<' + 'I'*len(words), *words))
