# SPDX-License-Identifier: BSD-2-Clause
"""Reference row placement for BONS v4, independent of compute-pipe count.

This is a deterministic capacity/transport planner, NOT the final bandwidth
striping policy or executable command scheduler. Runtime reservations must be
supplied explicitly; a weights-only placement is not proof that inference fits.
"""
from dataclasses import asdict, dataclass, replace
import hashlib

from trixium.capabilities import Access, Feature, RegionAddress, Reservation
from trixium.model.bons4 import align, _uint


@dataclass(frozen=True)
class Span:
    tensor_id: int
    row_start: int
    row_count: int
    region_id: int
    offset: int
    payload_bytes: int
    allocated_bytes: int  # Tail zero padding for aligned transfers; never logical weights.

    def __post_init__(self):
        for field in ('tensor_id', 'region_id'):
            _uint(getattr(self, field), 32, field)
        for field in ('row_start', 'row_count', 'offset', 'payload_bytes', 'allocated_bytes'):
            _uint(getattr(self, field), 64, field)
        if not self.row_count or not self.payload_bytes or self.allocated_bytes < self.payload_bytes:
            raise ValueError('empty or undersized placement span')


@dataclass(frozen=True)
class Placement:
    directory_sha256: bytes
    capabilities_sha256: bytes
    spans: tuple[Span, ...]
    runtime_reservations: tuple[Reservation, ...] = ()
    topology_only: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'spans', tuple(self.spans))
        object.__setattr__(self, 'runtime_reservations', tuple(self.runtime_reservations))
        if any(not isinstance(s, bytes) or len(s) != 32 for s in (self.directory_sha256, self.capabilities_sha256)):
            raise ValueError('placement digests must be 32 bytes')
        if type(self.topology_only) is not bool:
            raise ValueError('topology_only must be boolean')

    def validate(self, image, capabilities, *, for_transfer=False):
        if self.directory_sha256 != image.directory_sha256 or self.capabilities_sha256 != hashlib.sha256(capabilities.to_bytes()).digest():
            raise ValueError('placement refers to a different image/device capability blob')
        if for_transfer and self.topology_only:
            raise ValueError('topology-only placement cannot authorize transfers')
        caps = replace(capabilities, reservations=capabilities.reservations + self.runtime_reservations)
        if not self.topology_only and not caps.features & Feature.REGION_IO:
            raise ValueError('placement requires a region-addressed I/O backend')
        regions = {r.id: r for r in caps.regions}
        by_tensor = {t.id: [] for t in image.entries}
        for s in self.spans:
            if s.tensor_id not in by_tensor or s.region_id not in regions:
                raise ValueError('placement references an unknown tensor/region')
            t, r = image.entries[s.tensor_id], regions[s.region_id]
            if (s.payload_bytes != s.row_count * t.row_bytes
                    or s.allocated_bytes != align(s.payload_bytes, r.alignment)
                    or s.offset % r.alignment or s.offset + s.allocated_bytes > r.size):
                raise ValueError('placement span geometry/alignment/range mismatch')
            for reserved in caps.reservations:
                if (reserved.region_id == r.id and s.offset < reserved.offset + reserved.size
                        and reserved.offset < s.offset + s.allocated_bytes):
                    raise ValueError('placement overlaps reserved memory')
            if not self.topology_only:
                caps.resolve(RegionAddress(r.id, s.offset), s.allocated_bytes, Access.READ | Access.WRITE)
            by_tensor[t.id].append(s)
        for t in image.entries:
            cursor = 0
            for s in sorted(by_tensor[t.id], key=lambda s: s.row_start):
                if s.row_start != cursor:
                    raise ValueError('missing/duplicate logical rows')
                cursor += s.row_count
            if cursor != t.rows:
                raise ValueError('incomplete tensor row coverage')
        ordered = sorted(self.spans, key=lambda s: (s.region_id, s.offset))
        for a, b in zip(ordered, ordered[1:]):
            if a.region_id == b.region_id and a.offset + a.allocated_bytes > b.offset:
                raise ValueError('physical placement spans overlap')
        return True

    def to_dict(self):
        return {'version': 1, 'policy': 'first-fit-whole-rows-v1',
                'directory_sha256': self.directory_sha256.hex(),
                'capabilities_sha256': self.capabilities_sha256.hex(),
                'topology_only': self.topology_only,
                'scope': 'weights and explicitly listed reserves; not inference-fit qualification',
                'payload_bytes': sum(s.payload_bytes for s in self.spans),
                'allocated_bytes': sum(s.allocated_bytes for s in self.spans),
                'runtime_reservations': [asdict(r) for r in self.runtime_reservations],
                'spans': [asdict(s) for s in self.spans]}

    @classmethod
    def from_dict(cls, data, image, capabilities, *, for_transfer=False):
        """Load and revalidate a placement; do not trust its cached totals/flags."""
        if not isinstance(data, dict):
            raise ValueError('placement must be an object')
        try:
            result = cls(bytes.fromhex(data['directory_sha256']),
                         bytes.fromhex(data['capabilities_sha256']),
                         tuple(Span(**s) for s in data['spans']),
                         tuple(Reservation(**r) for r in data['runtime_reservations']),
                         data['topology_only'])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError('invalid placement document') from e
        if data != result.to_dict():
            raise ValueError('unsupported/inconsistent placement document')
        result.validate(image, capabilities, for_transfer=for_transfer)
        return result


def plan_weights(image, capabilities, *, runtime_reservations=(), topology_only=False):
    """Place whole logical rows into actual free physical regions, without I/O.

    No model/pipeline quantization changes or compute assignment are made. The
    explicit topology_only option permits offline planning for legacy hardware
    that describes memory but cannot yet execute region-addressed transfers.
    """
    reserves = tuple(runtime_reservations)
    caps = replace(capabilities, reservations=capabilities.reservations + reserves)
    if not topology_only and not caps.features & Feature.REGION_IO:
        raise ValueError('device does not support region-addressed I/O; use topology_only for estimates')
    free = []
    for r in sorted(caps.regions, key=lambda r: (r.locality, r.id)):
        if not topology_only and r.access & (Access.READ | Access.WRITE) != (Access.READ | Access.WRITE):
            continue
        cursor = 0
        for reserved in sorted((v for v in caps.reservations if v.region_id == r.id), key=lambda v: v.offset):
            if align(cursor, r.alignment) < reserved.offset:
                free.append([r, align(cursor, r.alignment), reserved.offset])
            cursor = reserved.offset + reserved.size
        if align(cursor, r.alignment) < r.size:
            free.append([r, align(cursor, r.alignment), r.size])
    spans = []
    for t in image.entries:
        row = 0
        for interval in free:
            r, start, end = interval
            # Aligned transfer must fit even if a reservation ends mid-beat.
            usable = (end - start) // r.alignment * r.alignment
            count = min(t.rows - row, usable // t.row_bytes)
            if not count:
                continue
            payload = count * t.row_bytes
            allocated = align(payload, r.alignment)
            spans.append(Span(t.id, row, count, r.id, start, payload, allocated))
            interval[1] += allocated
            row += count
            if row == t.rows:
                break
        if row != t.rows:
            raise ValueError(f'{t.name}: no whole-row placement fits remaining free regions')
    result = Placement(image.directory_sha256, hashlib.sha256(capabilities.to_bytes()).digest(),
                       tuple(spans), reserves, topology_only)
    result.validate(image, capabilities)
    return result
