# SPDX-License-Identifier: BSD-2-Clause
"""Read-only capability ROM and shared-cluster feature derivation."""
from migen import Memory, Mux
from litex.gen import LiteXModule
from litex.soc.interconnect.csr import CSRStatus, CSRStorage

from trixium.capabilities import ABI_MAJOR, ABI_MINOR, MAGIC, Capabilities, Feature, WeightEncoding


class BonsaiCapabilitiesCSR(LiteXModule):
    def __init__(self, capabilities):
        blob = capabilities.to_bytes()
        words = [int.from_bytes(blob[i:i+4], 'little') for i in range(0, len(blob), 4)]
        self.magic = CSRStatus(32, reset=int.from_bytes(MAGIC, 'little'))
        self.abi = CSRStatus(32, reset=ABI_MAJOR | (ABI_MINOR << 16))
        self.length = CSRStatus(32, reset=len(blob))
        self.index = CSRStorage(32, description='Read-only capability ROM word index; serialize callers.')
        self.word = CSRStatus(32)
        self.valid = CSRStatus(1)
        mem = Memory(32, len(words), init=words)
        port = mem.get_port(async_read=True)
        self.specials += mem, port
        self.comb += [port.adr.eq(self.index.storage),
                      self.valid.status.eq(self.index.storage < len(words)),
                      self.word.status.eq(Mux(self.valid.status, port.dat_r, 0))]


def add_capabilities(soc, config, memory_ports, regions, reservations=(), *, sys_hz):
    """Install after cluster and optional traffic monitors; never infer a board.

    The experimental native projection CSR path is not a complete region-I/O
    runtime: uploader and execution lowering are still missing. Keep the public
    contract conservative until those interfaces are implemented and qualified;
    physical capacity or an opt-in target flag alone must not advertise them.
    """
    if hasattr(soc, 'zzcaps'):
        raise ValueError('capabilities already installed')
    features = Feature.SOFT_RESET | Feature.COHERENT_OVERLAP
    if config.with_vecunit:
        features |= Feature.VECTOR | Feature.SEQUENCER
    if config.with_dual:
        features |= Feature.DUAL_Q
    if config.with_dnrec:
        features |= Feature.DELTANET_FP32
    if hasattr(soc, 'traffic_axi0'):
        features |= Feature.AXI_TRAFFIC
    if hasattr(soc, 'traffic_ddr0'):
        features |= Feature.DDR_TRAFFIC
    # Target supplies the configured frequency, not merely the requested one.
    # The current eng domain is the sys clock; independent clocks are future work.
    caps = Capabilities(sys_hz, sys_hz, len(memory_ports),
                        config.scratchpad_size, features,
                        (1 << WeightEncoding.TERNARY_ROWS) | (1 << WeightEncoding.Q8_ROWS),
                        regions, reservations)
    # Alphabetically after ztraffic: existing CSR bank positions stay unchanged.
    soc.zzcaps = BonsaiCapabilitiesCSR(caps)
    return caps
