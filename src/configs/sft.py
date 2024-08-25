from dataclasses import dataclass, field
from typing import Optional
import trl


@dataclass
class SFTConfig(trl.SFTConfig):
    sdpa_kernel: Optional[str] = field(default=None, metadata={"help": "kernel to enable for sdpa (math, mem, flash). None for auto"})
