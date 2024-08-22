from dataclasses import dataclass, field
from typing import Optional


@dataclass
class UtilsConfig:
    sdpa_kernel: Optional[str] = field(default=None, metadata={"help": "kernel to enable for sdpa (math, mem, flash). None for auto"})
