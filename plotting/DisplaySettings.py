from dataclasses import dataclass
from typing import Dict, Any, List, Optional

@dataclass
class DisplaySettings:
    """Settings for controlling how results are displayed in plots and outputs."""
    # Default settings that can be overridden
    colors = {
        'full': 'C3',
        'naive': 'C2',
        'sym': 'C0',
        'flash': 'C4',
    }
    legend_fontsize: int = 15
    axis_fontsize: int = 15
    ylim: Optional[List[float]] = None
    display_flash: bool = True
    max_N: Optional[int] = None
    