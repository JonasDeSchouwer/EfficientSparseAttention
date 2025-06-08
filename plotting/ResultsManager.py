from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Union

from matplotlib import pyplot as plt
import numpy as np
from DisplaySettings import DisplaySettings

@dataclass
class Results:
    """Dataclass holding profiling results for a specific algorithm."""
    label: Optional[str] = None
    Ns: Optional[List[int]] = None
    forward_means: Optional[np.ndarray] = None
    forward_stds: Optional[np.ndarray] = None
    backward_means: Optional[np.ndarray] = None
    backward_stds: Optional[np.ndarray] = None
    total_means: Optional[np.ndarray] = None
    total_stds: Optional[np.ndarray] = None
    peak_means: Optional[np.ndarray] = None
    peak_stds: Optional[np.ndarray] = None

    def print_table(self):
        """Print a table of the results, in LaTeX formatting."""

        lines = []

        # Helper function to format time values
        def format_time(seconds):
            if seconds < 1:  # Less than 1s
                return f"{seconds * 1000:.2f} ms"
            else:
                return f"{seconds:.3f} s"

        # Print data rows
        for N, forward, backward, total, peak in zip(
            self.Ns,
            self.forward_means,
            self.backward_means,
            self.total_means,
            self.peak_means
        ):
            # Format N as 10^x
            N_exp = np.log10(N)
            N_str = f"$\\bs{{10^{{{N_exp:.1f}}}}}$"
            
            # Format all values
            forward_time = format_time(forward)
            backward_time = format_time(backward)
            total_time = format_time(total)
            peak_mem = f"{peak:.2f}"

            lines.append(f"        {N_str} & - & {forward_time} & {backward_time} & {total_time} & {peak_mem} \\\\")

        print("\n".join(lines))


class ResultsManager:
    """Manages profiling results for multiple algorithms."""
    
    def __init__(self, display_settings: Optional[DisplaySettings] = None):
        self._results: Dict[str, Results] = {}
        self._display_settings = display_settings or DisplaySettings()

    @property
    def results(self) -> Dict[str, Results]:
        """Dictionary mapping algorithm names to their corresponding Results."""
        return self._results

    @property
    def display_settings(self) -> DisplaySettings:
        """Settings controlling how results are displayed."""
        return self._display_settings

    @display_settings.setter
    def display_settings(self, settings: DisplaySettings):
        """Set new display settings."""
        self._display_settings = settings

    def __getitem__(self, key: str) -> Results:
        """Get results for a specific algorithm."""
        return self._results[key]

    def __setitem__(self, key: str, value: Results):
        """Set results for a specific algorithm."""
        self._results[key] = value

    def _plot_means_std(self, means_attr: str, stds_attr: str, ylabel):
        """Plot means and standard deviations of the results on the current figure."""

        to_plot = self.results.items()
        if not self.display_settings.display_flash:
            to_plot = [r for r in to_plot if r[0] != "flash"]

        for algorithm, results in to_plot:
            means = np.array(getattr(results, means_attr))
            stds = np.array(getattr(results, stds_attr))
            Ns = np.array(results.Ns)

            if self.display_settings.max_N:
                # Find the largest index where Ns <= max_N
                num_Ns_under_max_N = sum(Ns <= self.display_settings.max_N)
                Ns = Ns[:num_Ns_under_max_N]
                means = means[:num_Ns_under_max_N]
                stds = stds[:num_Ns_under_max_N]

            color = self.display_settings.colors[algorithm]
            marker = self.display_settings.markers[algorithm]
            linestyle = self.display_settings.linestyles[algorithm]
            plt.plot(Ns, means, label=results.label, marker=marker, color=color)
            plt.fill_between(Ns, means - stds, means + stds, alpha=0.2, color=color)
        
        plt.xscale("log")
        plt.yscale("log")
        if self.display_settings.ylim:
            plt.ylim(self.display_settings.ylim)
        plt.xlabel("$N$", fontsize=self.display_settings.axis_fontsize)
        plt.ylabel(ylabel, fontsize=self.display_settings.axis_fontsize)

    def _show_means_std(self, means_attr: str, stds_attr: str, ylabel, save_path: Optional[Path] = None, show_legend=False):
        """Plot means and standard deviations of the results on a new figure."""
        fig = plt.figure()
        self._plot_means_std(means_attr, stds_attr, ylabel)


        if show_legend:
            legend_y = 1.45 if self.display_settings.display_flash else 1.35
            plt.legend(loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=1, fontsize=self.display_settings.legend_fontsize)
            
        if save_path:
            plt.savefig(save_path, bbox_inches="tight")

        plt.show()

    def show_forward(self, save_path: Optional[Path] = None, show_legend=False):
        self._show_means_std("forward_means", "forward_stds", "Time (s)", save_path, show_legend)

    def show_backward(self, save_path: Optional[Path] = None, show_legend=False):
        self._show_means_std("backward_means", "backward_stds", "Time (s)", save_path, show_legend)

    def show_total(self, save_path: Optional[Path] = None, show_legend=False):
        self._show_means_std("total_means", "total_stds", "Time (s)", save_path, show_legend)

    def show_peak(self, save_path: Optional[Path] = None, show_legend=False):
        self._show_means_std("peak_means", "peak_stds", "Memory (MB)", save_path, show_legend)