from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence, Self, List
import copy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

@dataclass
class CISeries:
    """Single confidence-interval series for plotting."""
    label: str
    paths: Sequence[str]
    varfield: str
    CIfield: str
    color: str
    varfieldfn: Callable = field(default=lambda x: x)
    alpha: float = None
    linestyle: str = "-"
    hist_linestyle: str = "-"
    linewidth: float = 1.5
    markersize: float = 10
    transparency: float = 0.1
    line_transparency: float = 1
    
    def __post_init__(self):
        if self.varfield == 'alpha': 
            if self.alpha is not None:
                raise ValueError("alpha parameter should not be specified when varfield == 'alpha'")
            if len(self.paths) > 1:
                raise ValueError("only one path shoud be specified when varfield == 'alpha'")
            with np.load(self.paths[0]) as data:
                self.vars = self.varfieldfn(1 - np.array(data['alpha'])) # always plot against 1-alpha
                self.lowers, self.means, self.uppers = data[self.CIfield]
        else: 
            if self.alpha is None:
                raise ValueError("alpha parameter needs to be specified when varfield != 'alpha'")
            self.vars = []
            self.lowers = []
            self.means = []
            self.uppers = []
            for path in self.paths:
                with np.load(path) as data:
                    self.vars.append(self.varfieldfn(data[self.varfield]))
                    
                    alpha_idxs = np.isclose(data['alpha'], self.alpha)
                    assert(alpha_idxs.sum() == 1)
                    ls, ms, us = np.einsum('ijk,j -> ik', data[self.CIfield], alpha_idxs)
                    
                    self.lowers.append(ls)
                    self.means.append(ms)
                    self.uppers.append(us)
            
            self.lowers = np.array(self.lowers)
            self.means = np.array(self.means)
            self.uppers = np.array(self.uppers)

def concat_CIseries(series_list: List[CISeries]):
    new_series = copy.deepcopy(series_list[0])
    new_series.lowers = np.concat([series.lowers.swapaxes(0,1) for series in series_list]).swapaxes(0,1)
    new_series.means = np.concat([series.means.swapaxes(0,1) for series in series_list]).swapaxes(0,1)
    new_series.uppers = np.concat([series.uppers.swapaxes(0,1) for series in series_list]).swapaxes(0,1)
    return new_series

def plot_CI_trend_fn(
    fn,
    series: Iterable[CISeries],
    ax: Optional[Axes] = None,
    figsize = (5, 6),
    fontsize = 10,
    guides: Sequence = (),
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    show_legend = True,
):
    """
    Plot trend of fn of CISeries
    
    :param fn: (l,m,u) -> val function that takes in the lower end of CI, the mean value and the upper end of CI and output a value
    :param series: a list of CISeries objects to be plotted
    """
    created_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_ax = True
    
    for guide in guides:
        ax.plot(series[0].vars, guide(series[0].vars), ":", color="black")  

    plot_vals = []
    for s in series:
        vals = [[fn(l,m,u) for l, m, u in zip(ls, ms, us)] for ls, ms, us in zip(s.lowers, s.means, s.uppers)]
        mean_vals = np.array([np.mean(val) for val in vals]).clip(-1e6,1e6)
        ste_vals = np.array([np.std(val) / np.sqrt(len(val)) for val in vals]).clip(-1e6,1e6)

        ax.plot(s.vars, mean_vals, s.linestyle, color=s.color, linewidth=s.linewidth, label=s.label, markersize=s.markersize, alpha=s.line_transparency)
        ax.fill_between(s.vars, mean_vals-ste_vals, mean_vals+ste_vals, color=s.color, alpha=s.transparency, lw=0)
        plot_vals.append((s.vars, mean_vals, ste_vals))

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if save_path and created_ax:
        if show_legend:
            ax.legend(fontsize=fontsize)
        ax.figure.savefig(f'figs/{save_path}')
    return ax, plot_vals

def plot_CI_freq_fn(
    fn,
    var_val,
    series: Iterable[CISeries],
    ax: Optional[Axes] = None,
    figsize = (5, 6),
    fontsize = 10,
    xlim: Optional[Sequence[float]] = None,
    ylim: Optional[Sequence[float]] = None,
    save_path: Optional[str] = None,
    num_bins: int = None,
    transparency: float = 0.1
):
    """
    Plot frequency of fn of a particular point of CISeries
    
    :param fn: (l,m,u) -> val function that takes in the lower end of CI, the mean value and the upper end of CI and output a value
    :param series: a list of CISeries objects to be plotted
    """
    created_ax = False
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
        created_ax = True
    for s in series:
        vars_arr = np.array(s.vars)
        idx = int(np.argmin(np.abs(vars_arr - var_val)))
        vals = np.array([fn(l, m, u) for l, m, u in zip(s.lowers[idx], s.means[idx], s.uppers[idx])])
        val_mean = vals.mean()

        # fill
        fill_color = list(mcolors.to_rgba(s.color))
        fill_color[3] = transparency
        ax.hist(vals, bins=num_bins, color=fill_color, density=True)
        # outline
        ax.hist(vals, bins=num_bins, color=s.color, density=True, histtype='step', linestyle=s.hist_linestyle)
        # label 
        ax.hist(-1e6, bins=1, color=fill_color, density=True, edgecolor=s.color, linewidth=1.5, label=s.label, linestyle=s.hist_linestyle)
        ax.vlines([val_mean], 0, 1e6, linestyles="--", colors=s.color)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    else:
        ax.set_xlim(vals.min(), vals.max())
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if save_path and created_ax:
        ax.legend(fontsize=fontsize)
        ax.figure.savefig(f'figs/{save_path}', bbox_inches='tight')
    return ax


# def plot_power_bands(
#     x: Sequence[float],
#     series: Iterable[PowerSeries],
#     *,
#     gaussian_pp: float = 1.96,
#     figsize=(5, 6),
#     xlim: Optional[Sequence[float]] = None,
#     ylim: Optional[Sequence[float]] = (-0.05, 1.05),
#     save_path: Optional[str] = None,
# ):
#     plt.figure(figsize=figsize)
#     x_arr = jnp.array(x)
#     for s in series:
#         mean = jnp.array(s.mean)
#         ste = jnp.array(s.ste)
#         plt.plot(x_arr, mean, s.linestyle, color=s.color, linewidth=s.linewidth, label=s.label)
#         plt.fill_between(
#             x_arr,
#             mean - gaussian_pp * ste,
#             mean + gaussian_pp * ste,
#             color=s.color,
#             alpha=s.alpha,
#         )

#     plt.legend(fontsize=10)
#     if xlim is not None:
#         plt.xlim(xlim[0], xlim[1])
#     if ylim is not None:
#         plt.ylim(ylim[0], ylim[1])
#     if save_path:
#         plt.savefig(save_path)
#     return plt.gca()
