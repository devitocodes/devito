import numpy as np
from math import log, floor, ceil
from os import path, makedirs
from collections import namedtuple, defaultdict, OrderedDict
from collections.abc import Mapping

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    # The below is needed on certain clusters
    # mpl.use("Agg")
    from matplotlib.ticker import FormatStrFormatter

    # Adjust font size
    font = {'size': 10}
    mpl.rc('font', **font)
except:
    mpl = None
    plt = None
try:
    import brewer2mpl as b2m
except ImportError:
    b2m = None
from .utils import bench_print


__all__ = ['Plotter', 'LinePlotter', 'RooflinePlotter', 'BarchartPlotter',
           'AxisScale']


def scale_limits(minval, maxval, base, type='log'):
    """
    Compute axis values from min and max values.
    """

    if type == 'log':
        basemin = floor(log(minval, base))
        basemax = ceil(log(maxval, base))
    else:
        basemin = floor(float(minval) / base)
        basemax = ceil(float(maxval) / base)
    nvals = basemax - basemin + 1
    dtype = np.float32
    basevals = np.linspace(basemin, basemax, nvals, dtype=dtype)
    if type == 'log':
        return dtype(base) ** basevals
    else:
        return dtype(base) * basevals


class AxisScale(object):
    """
    Utility class to describe and configure axis value labelling.
    """
    def __init__(self, scale='log', base=2., dtype=np.float32,
                 minval=None, maxval=None):
        self.scale = scale
        self.base = base
        self.dtype = dtype
        self.minval = minval
        self.maxval = maxval

        self._values = []

    @property
    def values(self):
        minv, maxv = min(self._values), max(self._values)
        minv = minv if self.minval is None else min(minv, self.minval)
        maxv = maxv if self.maxval is None else max(maxv, self.maxval)
        return scale_limits(minval=minv, maxval=maxv, base=self.base, type=self.scale)


class Plotter(object):
    """
    Plotting utility that provides data and basic diagram utilities.
    """
    figsize = (6, 4)
    dpi = 300
    marker = ['D', 'o', '^', 'v']

    if b2m is not None:
        color = b2m.get_map('Set2', 'qualitative', 6).hex_colors
    else:
        color = ['r', 'b', 'g', 'y']

    fonts = {'title': 7, 'axis': 8, 'minorticks': 3, 'legend': 7}

    def __init__(self, plotdir='plots'):
        if mpl is None or plt is None:
            bench_print("Matplotlib/PyPlot not found - unable to plot.")
            raise ImportError("Could not import matplotlib or pyplot")
        self.plotdir = plotdir

    def create_figure(self, figname):
        fig = plt.figure(figname, figsize=self.figsize, dpi=self.dpi)
        ax = fig.add_subplot(111)
        return fig, ax

    def set_xaxis(self, axis, label, values=None, dtype=np.float32):
        if values is not None:
            values = np.array(values).astype(dtype)
            axis.set_xlim(values[0], values[-1])
            axis.set_xticks(values)
            axis.set_xticklabels(values, fontsize=self.fonts['axis'])
        axis.set_xlabel(label, fontsize=self.fonts['axis'])

    def set_yaxis(self, axis, label, values=None, dtype=np.float32):
        if values is not None:
            values = np.array(values).astype(dtype)
            axis.set_ylim(values[0], values[-1])
            axis.set_yticks(values)
            axis.set_yticklabels(values, fontsize=self.fonts['axis'])
        axis.set_ylabel(label, fontsize=self.fonts['axis'])

    def save_figure(self, figure, figname):
        if not path.exists(self.plotdir):
            makedirs(self.plotdir)
        figpath = path.join(self.plotdir, figname)
        bench_print("Plotting %s " % figpath)
        figure.savefig(figpath, format='pdf', facecolor='white',
                       orientation='landscape', bbox_inches='tight')


class LinePlotter(Plotter):
    """
    Line plotter for generating scaling or error-cost plots.

    Parameters
    ----------
    figname : str
        Name of output file.
    plotdir : str
        Directory to store the plot in.
    title : str
        Plot title to be printed on top.

    Example usage:

    with LinePlotter(figname=..., plotdir=...) as plot:
        plot.add_line(y_values, x_values, label='Strong scaling')
    """

    def __init__(self, figname='plot', plotdir='plots', title=None,
                 plot_type='loglog', xscale=None, yscale=None,
                 xlabel=None, ylabel=None, legend=None,
                 yscale2=None, ylabel2=None):
        super(LinePlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = {'loc': 'best', 'ncol': 2,
                       'fancybox': True, 'fontsize': 10}
        self.legend.update(legend or {})  # Add user arguments to defaults
        self.plot_type = plot_type
        self.xlabel = xlabel or 'Number of processors'
        self.ylabel = ylabel or 'Wall time (s)'
        self.xscale = xscale or AxisScale(scale='log', base=2.)
        self.yscale = yscale or AxisScale(scale='log', base=2.)
        self.yscale2 = yscale2
        self.ylabel2 = ylabel2

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        self.plot = getattr(self.ax, self.plot_type)
        if self.title is not None:
            self.ax.set_title(self.title)

        if self.yscale2:
            self.ax2 = self.ax.twinx()
        return self

    def __exit__(self, *args):
        # Set axis labelling and generate plot file
        self.set_xaxis(self.ax, self.xlabel, values=self.xscale.values,
                       dtype=self.xscale.dtype)
        self.set_yaxis(self.ax, self.ylabel, values=self.yscale.values,
                       dtype=self.yscale.dtype)
        if self.yscale2:
            self.set_yaxis(self.ax2, self.ylabel2,
                           values=self.yscale2.values,
                           dtype=self.yscale2.dtype)

        # Add legend if labels were used
        lines, labels = self.ax.get_legend_handles_labels()
        if self.yscale2:
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        if len(lines) > 0:
            self.ax.legend(lines, labels, **self.legend)

        self.save_figure(self.fig, self.figname)

    def add_line(self, xvalues, yvalues, label=None, style=None,
                 annotations=None, secondary=False):
        """
        Adds a single line to the plot of from a set of measurements

        Parameters
        ----------
        yvalue :
            List of Y values of the  measurements
        xvalue :
            List of X values of the  measurements
        label :
            Optional legend label for data line
        style :
            Plotting style to use, defaults to black line ('-k')
        annotations:
            Point annotation strings to be place next
            to each point on the line.
        """
        style = style or 'k-'

        # Update mai/max values for axis limits
        self.xscale._values += xvalues
        if secondary:
            self.yscale2._values += yvalues
            self.ax2.semilogx(xvalues, yvalues, style, label=label, linewidth=2)
        else:
            self.yscale._values += yvalues
            self.plot(xvalues, yvalues, style, label=label, linewidth=2)

        # Add point annotations
        if annotations:
            for x, y, a in zip(xvalues, yvalues, annotations):
                plt.annotate(a, xy=(x, y), xytext=(4, 2),
                             textcoords='offset points', size=6)


class BarchartPlotter(Plotter):
    """
    Barchart plotter for generating direct comparison plots.

    Parameters
    ----------
    figname : str
        Name of output file
    plotdir : str
        Directory to store the plot in
    title : str
        Plot title to be printed on top

    Example usage:

    with BarchartPlotter(figname=..., plotdir=...) as barchart:
        barchart.add_point(gflops[0], oi[0], label='Point A')
        barchart.add_point(gflops[1], oi[1], label='Point B')
    """

    def __init__(self, figname='barchart', plotdir='plots',
                 title=None):
        super(BarchartPlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        self.legend = {}
        self.values = defaultdict(dict)

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        if self.title is not None:
            self.ax.set_title(self.title)
        return self

    def __exit__(self, *args):
        # Set axis labelling and generate plot file
        # self.ax.set_xticks(x_indices + width)
        # self.ax.set_xticklabels(self.values.keys())
        self.set_yaxis(self.ax, 'Runtime (s)')
        self.ax.legend(self.legend, loc='best', ncol=2,
                       fancybox=True, fontsize=10)
        self.save_figure(self.fig, self.figname)

    def add_value(self, value, grouplabel=None, color=None, label=None):
        """
        Adds a single point measurement to the barchart plot

        Parameters
        ----------
        value : str
            Y-value of the given point measurement
        grouplabel : str
            Group label to be put on the X-axis
        color : str
            Optional plotting color for data point
        label : str
            Optional legend label for data point
        """
        # Record all points keyed by group and legend labels
        self.values[grouplabel][label] = value

        # Record legend labels to avoid replication
        if label is not None:
            self.legend[label] = color


class RooflinePlotter(Plotter):
    """
    Roofline plotter for generating generic roofline plots.

    Parameters
    ----------
    figname : str
        Name of output file
    plotdir : str
        Directory to store the plot in
    title : str
        Plot title to be printed on top
    max_bw : float
        Maximum achievable memory bandwidth in GB/s.
        This defines the slope of the roofline.
    flop_ceils : tuple(float, str)
        Represents the maximum achievable performance
        in GFlops/s; the str indicates how the performance
        ceil was obtained (e.g., ideal peak, linpack)
    with_yminorticks : bool, optional
        Show minor ticks on yaxis.
    fancycolors : bool, optional
        Use beautiful colors, using the user-provided
        colors as key to establish a 1-to-1 mapping
        between user-provided colors and the new ones.
    legend : str, optional
        Additional arguments for legend entries, default:
        {loc='best', ncol=2, fancybox=True, fontsize=10}.
        Pass the string ``'drop'`` to show no legend.

    Example usage:

    with RooflinePlotter(figname=..., plotdir=...,
                         max_bw=..., flop_ceils=...) as roofline:
        roofline.add_point(gflops[0], oi[0], label='Point A')
        roofline.add_point(gflops[1], oi[1], label='Point B')
    """

    def __init__(self, figname='roofline', plotdir='plots', title=None,
                 max_bw=None, flop_ceils=None, with_yminorticks=False,
                 fancycolor=False, legend=None):

        super(RooflinePlotter, self).__init__(plotdir=plotdir)
        self.figname = figname
        self.title = title
        if legend != 'drop':
            self.legend = {'loc': 'best', 'ncol': 2, 'fancybox': True, 'fontsize': 10}
            self.legend.update(legend)  # Add user arguments to defaults
        else:
            self.legend = None
        self.legend_map = {}  # Label -> style map for legend entries

        self.max_bw = max_bw
        self.flop_ceils = OrderedDict(sorted(flop_ceils, key=lambda i: i[0]))
        self.max_flops = max(self.flop_ceils)

        self.xvals = [float(self.max_flops) / max_bw]
        self.yvals = [self.max_flops]
        self.with_yminorticks = with_yminorticks
        if fancycolor is True:
            self.fancycolor = ColorTracker({}, list(self.color))
        else:
            self.fancycolor = None

        # A set of OI values for which to add dotted lines
        self.oi_lines = []

    def __enter__(self):
        self.fig, self.ax = self.create_figure(self.figname)
        if self.title is not None:
            self.ax.set_title(self.title, {'fontsize': self.fonts['title']})
        return self

    def __exit__(self, *args):
        # Scale axis limits
        self.xvals = scale_limits(min(1.0, *self.xvals),
                                  max(64, *self.xvals), base=2., type='log')
        self.yvals = scale_limits(min(16.0, *self.yvals),
                                  max(self.yvals), base=2., type='log')

        # Add a dotted lines for stored OI values
        for oi in self.oi_lines:
            self.ax.plot([oi, oi], [1., min(oi * self.max_bw, self.max_flops)], 'k:')

        # Add the rooflines
        for i, j in self.flop_ceils.items():
            y_roofl = self.xvals * self.max_bw
            y_roofl[y_roofl > i] = i
            idx = (y_roofl >= i).argmax()
            x_roofl = np.insert(self.xvals, idx, i / self.max_bw)
            y_roofl = np.insert(y_roofl, idx, i)
            self.ax.loglog(x_roofl, y_roofl, 'k-')
            self.ax.annotate(**{'xy': (x_roofl[-1] - len(j), y_roofl[idx+1]),
                                'xytext': (-24, 2), 'textcoords': 'offset points',
                                'size': 7, 's': j})

        # Set axis labelling and generate plot file
        xlabel = 'Operational intensity (FLOPs/Byte)'
        ylabel = 'Performance (GFLOPs/s)'
        self.set_xaxis(self.ax, xlabel, values=self.xvals, dtype=np.int32)
        self.set_yaxis(self.ax, ylabel, values=self.yvals, dtype=np.int32)
        if self.legend is not None:
            self.ax.legend(**self.legend)
        self.save_figure(self.fig, self.figname)

    def set_yaxis(self, axis, label, values=None, dtype=np.float32):
        super(RooflinePlotter, self).set_yaxis(axis, label, values, dtype=np.float32)
        if values is not None:
            axis.yaxis.set_major_formatter(FormatStrFormatter("%d"))
            if self.with_yminorticks is True:
                axis.tick_params(axis='y', which='minor',
                                 labelsize=self.fonts['minorticks'])
                axis.yaxis.set_minor_formatter(FormatStrFormatter("%d"))
            else:
                axis.minorticks_off()

    def _select_point_color(self, usercolor):
        if usercolor is None:
            return self.color[0]
        elif not self.fancycolor:
            return usercolor
        elif usercolor not in self.fancycolor.mapper:
            try:
                fancycolor = self.fancycolor.available.pop(0)
                self.fancycolor.mapper[usercolor] = fancycolor
            except IndexError:
                bench_print("No more fancycolor available")
            return fancycolor
        else:
            return self.fancycolor.mapper[usercolor]

    def add_point(self, gflops, oi, marker=None, color=None, label=None, oi_line=True,
                  point_annotate=None, perf_annotate=None, oi_annotate=None):
        """
        Adds a single point measurement to the roofline plot.

        Parameters
        ----------
        gflops :
            Achieved performance in GFlops/s (y axis value)
        oi :
            Operational intensity in Flops/Byte (x axis value)
        marker :
            Optional plotting marker for point data
        color :
            Optional plotting color for point data
        label :
            Optional legend label for point data
        oi_line :
            Draw a vertical dotted line for the OI value
        point_annotate :
            Optional text to print next to point
        perf_annotate :
            Optional text showing the performance achieved
            relative to the peak
        oi_annotate :
            Optional text or options dict to add an annotation
            to the vertical OI line
        """
        self.xvals += [oi]
        self.yvals += [gflops]

        oi_top = min(oi * self.max_bw, self.max_flops)

        # Add dotted OI line and annotate
        if oi_line:
            self.ax.plot([oi, oi], [1., oi_top], ls=':', lw=0.3, c='black')
            if oi_annotate is not None:
                oi_ann = {'xy': (oi, 0.12), 'size': 8, 'rotation': -90,
                          'xycoords': ('data', 'axes fraction')}
                if isinstance(oi_annotate, Mapping):
                    oi_ann.update(oi_annotate)
                else:
                    oi_ann['s'] = oi_annotate
                plt.annotate(**oi_ann)

        # Add dotted gflops line
        if perf_annotate is not None:
            perf_ann = {'xy': (oi, oi_top), 'size': 5, 'textcoords': 'offset points',
                        'xytext': (-9, 4),
                        's': "%d%%" % (float("%.2f" % (gflops/oi_top))*100)}
            if isinstance(perf_annotate, Mapping):
                perf_ann.update(perf_annotate)
            plt.annotate(**perf_ann)

        # Plot and annotate the data point
        marker = marker or self.marker[0]
        self.ax.plot(oi, gflops, marker=marker, color=self._select_point_color(color),
                     label=label if label not in self.legend_map else None)
        if point_annotate is not None:
            p_ann = {'xy': (oi, gflops), 'size': 8, 'rotation': -45,
                     'xytext': (2, -13), 'textcoords': 'offset points',
                     'bbox': {'facecolor': 'w', 'edgecolor': 'none', 'pad': 1.0}}
            if isinstance(point_annotate, Mapping):
                p_ann.update(point_annotate)
            else:
                p_ann['s'] = point_annotate
            plt.annotate(**p_ann)

        # Record legend labels to avoid replication
        if label is not None:
            self.legend_map[label] = '%s%s' % (marker, color)


ColorTracker = namedtuple('ColorTracker', 'mapper available')
