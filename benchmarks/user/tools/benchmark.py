from collections import OrderedDict
from collections.abc import Iterable
from itertools import product
from datetime import datetime
from os import path, makedirs
import json

from devito.logger import warning


__all__ = ['Benchmark']


class Benchmark(object):
    """
    Performance data for a set of benchmark runs indexed by a multi-parameter key.

    Parameters
    ----------
    parameters : dict
        Dict of parameter names and value ranges that define the parameter
        space for this benchmark.
    resultsdir : str, optional
        Define directory name to store results in. Defaults to 'results'.
    name : str, optional
        Set name of the benchmark instance. Defaults to 'Benchmark'.
    """

    def __init__(self, parameters, resultsdir='results', name='Benchmark'):
        self.name = name
        self.resultsdir = resultsdir

        self._params = parameters.keys()
        self._values = parameters

        self.timings = {}
        self.meta = {}

    @property
    def params(self):
        """
        Lexicographically sorted parameter key.
        """
        return tuple(sorted(self._params))

    @property
    def loaded(self):
        return self.timings and self.meta

    def values(self, keys=None):
        """
        Sorted dict of parameter-value mappings for all parameters.

        Parameters
        ----------
        keys : Dict or dict-like, optional
            Key-value dict to generate a subset of values.
        """
        # Ensure all values are lists
        values = [(k, [v]) if not isinstance(v, list) else (k, v) for
                  k, v in self._values.items()]
        if keys is not None:
            # Ensure all keys are lists
            keys = dict([(k, [v]) if not isinstance(v, list) else (k, v)
                         for k, v in keys.items()])
            values = [(k, keys[k]) if k in keys else (k, v) for k, v in values]
        valdict = OrderedDict(sorted(values))
        assert(len(valdict) == len(self.params))
        return valdict

    def sweep(self, keys=None):
        """
        List of value mappings for each instance of a parameter sweep.

        Parameters
        ----------
        keys : dict or dict-like, optional
            Dict with parameter value mappings over which to sweep.
        """
        values = self.values(keys=keys)
        return [OrderedDict(zip(self.params, v)) for v in product(*values.values())]

    def param_string(self, params):
        """
        Convert parameter tuple to string.
        """
        return '_'.join(['%s%s' % p for p in params])

    def lookup(self, params={}, event=None, measure='time', category='timings'):
        """
        Lookup a set of results according to a parameter set.

        Parameters
        ----------
        params : dict
            Parameter set by which to filter results.
        event : str or iterable of str, optional
            One or more events for which to retrieve data.
        measure : str, optional
            The results measure, for example 'time'.
        category : str, optional
            Either 'timings' or 'meta'.
        """
        assert(category in ['timings', 'meta'])
        result = OrderedDict()
        for params in self.sweep(params):
            key = tuple(params.items())
            datadict = getattr(self, category)
            if key in datadict:
                if event is None:
                    result[key] = datadict[key][measure]
                elif isinstance(event, Iterable) and not isinstance(event, str):
                    result[key] = [datadict[key][ev][measure]
                                   for ev in event]
                else:
                    result[key] = datadict[key][event][measure]
        return result

    def execute(self, executor, warmups=1, repeats=3):
        """
        Main execution function that invokes the given executor for each
        combination of the parameter sweep.
        """
        for params in self.sweep():
            # Execute the benchmark
            executor.execute(warmups=warmups, repeats=repeats, **params)

            # Store timing and meta data under the parameter key
            self.timings[tuple(params.items())] = executor.timings
            self.meta[tuple(params.items())] = executor.meta

    def save(self):
        """
        Save all timing results in individually keyed files.
        """
        if not path.exists(self.resultsdir):
            makedirs(self.resultsdir)
        timestamp = datetime.now().strftime('%Y-%m-%dT%H%M%S')

        for key in self.timings.keys():
            datadict = OrderedDict()
            datadict['timestamp'] = timestamp
            datadict['meta'] = self.meta[key]
            datadict['timings'] = self.timings[key]

            filename = '%s_%s.json' % (self.name, self.param_string(key))
            with open(path.join(self.resultsdir, filename), 'w') as f:
                json.dump(datadict, f, indent=4)

    def load(self):
        """
        Load timing results from individually keyed files.
        """
        for params in self.sweep():
            filename = '%s_%s.json' % (self.name, self.param_string(params.items()))
            try:
                with open(path.join(self.resultsdir, filename), 'r') as f:
                    datadict = json.loads(f.read())
                    self.timings[tuple(params.items())] = datadict['timings']
                    self.meta[tuple(params.items())] = datadict['meta']
            except:
                warning("Could not load file: %s" % filename)
