from collections import defaultdict
from devito.logger import info
import numpy as np


__all__ = ['Executor']


class Executor(object):

    """
    Abstract container class for a single benchmark data point.
    """

    def setup(self, **kwargs):
        """
        Prepares a single benchmark invocation.
        """
        pass

    def teardown(self, **kwargs):
        """
        Cleans up a single benchmark invocation.
        """
        pass

    def postprocess(self, **kwargs):
        """
        Global post-processing method to collect meta-data.
        """
        pass

    def reset(self):
        """
        Reset the data dictionaries.
        """
        self.meta = {}
        self.timings = defaultdict(lambda: defaultdict(float))

    def run(self, **kwargs):
        """
        This method needs to be overridden by the user.
        """
        raise NotImplementedError("No custom executor function specified")
    
    def unify_mpi_measures(self, gflopss, oi, timings):
        """
        Unify the measures of all MPI ranks,
        reducing the values of all ranks to rank0 only
        """
        keys = list(timings.keys())
        
        gflopss_temp = {}
        oi_temp = {}
        timings_temp = {}
        
        for key in keys:
            # MPI disabled
            if key.rank == None:
                return gflopss, oi, timings
            
            # MPI enabled
            if key.rank == 0:
                gflopss_temp[(key.name, key.rank)] = gflopss[key]
                oi_temp[(key.name, key.rank)] = oi[key]
                timings_temp[(key.name, key.rank)] = timings[key]
            else:
                # Global gflops must the sum of gflops of all ranks
                gflopss_temp[(key.name, 0)] += gflopss[key]
                del gflopss[key]
                
                # Global oi must be the maximum oi of all ranks
                oi_temp[(key.name, 0)] = np.maximum(oi_temp[(key.name, 0)], oi[key])
                del oi[key]
                
                # Global timing must be the maximum timig of all ranks
                timings_temp[(key.name, 0)] = np.maximum(timings_temp[(key.name, 0)], timings[key])
                del timings[key]
        
        # Update values in rank0
        for key in timings.keys():
            gflopss[key] = gflopss_temp[(key.name, 0)]
            oi[key] = oi_temp[(key.name, 0)]
            timings[key] = timings_temp[(key.name, 0)]
        
        return gflopss, oi, timings
    
    def register(self, value, event='execute', measure='time'):
        """
        Register a single timing value for a given event key.

        Parameters
        ----------
        event : str
            key for the measured event, ie. 'assembly' or 'solve'
        value : float
            measured value to store
        measure : str
            name of the value type, eg. 'time' or 'flops'
        """
        self.timings[event][measure] += value

    def execute(self, warmups=1, repeats=3, **params):
        """
        Execute a single benchmark repeatedly, including
        setup, teardown and postprocessing methods.
        """
        info("Running %d repeats - parameters: %s" % (repeats,
             ', '.join(['%s: %s' % (k, v) for k, v in params.items()])))

        self.reset()
        for i in range(warmups):
            info("--- Warmup %d ---" % i)
            self.setup(**params)
            self.run(**params)
            self.teardown(**params)
            info("--- Warmup %d finished ---" % i)

        self.reset()
        for i in range(repeats):
            info("--- Run %d ---" % i)
            self.setup(**params)
            self.run(**params)
            self.teardown(**params)
            info("--- Run %d finished ---" % i)

        info("")

        # Average timings across repeats
        for event in self.timings.keys():
            for measure in self.timings[event].keys():
                self.timings[event][measure] /= repeats

        # Collect meta-information via post-processing methods
        self.postprocess(**params)
