from devito.passes.iet.languages.C import CDataManager, COrchestrator
from devito.passes.iet.languages.openmp import (SimdOmpizer, Ompizer, DeviceOmpizer,
                                                OmpDataManager, DeviceOmpDataManager,
                                                OmpOrchestrator, DeviceOmpOrchestrator)
from devito.passes.iet.languages.openacc import (DeviceAccizer, DeviceAccDataManager,
                                                 AccOrchestrator)

__all__ = ['CTarget', 'OmpTarget', 'DeviceOmpTarget', 'DeviceAccTarget']


class Target(object):
    Parizer = None
    DataManager = None
    Orchestrator = None


class CTarget(Target):
    Parizer = SimdOmpizer
    DataManager = CDataManager
    Orchestrator = COrchestrator


class OmpTarget(Target):
    Parizer = Ompizer
    DataManager = OmpDataManager
    Orchestrator = OmpOrchestrator


class DeviceOmpTarget(Target):
    Parizer = DeviceOmpizer
    DataManager = DeviceOmpDataManager
    Orchestrator = DeviceOmpOrchestrator


class DeviceAccTarget(Target):
    Parizer = DeviceAccizer
    DataManager = DeviceAccDataManager
    Orchestrator = AccOrchestrator
