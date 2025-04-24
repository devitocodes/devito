from devito.passes.iet.languages.C import CDataManager, COrchestrator, CPrinter
from devito.passes.iet.languages.CXX import CXXDataManager, CXXOrchestrator, CXXPrinter
from devito.passes.iet.languages.openmp import (SimdOmpizer, Ompizer, DeviceOmpizer,
                                                OmpDataManager, DeviceOmpDataManager,
                                                OmpOrchestrator, DeviceOmpOrchestrator,
                                                CXXSimdOmpizer, CXXOmpizer,
                                                CXXOmpDataManager, CXXOmpOrchestrator)
from devito.passes.iet.languages.openacc import (DeviceAccizer, DeviceAccDataManager,
                                                 AccOrchestrator, AccPrinter)
from devito.passes.iet.instrument import instrument

__all__ = ['CTarget', 'OmpTarget', 'COmpTarget', 'DeviceOmpTarget', 'DeviceAccTarget',
           'CXXTarget', 'CXXOmpTarget', 'DeviceCXXOmpTarget']


class Target:
    Parizer = None
    DataManager = None
    Orchestrator = None
    Printer = None

    @classmethod
    def langbb(cls):
        return cls.Parizer.langbb

    @classmethod
    def instrument(cls, *args, **kwargs):
        instrument(*args, **kwargs)


class CTarget(Target):
    Parizer = SimdOmpizer
    DataManager = CDataManager
    Orchestrator = COrchestrator
    Printer = CPrinter


class CXXTarget(Target):
    Parizer = CXXSimdOmpizer
    DataManager = CXXDataManager
    Orchestrator = CXXOrchestrator
    Printer = CXXPrinter


class COmpTarget(Target):
    Parizer = Ompizer
    DataManager = OmpDataManager
    Orchestrator = OmpOrchestrator
    Printer = CPrinter


OmpTarget = COmpTarget


class CXXOmpTarget(Target):
    Parizer = CXXOmpizer
    DataManager = CXXOmpDataManager
    Orchestrator = CXXOmpOrchestrator
    Printer = CXXPrinter


class DeviceCOmpTarget(Target):
    Parizer = DeviceOmpizer
    DataManager = DeviceOmpDataManager
    Orchestrator = DeviceOmpOrchestrator
    Printer = CPrinter


DeviceOmpTarget = DeviceCOmpTarget


class DeviceCXXOmpTarget(Target):
    Parizer = DeviceOmpizer
    DataManager = DeviceOmpDataManager
    Orchestrator = DeviceOmpOrchestrator
    Printer = CXXPrinter


class DeviceAccTarget(Target):
    Parizer = DeviceAccizer
    DataManager = DeviceAccDataManager
    Orchestrator = AccOrchestrator
    Printer = AccPrinter
