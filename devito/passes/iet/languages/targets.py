from devito.passes.iet.instrument import instrument
from devito.passes.iet.languages.C import CDataManager, COrchestrator, CPrinter
from devito.passes.iet.languages.CXX import CXXDataManager, CXXOrchestrator, CXXPrinter
from devito.passes.iet.languages.openacc import (
    AccOrchestrator, AccPrinter, DeviceAccDataManager, DeviceAccizer
)
from devito.passes.iet.languages.openmp import (
    CXXOmpDataManager, CXXOmpizer, CXXOmpOrchestrator, CXXSimdOmpizer,
    DeviceOmpDataManager, DeviceOmpizer, DeviceOmpOrchestrator, OmpDataManager, Ompizer,
    OmpOrchestrator, SimdOmpizer
)

__all__ = [
    'COmpTarget',
    'CTarget',
    'CXXOmpTarget',
    'CXXTarget',
    'DeviceAccTarget',
    'DeviceCXXOmpTarget',
    'DeviceOmpTarget',
    'OmpTarget',
]


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
