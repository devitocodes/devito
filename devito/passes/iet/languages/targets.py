from devito.passes.iet.languages.C import CDataManager, COrchestrator, CDevitoPrinter
from devito.passes.iet.languages.openmp import (SimdOmpizer, Ompizer, DeviceOmpizer,
                                                OmpDataManager, DeviceOmpDataManager,
                                                OmpOrchestrator, DeviceOmpOrchestrator)
from devito.passes.iet.languages.openacc import (DeviceAccizer, DeviceAccDataManager,
                                                 AccOrchestrator, AccDevitoPrinter)
from devito.passes.iet.instrument import instrument

__all__ = ['CTarget', 'OmpTarget', 'DeviceOmpTarget', 'DeviceAccTarget']


class Target:
    Parizer = None
    DataManager = None
    Orchestrator = None
    Printer = None

    @classmethod
    def lang(cls):
        return cls.Parizer.lang

    @classmethod
    def instrument(cls, *args, **kwargs):
        instrument(*args, lang=cls.lang(), **kwargs)


class CTarget(Target):
    Parizer = SimdOmpizer
    DataManager = CDataManager
    Orchestrator = COrchestrator
    Printer = CDevitoPrinter


class OmpTarget(Target):
    Parizer = Ompizer
    DataManager = OmpDataManager
    Orchestrator = OmpOrchestrator
    Printer = CDevitoPrinter


class DeviceOmpTarget(Target):
    Parizer = DeviceOmpizer
    DataManager = DeviceOmpDataManager
    Orchestrator = DeviceOmpOrchestrator
    Printer = CDevitoPrinter


class DeviceAccTarget(Target):
    Parizer = DeviceAccizer
    DataManager = DeviceAccDataManager
    Orchestrator = AccOrchestrator
    Printer = AccDevitoPrinter
