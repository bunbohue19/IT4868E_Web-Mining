from abc import ABC, abstractmethod

class RecordType(ABC):
    @abstractmethod
    def metadata_func(self):
        pass
    @abstractmethod
    def init_loader(self):
        pass
