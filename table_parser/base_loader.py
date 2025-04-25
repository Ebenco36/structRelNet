# table_parser/base_loader.py

from abc import ABC, abstractmethod
from .table_object import CanonicalTable

class BaseTableLoader(ABC):
    @abstractmethod
    def load(self, path: str, **kwargs) -> CanonicalTable:
        pass
