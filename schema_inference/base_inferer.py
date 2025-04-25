# schema_inference/base_inferer.py

from abc import ABC, abstractmethod
from table_parser.table_object import CanonicalTable

class BaseSchemaInferer(ABC):
    @abstractmethod
    def infer(self, table: CanonicalTable) -> CanonicalTable:
        pass
