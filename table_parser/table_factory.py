# table_parser/table_factory.py

import os
from .csv_loader import CSVTableLoader
from .tsv_loader import TSVTableLoader
from .excel_loader import ExcelTableLoader
from .html_loader import HTMLTableLoader
from .table_object import CanonicalTable

class TableLoaderFactory:
    @staticmethod
    def get_loader(path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext == '.csv':
            return CSVTableLoader()
        elif ext == '.tsv':
            return TSVTableLoader()
        elif ext in ['.xls', '.xlsx']:
            return ExcelTableLoader()
        elif ext in ['.html', '.htm']:
            return HTMLTableLoader()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def load(path: str, **kwargs) -> CanonicalTable:
        loader = TableLoaderFactory.get_loader(path)
        return loader.load(path, **kwargs)
