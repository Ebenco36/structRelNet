## table_parser/table_loader.py
from .table_factory import TableLoaderFactory
from .table_object import CanonicalTable


def load_table(path: str, **kwargs) -> CanonicalTable:
    """
    Unified entry point for loading tables via TableLoaderFactory.
    Supports CSV, TSV, Excel, and HTML formats.
    """
    return TableLoaderFactory.load(path, **kwargs)