# table_parser/csv_loader.py

import csv
from .base_loader import BaseTableLoader
from .table_object import CanonicalTable
from .preprocess import clean_table

class CSVTableLoader(BaseTableLoader):
    def load(self, path: str, delimiter=',', has_header=True, encoding='utf-8-sig', **kwargs) -> CanonicalTable:
        with open(path, 'r', encoding=encoding, newline='') as f:
            reader = csv.reader(f, delimiter=delimiter)
            data = [row for row in reader if any(cell.strip() for cell in row)]

        if not data:
            raise ValueError("File appears empty or invalid.")

        header, rows = (data[0], data[1:]) if has_header else ([f"col_{i}" for i in range(len(data[0]))], data)
        cleaned_header, cleaned_rows = clean_table(header, rows)

        return CanonicalTable(header=cleaned_header, rows=cleaned_rows)
