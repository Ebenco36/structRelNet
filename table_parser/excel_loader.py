# table_parser/excel_loader.py

import pandas as pd
from .base_loader import BaseTableLoader
from .table_object import CanonicalTable
from .preprocess import clean_table

class ExcelTableLoader(BaseTableLoader):
    def load(self, path: str, sheet_name=0, has_header=True, **kwargs) -> CanonicalTable:
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=str)
        df.fillna("<EMPTY>", inplace=True)

        header = df.columns.tolist() if has_header else [f"col_{i}" for i in range(len(df.columns))]
        rows = df.values.tolist()
        cleaned_header, cleaned_rows = clean_table(header, rows)

        return CanonicalTable(header=cleaned_header, rows=cleaned_rows)
