# table_parser/html_loader.py

import pandas as pd
from .base_loader import BaseTableLoader
from .table_object import CanonicalTable
from .preprocess import clean_table

class HTMLTableLoader(BaseTableLoader):
    def load(self, path: str, table_index=0, **kwargs) -> CanonicalTable:
        tables = pd.read_html(path, encoding='utf-8')
        df = tables[table_index]
        df.fillna("<EMPTY>", inplace=True)

        header = df.columns.tolist()
        rows = df.values.tolist()
        cleaned_header, cleaned_rows = clean_table(header, rows)

        return CanonicalTable(header=cleaned_header, rows=cleaned_rows)
