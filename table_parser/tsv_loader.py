# table_parser/tsv_loader.py

from .csv_loader import CSVTableLoader

class TSVTableLoader(CSVTableLoader):
    def load(self, path: str, **kwargs):
        # Explicitly override delimiter to '\t'
        kwargs["delimiter"] = "\t"
        return super().load(path, **kwargs)
