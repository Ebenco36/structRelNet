# schema_inference/infer_schema.py

from schema_inference.base_inferer import BaseSchemaInferer
from table_parser.table_object import CanonicalTable
from schema_inference.heuristics import is_numeric, is_categorical, is_date, is_text

class SchemaInferer(BaseSchemaInferer):
    def infer(self, table: CanonicalTable) -> CanonicalTable:
        types = []

        for col_idx in range(len(table.header)):
            col = [row[col_idx] for row in table.rows]

            if is_numeric(col):
                col_type = "numeric"
            elif is_date(col):
                col_type = "date"
            elif is_text(col):
                col_type = "text"
            elif is_categorical(col):
                col_type = "categorical"
            else:
                col_type = "unknown"

            types.append(col_type)

        table.types = types
        return table
