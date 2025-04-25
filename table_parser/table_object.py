from typing import List, Optional

class CanonicalTable:
    def __init__(
        self,
        header: List[str],
        rows: List[List[str]],
        types: Optional[List[str]] = None,
        meta: Optional[dict] = None,
    ):
        self.header = header
        self.rows = rows
        self.types = types or []
        self.meta = meta or {}

    def __repr__(self):
        return f"CanonicalTable(header={self.header}, rows={len(self.rows)} rows, types={self.types})"
