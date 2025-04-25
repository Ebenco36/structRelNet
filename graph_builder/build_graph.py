import networkx as nx
from table_parser.table_object import CanonicalTable

class StructRelGraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def _add_node(self, node_id, node_type, **attrs):
        # Use 'node_type' key to avoid shadowing Python's type
        self.graph.add_node(node_id, node_type=node_type, **attrs)

    def _add_edge(self, src, dst, edge_type):
        self.graph.add_edge(src, dst, type=edge_type)

    def build(self, table: CanonicalTable) -> nx.MultiDiGraph:
        self.graph.clear()
        n_rows = len(table.rows)
        n_cols = len(table.header)
        types = getattr(table, 'types', None) or ['unknown'] * n_cols

        # 1) Header and column nodes
        for col_idx, header in enumerate(table.header):
            header_id = f"header_{col_idx}"
            col_id    = f"col_{col_idx}"
            self._add_node(header_id, "header",
                           name=header,
                           col_index=col_idx)
            self._add_node(col_id, "column",
                           index=col_idx,
                           header=header,
                           col_type=types[col_idx])
            self._add_edge(header_id, col_id, "header_for")

        # 2) Row and cell nodes + row/column/header edges
        for row_idx, row in enumerate(table.rows):
            row_id = f"row_{row_idx}"
            self._add_node(row_id, "row", index=row_idx)

            for col_idx, cell_value in enumerate(row):
                cell_id = f"cell_{row_idx}_{col_idx}"
                self._add_node(cell_id, "cell",
                    value=cell_value,
                    row=row_idx,
                    col=col_idx,
                    col_name=table.header[col_idx],
                    col_type=types[col_idx]
                )
                # connect row→cell, column→cell, cell→header
                self._add_edge(row_id, cell_id, "row_contains")
                self._add_edge(f"col_{col_idx}", cell_id, "column_contains")
                self._add_edge(cell_id, f"header_{col_idx}", "cell_to_header")

        # 3) Same‐row relations (bidirectional)
        for row_idx in range(n_rows):
            cell_ids = [f"cell_{row_idx}_{c}" for c in range(n_cols)]
            for src in cell_ids:
                for dst in cell_ids:
                    if src != dst:
                        self._add_edge(src, dst, "same_row")

        # 4) Same‐column relations (bidirectional)
        for col_idx in range(n_cols):
            cell_ids = [f"cell_{r}_{col_idx}" for r in range(n_rows)]
            for src in cell_ids:
                for dst in cell_ids:
                    if src != dst:
                        self._add_edge(src, dst, "same_column")

        return self.graph
