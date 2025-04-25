# table_parser/preprocess.py

def clean_table(header, rows, missing_token="<EMPTY>"):
    cleaned_header = [h.strip() for h in header]
    cleaned_rows = [
        [cell.strip() if isinstance(cell, str) and cell.strip() else missing_token for cell in row]
        for row in rows
    ]
    return cleaned_header, cleaned_rows
