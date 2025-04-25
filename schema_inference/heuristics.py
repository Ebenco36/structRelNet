# schema_inference/heuristics.py

import re
from dateutil.parser import parse as parse_date

def is_numeric(col):
    try:
        return all(cell == "<EMPTY>" or isinstance(cell, (int, float)) or str(cell).replace(".", "", 1).isdigit() for cell in col)
    except:
        return False

def is_date(col):
    try:
        parsed = [parse_date(cell) for cell in col if cell != "<EMPTY>"]
        return len(parsed) >= len(col) * 0.7
    except:
        return False

def is_categorical(col, threshold=30):
    unique_vals = set(cell for cell in col if cell != "<EMPTY>")
    return 1 < len(unique_vals) <= threshold

def is_text(col, threshold=50):
    return any(len(str(cell)) > threshold for cell in col if cell != "<EMPTY>")
