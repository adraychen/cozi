"""Explicit, atomic conversion of currency text to nullable numeric values."""
from collections import Counter
from decimal import Decimal, InvalidOperation
import math
import numbers
import re

import pandas as pd
from utils.table_transforms import validate_table


class CurrencyConversionError(ValueError):
    def __init__(self, invalid):
        self.invalid = invalid.copy() if isinstance(invalid, pd.DataFrame) else pd.DataFrame([{'Value': value, 'Count': count} for value, count in invalid.items()])
        super().__init__(f"Conversion incomplete — {self.invalid['Count'].sum()} values cannot be converted to numeric ({len(self.invalid)} distinct entries).")


def _parse_currency(value):
    if pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        accounting = re.fullmatch(r'([$£¥€]?)\s*\(([^()]*)\)', text)
        if accounting:
            # Accept $(3000), ($3000), and (3000), but reject conflicting signs.
            text = accounting[1] + accounting[2].strip()
        if re.fullmatch(r'[$£¥€]\s*-', text):
            if accounting:
                raise ValueError('Invalid parenthesized currency value')
            return Decimal(0)
        # One optional currency symbol, with a sign before or after it.
        match = re.fullmatch(r'([+-]?)\s*([$£¥€]?)\s*([+-]?)\s*((?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+|[0-9]{1,2}(?:,[0-9]{2})+,[0-9]{3})(?:\.[0-9]+)?|\.[0-9]+)', text)
        if not match or (match[1] and match[3]):
            raise ValueError('Invalid currency value')
        if accounting:
            if match[1] or match[3]:
                raise ValueError('Parentheses cannot be combined with a sign')
            return Decimal('-' + match[4].replace(',', ''))
        return Decimal((match[1] or match[3]) + match[4].replace(',', ''))
    if isinstance(value, (numbers.Real, Decimal)) and not isinstance(value, bool):
        parsed = Decimal(str(value))
        if parsed.is_finite():
            return parsed
    raise ValueError('Not numeric text or a finite number')


def detect_currency_columns(frame):
    """Suggest columns with at least one valid, explicitly symbol-marked amount."""
    detected = []
    for column in frame:
        for value in frame[column]:
            if not isinstance(value, str) or not re.search(r'[$£¥€]', value):
                continue
            try:
                _parse_currency(value)
            except (ValueError, TypeError, InvalidOperation, OverflowError):
                continue
            detected.append(column)
            break
    return detected


def convert_currency_to_numeric(frame, column):
    if column not in frame:
        raise ValueError('Choose an existing column.')
    values, invalid = [], Counter()
    for value in frame[column]:
        try:
            parsed = _parse_currency(value)
            if parsed is not None and (not math.isfinite(float(parsed)) or (parsed != 0 and float(parsed) == 0)):
                raise ValueError('Value outside supported numeric range')
            values.append(parsed)
        except (ValueError, TypeError, InvalidOperation, OverflowError):
            invalid[repr(value)] += 1
            values.append(None)
    if invalid:
        raise CurrencyConversionError(invalid)
    whole = all(value is None or value == value.to_integral_value() for value in values)
    if whole:
        for original, value in zip(frame[column], values):
            if value is not None and not -(2**63) <= value < 2**63:
                invalid[repr(original)] += 1
        if invalid:
            raise CurrencyConversionError(invalid)
    converted = pd.array([pd.NA if value is None else int(value) if whole else float(value) for value in values], dtype='Int64' if whole else 'Float64')
    result = frame.copy()
    result[column] = converted
    return validate_table(result)
