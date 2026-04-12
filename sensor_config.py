"""
sensor_config.py
----------------
Shared utility for reading/writing the sensor packet format config.
Lives at the project root so both app/main.py and hardware/data_logger.py
can import it via sys.path.

Usage:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    import sensor_config as sc

    fields   = sc.load_config()          # OrderedDict of field -> bool
    active   = sc.get_active_fields()    # ['flex_1', 'flex_2', ...]
    n        = sc.get_expected_len()     # 5
    parsed   = sc.parse_packet("0.71,1.56,0.00,0.04,1.56")
    # -> {'flex_1': 0.71, 'flex_2': 1.56, ...} or None on bad packet
"""

import json
import os
from collections import OrderedDict

# Always resolve relative to this file so scripts in subdirs work correctly
_HERE        = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH  = os.path.join(_HERE, "sensor_config.json")

# ── Field metadata ────────────────────────────────────────────────────────────
# Defines canonical order, display label, group, and validation range.
FIELD_META = OrderedDict([
    ("timestamp", {"label": "Timestamp", "group": "meta",
                   "min": None,     "max": None}),
    ("flex_1",    {"label": "Flex 1",    "group": "flex",
                   "min": 0.0,      "max": 100.0}),
    ("flex_2",    {"label": "Flex 2",    "group": "flex",
                   "min": 0.0,      "max": 100.0}),
    ("flex_3",    {"label": "Flex 3",    "group": "flex",
                   "min": 0.0,      "max": 100.0}),
    ("flex_4",    {"label": "Flex 4",    "group": "flex",
                   "min": 0.0,      "max": 100.0}),
    ("flex_5",    {"label": "Flex 5",    "group": "flex",
                   "min": 0.0,      "max": 100.0}),
    ("accel_x",   {"label": "Accel X",   "group": "accel",
                   "min": -20000.0, "max": 20000.0}),
    ("accel_y",   {"label": "Accel Y",   "group": "accel",
                   "min": -20000.0, "max": 20000.0}),
    ("accel_z",   {"label": "Accel Z",   "group": "accel",
                   "min": -20000.0, "max": 20000.0}),
    ("gyro_x",    {"label": "Gyro X",    "group": "gyro",
                   "min": -20000.0, "max": 20000.0}),
    ("gyro_y",    {"label": "Gyro Y",    "group": "gyro",
                   "min": -20000.0, "max": 20000.0}),
    ("gyro_z",    {"label": "Gyro Z",    "group": "gyro",
                   "min": -20000.0, "max": 20000.0}),
])

# Group colors used by the UI diagram
GROUP_COLORS = {
    "meta":  "#9E9E9E",   # gray
    "flex":  "#1E3A8A",   # blue  (PRIMARY_COLOR)
    "accel": "#2E7D32",   # green
    "gyro":  "#E65100",   # orange
}

# ── I/O ───────────────────────────────────────────────────────────────────────

def load_config() -> dict:
    """Return {field: bool} for all fields. Creates default file if missing."""
    if not os.path.exists(CONFIG_PATH):
        _write_default()

    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)

    # Fill in any missing keys with False so new fields are opt-in
    cfg = {}
    for field in FIELD_META:
        cfg[field] = data.get("fields", {}).get(field, False)
    return cfg


def save_config(fields: dict) -> None:
    """Write {field: bool} dict back to disk."""
    with open(CONFIG_PATH, "w") as f:
        json.dump({"fields": fields}, f, indent=4)


def _write_default() -> None:
    defaults = {f: (f.startswith("flex_")) for f in FIELD_META}
    save_config(defaults)


# ── Derived helpers ───────────────────────────────────────────────────────────

def get_active_fields() -> list:
    """Return ordered list of enabled field names, e.g. ['flex_1',...,'gyro_z']."""
    cfg = load_config()
    return [f for f in FIELD_META if cfg.get(f, False)]


def get_active_sensor_fields() -> list:
    """Like get_active_fields() but always excludes 'timestamp'.
    This is the list that becomes CSV columns and ML features.
    """
    return [f for f in get_active_fields() if f != "timestamp"]


def get_expected_len() -> int:
    """Total number of comma-separated values expected in one serial packet."""
    return len(get_active_fields())


def get_csv_header(include_label: bool = True) -> list:
    """CSV column names for data_logger output."""
    cols = get_active_sensor_fields()
    if include_label:
        cols = cols + ["label"]
    return cols


# ── Packet parsing ────────────────────────────────────────────────────────────

def parse_packet(line: str) -> dict | None:
    """
    Parse one raw serial line according to the current config.

    Returns an OrderedDict {field_name: float} for sensor fields only
    (timestamp is stored separately if present).
    Returns None if the packet length doesn't match or a value can't be parsed.

    Special keys in return dict:
        '__timestamp__' : raw timestamp string if timestamp field is active
    """
    active = get_active_fields()
    expected = len(active)

    raw = line.strip()
    if raw.startswith("{") and raw.endswith("}"):
        raw = raw[1:-1]

    parts = raw.split(",")
    if len(parts) != expected:
        return None

    result = OrderedDict()
    for field, raw_val in zip(active, parts):
        raw_val = raw_val.strip()
        if field == "timestamp":
            result["__timestamp__"] = raw_val
            continue
        try:
            val = float(raw_val)
        except ValueError:
            return None

        # Range validation (warn but don't reject — caller decides)
        meta = FIELD_META[field]
        if meta["min"] is not None and not (meta["min"] <= val <= meta["max"]):
            pass  # caller can check; we still return the value

        result[field] = val

    return result


def validate_packet(parsed: dict) -> tuple[bool, list]:
    """
    Check all sensor values in a parsed packet against their allowed ranges.
    Returns (all_valid: bool, list_of_warning_strings).
    """
    warnings = []
    for field, val in parsed.items():
        if field == "__timestamp__":
            continue
        meta = FIELD_META.get(field, {})
        lo, hi = meta.get("min"), meta.get("max")
        if lo is not None and not (lo <= val <= hi):
            warnings.append(
                f"{field} value {val:.2f} out of range [{lo}, {hi}]"
            )
    return (len(warnings) == 0), warnings
