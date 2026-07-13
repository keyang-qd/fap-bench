from datetime import date, datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

from fapbench.utils import write_json


def test_write_json_supports_yaml_dates_and_scientific_types(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    payload = {
        "date": date(2014, 12, 31),
        "datetime": datetime(2018, 12, 31, 12, 30),
        "timestamp": pd.Timestamp("2020-01-31"),
        "period": pd.Period("2020-01", freq="M"),
        "np_date": np.datetime64("2021-02-28"),
        "np_bool": np.bool_(True),
        "np_int": np.int64(7),
        "path": tmp_path,
    }
    write_json(payload, path)
    restored = json.loads(path.read_text())
    assert restored["date"] == "2014-12-31"
    assert restored["datetime"] == "2018-12-31T12:30:00"
    assert restored["np_bool"] is True
    assert restored["np_int"] == 7


def test_write_json_replaces_existing_file_atomically(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_text('{"old": true}\n')
    write_json({"new": date(2026, 7, 13)}, path)
    assert json.loads(path.read_text()) == {"new": "2026-07-13"}
    assert not list(tmp_path.glob(".*.tmp-*"))
