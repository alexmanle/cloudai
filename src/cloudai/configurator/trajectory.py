# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pandas-native trajectory storage with flat, namespaced columns."""

from __future__ import annotations

import csv
import logging
from ast import literal_eval
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cloudai.util.lazy_imports import lazy

if TYPE_CHECKING:
    import pandas as pd


class Trajectory:
    """An ordered DataFrame of DSE steps persisted as core and metadata CSVs."""

    file_name = "trajectory.csv"
    metadata_file_name = "metadata.csv"
    _core_fields = ("step", "action", "reward", "observation")
    _core_domains = frozenset(_core_fields)

    def __init__(
        self,
        *,
        iteration_dir: Path,
        dataframe: pd.DataFrame | None = None,
    ) -> None:
        self._iteration_dir = iteration_dir
        self._dataframe = lazy.pd.DataFrame() if dataframe is None else _copy_dataframe(dataframe)
        self._validate_dataframe()
        self._dataframe = self._dataframe.astype(object)
        logging.debug(
            "Initializing Trajectory: entries=%s, columns=%s.",
            len(self),
            list(self._dataframe.columns),
        )

    def __len__(self) -> int:
        """Return the number of trajectory rows."""
        return len(self._dataframe)

    @property
    def dataframe(self) -> pd.DataFrame:
        """Return a copy of the trajectory DataFrame for analysis."""
        return _copy_dataframe(self._dataframe)

    @property
    def output_path(self) -> Path:
        """Return the trajectory CSV path."""
        return self._iteration_dir / self.file_name

    @property
    def metadata_path(self) -> Path:
        """Return the trajectory metadata CSV path."""
        return self._iteration_dir / self.metadata_file_name

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        """Load a persisted trajectory and its optional metadata CSV."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"trajectory file not found: {path}")

        core = lazy.pd.read_csv(path, dtype=object)
        fields = tuple(core.columns)
        if fields != cls._core_fields:
            raise ValueError(f"trajectory file fields do not match: expected {cls._core_fields}, got {fields}")
        core = core.map(_parse_csv_value)

        dataframe_records: list[dict[str, object]] = []
        for record in core.to_dict(orient="records"):
            dataframe_record = {"step": record["step"]}
            for domain in cls._core_fields[1:]:
                dataframe_record.update(_flatten_value(domain, record[domain]))
            dataframe_records.append(dataframe_record)
        dataframe = lazy.pd.DataFrame(dataframe_records, dtype=object)

        metadata_path = path.with_name(cls.metadata_file_name)
        if metadata_path.is_file():
            metadata = lazy.pd.read_csv(metadata_path, dtype=object).map(_parse_csv_value)
            metadata_fields = tuple(metadata.columns)
            if not metadata_fields or metadata_fields[0] != "step":
                raise ValueError(f"metadata file must begin with a step field: {metadata_fields}")
            invalid_fields = [
                field
                for field in metadata_fields[1:]
                if field.split(".", maxsplit=1)[0] in cls._core_domains
            ]
            if invalid_fields:
                raise ValueError(f"metadata file contains reserved fields: {invalid_fields}")
            dataframe = dataframe.merge(metadata, on="step", how="left", validate="one_to_one")

        return cls(iteration_dir=path.parent, dataframe=dataframe)

    def append(
        self,
        *,
        step: int,
        action: object,
        reward: object,
        observation: object,
        **values: object,
    ) -> pd.Series:
        """Flatten, persist, and store one trajectory row."""
        self._validate_step(step)
        if len(self) and step <= self._dataframe.iloc[-1]["step"]:
            raise ValueError(
                f"trajectory steps must increase: last step is {self._dataframe.iloc[-1]['step']}, got {step}"
            )

        record: dict[str, object] = {"step": step}
        domains = {"action": action, "reward": reward, "observation": observation, **values}
        for domain, value in domains.items():
            for field, field_value in _flatten_value(domain, value).items():
                if field in record:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                record[field] = deepcopy(field_value)

        fields = tuple(record)
        expected_fields = tuple(self._dataframe.columns)
        if expected_fields and fields != expected_fields:
            raise ValueError(f"trajectory record fields changed: expected {expected_fields}, got {fields}")

        row = lazy.pd.Series(record, dtype=object)
        row_trajectory = lazy.pd.Series(
            {
                "step": row["step"],
                "action": action,
                "reward": reward,
                "observation": list(observation.values()) if isinstance(observation, Mapping) else observation,
            },
            dtype=object,
        )
        metadata_fields = tuple(
            field for field in row.index if field == "step" or field.split(".", maxsplit=1)[0] not in self._core_domains
        )
        row_metadata = row[list(metadata_fields)] if len(metadata_fields) > 1 else None

        if row_metadata is not None:
            _validate_csv_header(self.metadata_path, tuple(row_metadata.index))
        _append_csv_row(row_trajectory, self.output_path)
        if row_metadata is not None:
            _append_csv_row(row_metadata, self.metadata_path)

        row_frame = row.to_frame().T.astype(object)
        self._dataframe = lazy.pd.concat([self._dataframe, row_frame], ignore_index=True).astype(object)
        logging.debug("Appended trajectory row for step %s (total rows: %s).", step, len(self))
        return _copy_series(row)

    def find(self, **values: object) -> pd.Series | None:
        """Return a copy of the first row matching all supplied domain values."""
        criteria: dict[str, object] = {}
        for domain, value in values.items():
            for field, field_value in _flatten_value(domain, value).items():
                if field in criteria:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                criteria[field] = field_value

        if any(field not in self._dataframe.columns for field in criteria):
            return None

        for _, row in self._dataframe.iterrows():
            if all(_values_match_exact(row[field], value) for field, value in criteria.items()):
                logging.debug("Found matching trajectory row at step %s for %s.", row["step"], values)
                return _copy_series(row)
        logging.debug("No matching trajectory row found for %s.", values)
        return None

    def _validate_dataframe(self) -> None:
        if not self._dataframe.columns.is_unique:
            raise ValueError("trajectory dataframe columns must be unique")
        non_string_columns = [column for column in self._dataframe.columns if not isinstance(column, str)]
        if non_string_columns:
            raise TypeError(f"trajectory dataframe columns must be strings: {non_string_columns}")
        if self._dataframe.empty and len(self._dataframe.columns) == 0:
            return
        if "step" not in self._dataframe.columns:
            raise ValueError("trajectory dataframe must contain a step column")

        previous_step: int | None = None
        for dataframe_step in self._dataframe["step"]:
            step = dataframe_step.item() if hasattr(dataframe_step, "item") else dataframe_step
            self._validate_step(step)
            if previous_step is not None and step <= previous_step:
                raise ValueError(f"trajectory steps must increase: last step is {previous_step}, got {step}")
            previous_step = step

    @staticmethod
    def _validate_step(step: object) -> None:
        if type(step) is not int or step < 1:
            raise ValueError(f"trajectory step must be a positive integer; got {step}")


def _append_csv_row(row: pd.Series, path: Path) -> None:
    """Validate and append one Series to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _validate_csv_header(path, tuple(row.index))
    row.to_frame().T.to_csv(path, mode="a", header=header, index=False)
    logging.debug("Wrote trajectory row to %s.", path)


def _validate_csv_header(path: Path, fields: tuple[str, ...]) -> bool:
    """Validate an existing CSV header and return whether a header must be written."""
    write_header = not path.exists() or path.stat().st_size == 0
    if write_header:
        return True
    with path.open(newline="") as file:
        existing_fields = tuple(next(csv.reader(file), ()))
    if existing_fields != fields:
        raise ValueError(f"trajectory file fields do not match: expected {fields}, got {existing_fields}")
    return False


def _parse_csv_value(value: object) -> object:
    """Parse Python-literal CSV values while preserving ordinary strings."""
    if not isinstance(value, str):
        return value
    try:
        return literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def _flatten_value(key: str, value: object) -> dict[str, object]:
    """Flatten a value into dot-separated columns."""
    if isinstance(value, Mapping):
        record: dict[str, object] = {}
        for child_key, child_value in value.items():
            if not isinstance(child_key, str):
                raise TypeError(f"trajectory mapping keys must be strings: {child_key}")
            for field, field_value in _flatten_value(f"{key}.{child_key}", child_value).items():
                if field in record:
                    raise ValueError(f"trajectory values produce duplicate column: {field}")
                record[field] = field_value
        return record
    return {key: value}


def _copy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Copy a DataFrame and recursively snapshot object cells."""
    copied = dataframe.copy(deep=True).astype(object)
    for row_index in range(dataframe.shape[0]):
        for column_index in range(dataframe.shape[1]):
            copied.iat[row_index, column_index] = deepcopy(dataframe.iat[row_index, column_index])
    return copied


def _copy_series(series: pd.Series) -> pd.Series:
    """Copy a Series and recursively snapshot object cells."""
    copied = series.copy(deep=True).astype(object)
    for index in range(len(series)):
        copied.iat[index] = deepcopy(series.iat[index])
    return copied


def _values_match_exact(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, Mapping):
        if set(left) != set(right):
            return False
        return all(_values_match_exact(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        return len(left) == len(right) and all(
            _values_match_exact(left_item, right_item) for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)
