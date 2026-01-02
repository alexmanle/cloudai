# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy


class NeMoRLReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMo RL test outputs."""

    metrics: ClassVar[list[str]] = ["default"]

    def can_handle_directory(self) -> bool:
        """
        Determine if this strategy can process the test run's output directory.

        Returns:
            bool: True if directory can be handled, False otherwise.
        """
        output_file = self.test_run.output_path / "metrics.json"

        if not output_file.exists():
            return False

        try:
            with output_file.open("r") as file:
                data = json.load(file)
                # If we can parse the JSON and it's not empty, we can handle it
                return bool(data)
        except (json.JSONDecodeError, Exception):
            return False

    def generate_report(self) -> None:
        """
        Parse test output and generate report files with metrics.

        This method:
        - Locates and validates the metrics.json file
        - Extracts relevant metrics
        - Writes report file to output directory
        """
        results_file = self.test_run.output_path / "metrics.json"
        if not results_file.exists():
            logging.error(f"{results_file} not found")
            return

        try:
            with results_file.open("r") as file:
                metrics_data = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse {results_file}: {e}")
            return
        except Exception as e:
            logging.error(f"Error reading {results_file}: {e}")
            return

        if not metrics_data:
            logging.error(f"No valid metrics found in {results_file}. Report generation aborted.")
            return

        # Write report to output directory
        report_file = self.test_run.output_path / "report.txt"
        with open(report_file, "w") as f:
            f.write("NeMo RL Test Report\n")
            f.write("=" * 50 + "\n\n")
            f.write("Metrics:\n")
            f.write("-" * 50 + "\n")
            for key, value in metrics_data.items():
                f.write(f"{key}: {value}\n")

        logging.info(f"Report generated successfully at {report_file}")

    def get_metric(self, metric: str) -> float:
        """
        Return specific metric value by name.

        Args:
            metric: Name of the metric to retrieve.

        Returns:
            float: Metric value, or METRIC_ERROR if not found.
        """
        results_file = self.test_run.output_path / "metrics.json"
        logging.debug(f"Getting metric {metric} from {results_file.absolute()}")

        if not results_file.exists():
            return METRIC_ERROR

        try:
            with results_file.open("r") as file:
                metrics_data = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error reading metrics from {results_file}: {e}")
            return METRIC_ERROR

        if not metrics_data:
            return METRIC_ERROR

        if metric == "default":
            # Return the first metric value found, or METRIC_ERROR if none
            for value in metrics_data.values():
                try:
                    return float(value)
                except (ValueError, TypeError):
                    continue
            return METRIC_ERROR

        # Try to get the specific metric
        if metric in metrics_data:
            try:
                return float(metrics_data[metric])
            except (ValueError, TypeError):
                return METRIC_ERROR

        return METRIC_ERROR
