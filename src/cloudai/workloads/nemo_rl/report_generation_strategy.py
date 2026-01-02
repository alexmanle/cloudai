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
import re
import shutil
from pathlib import Path
from typing import ClassVar, Optional

import pandas as pd

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy


class NeMoRLReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from NeMo RL test outputs."""

    # Metric definitions: metric_name -> (display_name, json_path)
    METRICS: ClassVar[dict[str, tuple[str, str]]] = {
        "e2e": ("E2E", "performance/tokens_per_sec_per_gpu"),
        "generation": ("Generation", "performance/generation_tokens_per_sec_per_gpu"),
        "policy": ("Policy", "performance/training_worker_group_tokens_per_sec_per_gpu"),
        "training": ("Training", "performance/policy_training_tokens_per_sec_per_gpu"),
        "logprob": ("LogProb", "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu"),
    }

    # Derive metrics list from METRICS dict, adding "default" which returns E2E metric
    metrics: ClassVar[list[str]] = ["default"] + list(METRICS.keys())
    
    @property
    def results_file(self) -> Path:
        return self.test_run.output_path / "metrics.json"

    @property
    def launcher_log(self) -> Path:
        return self.test_run.output_path / "logs" / "nemo_rl_launcher.log"

    @property
    def check_metrics_log(self) -> Path:
        return self.test_run.output_path / "logs" / "metrics_checks.log"

    def can_handle_directory(self) -> bool:
        """
        Determine if this strategy can process the test run's output directory.

        Returns:
            bool: True if directory can be handled, False otherwise.
        """

        # Clean up clutter log files regardless of whether metrics.json exists
        logging.debug(f"Cleaning up clutter log files for {self.test_run.output_path}")
        try:
            self.cleanup_log_files()
        except Exception as e:
            logging.error(f"Error cleaning up clutter log files for {self.test_run.output_path}: {e}")

        if not self.results_file.exists():
            return False

        try:
            with self.results_file.open("r") as file:
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
        - Extracts relevant performance metrics using get_metric()
        - Computes statistics (mean, min, max, stddev) for each metric
        - Parses check_metrics.log for pass/fail status
        - Writes report file to output directory with formatted metrics
        """
        if not self.results_file.exists():
            logging.error(f"{self.results_file} not found")
            return

        # Load metrics data
        try:
            with self.results_file.open("r") as file:
                metrics_data = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error reading metrics from {self.results_file}: {e}")
            return

        # Write report to output directory
        report_file = self.test_run.output_path / "report.txt"
        with open(report_file, "w") as f:
            f.write("NeMo RL Test Report\n")  
            f.write("=" * 90 + "\n\n")
            
            # Performance Metrics Section
            f.write("Performance Metrics (Tokens/sec/gpu):\n")
            f.write("-" * 90 + "\n")
            
            # Build DataFrame for metrics statistics
            metrics_stats = []
            for metric_key, (display_name, json_key) in self.METRICS.items():
                if json_key in metrics_data:
                    values_dict = metrics_data[json_key]
                    if isinstance(values_dict, dict):
                        try:
                            values = [float(v) for k, v in values_dict.items()]
                            if values:
                                stats = {
                                    "Metric": display_name,
                                    "Mean": pd.Series(values).mean(),
                                    "Min": pd.Series(values).min(),
                                    "Max": pd.Series(values).max(),
                                    "StdDev": pd.Series(values).std(),
                                }
                                metrics_stats.append(stats)
                        except (ValueError, TypeError) as e:
                            logging.debug(f"Could not compute stats for {metric_key}: {e}")
            
            if metrics_stats:
                df = pd.DataFrame(metrics_stats)
                f.write(f"{'Metric':<15} {'Mean':>12} {'Min':>12} {'Max':>12} {'StdDev':>12}\n")
                f.write("-" * 90 + "\n")
                for _, row in df.iterrows():
                    f.write(f"{row['Metric']:<15} {row['Mean']:>12.2f} {row['Min']:>12.2f} "
                           f"{row['Max']:>12.2f} {row['StdDev']:>12.2f}\n")
            else:
                f.write("No performance metrics available\n")
            
            f.write("\nSee logs/metrics_checks.log for detailed metric checks.")

        logging.info(f"Report generated successfully at {report_file}")

    def _extract_job_id(self) -> Optional[str]:
        """
        Extract the JOB_ID from the nemo_rl_launcher.log file.

        Returns:
            Optional[str]: The extracted JOB_ID, or None if not found.
        """
        
        if not self.launcher_log.exists():
            logging.warning(f"Launcher log not found at {self.launcher_log}")
            return None

        try:
            with self.launcher_log.open("r") as file:
                content = file.read()
                # Look for pattern: "Submitted batch job <job_id>"
                match = re.search(r"Submitted batch job (\d+)", content)
                if match:
                    return match.group(1)
        except Exception as e:
            logging.error(f"Error reading launcher log {self.launcher_log}: {e}")

        return None

    def cleanup_log_files(self) -> None:
        """
        Move clutter log files to output directory after test completion.

        This method:
        - Extracts the JOB_ID from nemo_rl_launcher.log
        - Moves the {JOB_ID}-logs directory to output_path/logs/{JOB_ID}-logs/
        - Moves the slurm-{JOB_ID}.log and slurm-{JOB_ID}.out files to output_path/logs/
        """
        job_id = self._extract_job_id()
        if not job_id:
            logging.warning("Could not extract JOB_ID from launcher log. Skipping cleanup.")
            return

        # Get the workspace root (parent directories up from output_path)
        # The output_path is typically: <workspace_root>/results/<run_name>/Tests.<test_name>/<id>/
        # The log files are in the workspace root
        workspace_root = self.test_run.output_path.parent.parent.parent.parent

        # Create logs directory in output_path
        logs_dir = self.test_run.output_path / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Move {JOB_ID}-logs directory
        log_dir = workspace_root / f"{job_id}-logs"
        if log_dir.exists() and log_dir.is_dir():
            try:
                dest = logs_dir / f"{job_id}-logs"
                shutil.move(str(log_dir), str(dest))
                logging.info(f"Moved log directory: {log_dir} -> {dest}")
            except Exception as e:
                logging.error(f"Failed to move log directory {log_dir}: {e}")

        # Move slurm-{JOB_ID}.log file
        slurm_log = workspace_root / f"slurm-{job_id}.log"
        if slurm_log.exists() and slurm_log.is_file():
            try:
                dest = logs_dir / f"slurm-{job_id}.log"
                shutil.move(str(slurm_log), str(dest))
                logging.info(f"Moved slurm log file: {slurm_log} -> {dest}")
            except Exception as e:
                logging.error(f"Failed to move slurm log file {slurm_log}: {e}")

        # Also check for slurm-{JOB_ID}.out file (common alternative)
        slurm_out = workspace_root / f"slurm-{job_id}.out"
        if slurm_out.exists() and slurm_out.is_file():
            try:
                dest = logs_dir / f"slurm-{job_id}.out"
                shutil.move(str(slurm_out), str(dest))
                logging.info(f"Moved slurm output file: {slurm_out} -> {dest}")
            except Exception as e:
                logging.error(f"Failed to move slurm output file {slurm_out}: {e}")

    def get_metric(self, metric: str) -> float:
        """
        Return specific metric value by name.

        Args:
            metric: Name of the metric to retrieve (e.g., "e2e", "generation", "policy", "default")

        Returns:
            float: Metric value, or METRIC_ERROR if not found.
        """
        # Handle "default" by returning E2E metric
        if metric == "default":
            return self.get_metric("e2e")

        # Check if results file exists and load metrics
        if not self.results_file.exists():
            logging.debug(f"Results file does not exist: {self.results_file}")
            return METRIC_ERROR

        try:
            with self.results_file.open("r") as file:
                metrics_data = json.load(file)
        except (json.JSONDecodeError, Exception) as e:
            logging.error(f"Error reading metrics from {self.results_file}: {e}")
            return METRIC_ERROR

        if not metrics_data:
            logging.debug("Metrics data is empty")
            return METRIC_ERROR

        # Get the JSON path for the requested metric
        metric_lower = metric.lower()
        if metric_lower not in self.METRICS:
            logging.debug(f"Metric '{metric_lower}' not found in METRICS")
            return METRIC_ERROR

        _, json_key = self.METRICS[metric_lower]

        # Get the value directly using the full key (e.g., "performance/tokens_per_sec_per_gpu")
        if json_key not in metrics_data:
            logging.debug(f"Key '{json_key}' not found in metrics data")
            return METRIC_ERROR
        
        current = metrics_data[json_key]
        logging.debug(f"Found value for {metric}: {current} (type: {type(current).__name__})")

        # Handle dict of run metrics (e.g., {"1": value1, "2": value2})
        if isinstance(current, dict):
            try:
                # Extract numeric values and compute average
                values = []
                for k, v in current.items():
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError) as e:
                        logging.debug(f"Skipping non-numeric value for key {k}: {v} ({e})")
                
                if not values:
                    logging.debug(f"No numeric values found in dict: {current}")
                    return METRIC_ERROR
                
                avg = sum(values) / len(values)
                logging.debug(f"Computed average from {len(values)} values: {avg}")
                return avg
            except Exception as e:
                logging.error(f"Error computing average for {metric}: {e}")
                return METRIC_ERROR

        # Handle single value
        try:
            return float(current)
        except (ValueError, TypeError) as e:
            logging.debug(f"Failed to convert to float: {current} ({e})")
            return METRIC_ERROR