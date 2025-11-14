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

import logging
import re
from pathlib import Path
from typing import ClassVar

from cloudai.core import METRIC_ERROR, ReportGenerationStrategy
from cloudai.util.lazy_imports import lazy


class Llama4MegatronTestReportGenerationStrategy(ReportGenerationStrategy):
    """Strategy for generating reports from llama4_megatron test outputs."""

    metrics: ClassVar[list[str]] = [
        "default",
        "elapsed_time_per_iteration_ms",
        "throughput_per_gpu_tflops"
    ]

    def can_handle_directory(self) -> bool:
        """Check if this strategy can process the test run's output directory."""
        # Look for output files matching the pattern
        output_files = list(self.test_run.output_path.glob("llama4-*.out"))
        
        if output_files:
            # Check the first matching file for success keyword
            with output_files[0].open("r") as file:
                content = file.read()
                return "Configuration: " in content
        
        # Also check for stdout.txt as fallback
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            with stdout_path.open("r") as file:
                content = file.read()
                return "Configuration: " in content
        
        return False

    def _find_output_file(self) -> Path | None:
        """Find the output file for this test run."""
        # Look for files matching the pattern
        output_files = list(self.test_run.output_path.glob("llama4-*.out"))
        if output_files:
            return output_files[0]
        
        # Fallback to stdout.txt
        stdout_path = self.test_run.output_path / "stdout.txt"
        if stdout_path.exists():
            return stdout_path
        
        return None

    def _extract_metrics(self, output_file: Path) -> dict[str, list[float]]:
        """Extract metrics from the output file."""
        metrics_data = {
            "elapsed_time_per_iteration_ms": [],
            "throughput_per_gpu_tflops": []
        }
        
        with output_file.open("r") as file:
            content = file.read()
            
            # Extract elapsed time per iteration
            # Pattern: "elapsed time per iteration (ms): 123.45"
            elapsed_time_matches = re.findall(
                r"elapsed time per iteration \(ms\):\s*([\d.]+)",
                content,
                re.IGNORECASE
            )
            if elapsed_time_matches:
                metrics_data["elapsed_time_per_iteration_ms"] = [
                    float(x) for x in elapsed_time_matches
                ]
            
            # Extract throughput per GPU
            # Pattern: "throughput per GPU (TFLOP/s/GPU): 123.45"
            throughput_matches = re.findall(
                r"throughput per GPU \(TFLOP/s/GPU\):\s*([\d.]+)",
                content,
                re.IGNORECASE
            )
            if throughput_matches:
                metrics_data["throughput_per_gpu_tflops"] = [
                    float(x) for x in throughput_matches
                ]
        
        return metrics_data

    def generate_report(self) -> None:
        """Parse test output and generate report files."""
        output_file = self._find_output_file()
        if not output_file:
            logging.error(f"No output file found in {self.test_run.output_path}")
            return
        
        # Extract metrics
        metrics_data = self._extract_metrics(output_file)
        
        # Check if we have any data
        has_data = any(len(v) > 0 for v in metrics_data.values())
        if not has_data:
            logging.warning(
                f"No valid metrics found in {output_file}. Report generation may be incomplete."
            )
        
        # Compute statistics
        report_lines = ["Llama4 Megatron Test Report", "=" * 40, ""]
        
        for metric_name, data_points in metrics_data.items():
            if data_points:
                stats = {
                    "count": len(data_points),
                    "mean": lazy.np.mean(data_points),
                    "median": lazy.np.median(data_points),
                    "min": lazy.np.min(data_points),
                    "max": lazy.np.max(data_points),
                    "std": lazy.np.std(data_points)
                }
                
                report_lines.append(f"{metric_name.replace('_', ' ').title()}:")
                for stat_name, stat_value in stats.items():
                    report_lines.append(f"  {stat_name.capitalize()}: {stat_value:.4f}")
                report_lines.append("")
        
        # Write report file
        report_file = self.test_run.output_path / "report.txt"
        with report_file.open("w") as f:
            f.write("\n".join(report_lines))
        
        logging.info(f"Report generated: {report_file}")

    def get_metric(self, metric: str) -> float:
        """Return a specific metric value by name."""
        logging.debug(f"Getting metric {metric} from {self.test_run.output_path}")
        
        output_file = self._find_output_file()
        if not output_file:
            return METRIC_ERROR
        
        metrics_data = self._extract_metrics(output_file)
        
        # Map metric names to data
        if metric == "default" or metric == "throughput_per_gpu_tflops":
            data_points = metrics_data.get("throughput_per_gpu_tflops", [])
            if data_points:
                return float(lazy.np.mean(data_points))
        elif metric == "elapsed_time_per_iteration_ms":
            data_points = metrics_data.get("elapsed_time_per_iteration_ms", [])
            if data_points:
                return float(lazy.np.mean(data_points))
        
        return METRIC_ERROR

