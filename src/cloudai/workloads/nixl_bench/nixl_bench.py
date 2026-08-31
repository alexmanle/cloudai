# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import Any, Literal, cast

import pydantic
import toml

import cloudai.metrics
from cloudai.core import JobStatusResult, System, TestRun
from cloudai.systems.slurm import SlurmJobMetadata
from cloudai.util.lazy_imports import lazy
from cloudai.workloads.common.nixl import (
    MANAGED_ETCD_ENDPOINTS,
    NIXLBaseCmdArgs,
    NIXLBaseTestDefinition,
    NIXLExtendedCmdArgs,
    extract_nixlbench_data,
)


class NIXLBenchCmdArgs(NIXLBaseCmdArgs, NIXLExtendedCmdArgs):
    """Command line arguments for a NIXL Bench test."""

    path_to_benchmark: str
    etcd_endpoints: str = MANAGED_ETCD_ENDPOINTS
    runtime_type: Literal["ETCD", "ASIO"] = "ETCD"
    asio_address: str = "$NIXL_ASIO_ADDRESS"
    asio_port: int = pydantic.Field(default=12345, ge=1, le=65535)


class NIXLBenchTestDefinition(NIXLBaseTestDefinition[NIXLBenchCmdArgs]):
    """Test definition for a NIXL Bench test."""

    @property
    def uses_etcd(self) -> bool:
        """Return whether CloudAI should launch ETCD for this benchmark."""
        return self.cmd_args.runtime_type == "ETCD" and self.cmd_args.etcd_endpoints == MANAGED_ETCD_ENDPOINTS

    @property
    def uses_asio(self) -> bool:
        """Return whether this benchmark uses NIXLBench's ASIO runtime."""
        return self.cmd_args.runtime_type == "ASIO"

    @property
    def cmd_args_dict(self) -> dict[str, str | list[str]]:
        cmd_args = self.cmd_args.model_dump(
            exclude={
                "docker_image_url",
                "path_to_benchmark",
                "cmd_args",
                "etcd_path",
                "wait_etcd_for",
                "etcd_image_url",
            },
            exclude_none=True,
        )
        if self.cmd_args.runtime_type == "ETCD":
            cmd_args.pop("runtime_type")
            cmd_args.pop("asio_address")
            cmd_args.pop("asio_port")
            if not self.cmd_args.etcd_endpoints:
                cmd_args.pop("etcd_endpoints")
        else:
            cmd_args.pop("etcd_endpoints")
        return cmd_args

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        slurm_job_path = tr.output_path / "slurm-job.toml"
        if slurm_job_path.is_file():
            with slurm_job_path.open("r", encoding="utf-8") as file:
                metadata = SlurmJobMetadata.model_validate(toml.load(file))

            if metadata.state != "COMPLETED" or metadata.exit_code != "0:0":
                return JobStatusResult(
                    is_successful=False,
                    error_message=(
                        f"NIXLBench Slurm job failed for {tr.output_path}: "
                        f"state={metadata.state}, exit_code={metadata.exit_code}."
                    ),
                )

            benchmark_steps = [
                step for step in metadata.job_steps if self.cmd_args.path_to_benchmark in step.submit_line
            ]
            failed_steps = [step for step in benchmark_steps if step.state != "COMPLETED" or step.exit_code != "0:0"]
            if failed_steps:
                failures = ", ".join(
                    f"{step.job_id}.{step.step_id} state={step.state}, exit_code={step.exit_code}"
                    for step in failed_steps
                )
                return JobStatusResult(
                    is_successful=False,
                    error_message=f"NIXLBench Slurm step failed for {tr.output_path}: {failures}.",
                )

        df = extract_nixlbench_data(tr.output_path / "stdout.txt")
        if df.empty:
            return JobStatusResult(is_successful=False, error_message=f"NIXLBench data not found in {tr.output_path}.")

        return JobStatusResult(is_successful=True)

    def metric_observations(self, system: System, tr: TestRun) -> list[cloudai.metrics.MetricObservation]:
        del system
        csv_path = tr.output_path / "nixlbench.csv"
        df = lazy.pd.read_csv(csv_path) if csv_path.is_file() else extract_nixlbench_data(tr.output_path / "stdout.txt")
        observations: list[cloudai.metrics.MetricObservation] = []
        for row in df.itertuples(index=False):
            row = cast(Any, row)
            dimensions: cloudai.metrics.MetricDimensions = {
                "operation": str(getattr(self.cmd_args, "op_type", None) or "default").lower(),
                "size_bytes": int(row.block_size),
                "batch_size": int(row.batch_size),
                "backend": str(getattr(self.cmd_args, "backend", None) or "default").lower(),
                "source_memory": str(getattr(self.cmd_args, "initiator_seg_type", None) or "default").lower(),
                "target_memory": str(getattr(self.cmd_args, "target_seg_type", None) or "default").lower(),
            }
            observations.extend(
                [
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.LATENCY,
                        float(row.avg_lat),
                        dimensions,
                    ),
                    cloudai.metrics.MetricObservation(
                        cloudai.metrics.BANDWIDTH,
                        float(row.bw_gb_sec),
                        {**dimensions, "bandwidth_basis": "payload"},
                    ),
                ]
            )
        return observations
