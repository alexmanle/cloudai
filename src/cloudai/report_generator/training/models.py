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

"""Data models for training parsers."""

import statistics
from collections.abc import Hashable
from dataclasses import MISSING, dataclass, fields
from typing import Any, List, Optional

SCHEMA_VERSION = "1.0"  # training report schema; bump on breaking changes to TrainingConfig/TrainingResults


@dataclass
class MetricStats:
    """Aggregated statistics for one metric over the filtered steps."""

    mean: float
    min: float
    max: float
    std: float
    t99: float
    t95: float

    @classmethod
    def from_values(cls, values: list[float]) -> "MetricStats":
        """Build stats from a non-empty list of values (population std; inclusive percentiles)."""
        return cls(
            mean=statistics.mean(values),
            min=min(values),
            max=max(values),
            std=statistics.pstdev(values),
            t99=cls._percentile(values, 99),
            t95=cls._percentile(values, 95),
        )

    @staticmethod
    def _percentile(values: list[float], p: int) -> float:
        """Inclusive, linearly-interpolated p-th percentile; returns the sole value for a single sample."""
        if len(values) == 1:
            return float(values[0])
        return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


@dataclass(frozen=True)
class Scalar:
    """A single scalar event from a training run (source-agnostic: TensorBoard today, others later)."""

    tag: str
    step: int
    value: float
    wall_time: float

    @classmethod
    def from_record(cls, record: dict[Hashable, Any]) -> "Scalar":
        """Build from a {column: value} record (e.g. a tbparse DataFrame row)."""
        return cls(tag=record["tag"], step=record["step"], value=record["value"], wall_time=record["wall_time"])


@dataclass(kw_only=True)
class TrainingStep:
    """Results for a single training iteration."""

    iteration: int
    step_time_sec: float
    loss: float
    memory_reserved_bytes: float
    memory_allocated_bytes: float
    tflops_per_gpu: Optional[float] = None  # NeMo FLOPs could be missing for some models


OPTIONAL_STEP_FIELDS = {f.name for f in fields(TrainingStep) if f.default is not MISSING}


@dataclass(kw_only=True)
class StepAggregation:
    """Per-metric aggregated statistics over the filtered steps."""

    step_time_sec: MetricStats
    loss: MetricStats
    memory_reserved_bytes: MetricStats
    memory_allocated_bytes: MetricStats
    tflops_per_gpu: Optional[MetricStats] = None

    @classmethod
    def from_steps(cls, steps: list["TrainingStep"]) -> "StepAggregation":
        """Build per-metric stats from a non-empty list of already-filtered steps."""
        tflops = [s.tflops_per_gpu for s in steps if s.tflops_per_gpu is not None]
        return cls(
            step_time_sec=MetricStats.from_values([s.step_time_sec for s in steps]),
            loss=MetricStats.from_values([s.loss for s in steps]),
            memory_reserved_bytes=MetricStats.from_values([s.memory_reserved_bytes for s in steps]),
            memory_allocated_bytes=MetricStats.from_values([s.memory_allocated_bytes for s in steps]),
            tflops_per_gpu=MetricStats.from_values(tflops) if tflops else None,
        )


@dataclass(kw_only=True)
class TrainingConfig:
    """
    Resolved training configuration from the framework artifact + CloudAI.

    CloudAI-computed fields are supplied by the parser during construction.
    """

    # Test identity
    test_id: str  # scenario section id
    test_name: str  # test definition name
    description: str
    test_scenario_name: str
    test_template_name: str

    # Environment
    container_image: str = ""
    cloudai_execution_node: str
    env_vars: dict[str, Any]  # system global env + test extra env

    # Hardware
    # Depends on: num_nodes, gpus_per_node
    world_size: Optional[int] = None
    # Source: TestRun.nnodes
    num_nodes: int
    gpus_per_node: Optional[int] = None
    nodes: list[str]  # compressed nodelist from the scenario
    # Depends on: env_vars["CLIQUE_SIZE"]
    clique_size: Optional[int] = None

    # Precision
    fp8: Optional[str] = None
    # Depends on: fp8
    fp8_recipe: Optional[str] = None

    # Batch
    micro_batch_size: int
    global_batch_size: int
    seq_length: int

    # Parallelism
    tensor_parallel_size: int
    pipeline_parallel_size: int
    context_parallel_size: Optional[int]
    virtual_pipeline_parallel_size: Optional[int]
    sequence_parallel: bool
    expert_parallel_size: int
    expert_tensor_parallel_size: int
    # Depends on: world_size / (tensor_parallel_size * pipeline_parallel_size * context_parallel_size)
    data_parallel_size: Optional[int] = None

    # Model architecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_query_groups: int
    ffn_hidden_size: int
    kv_channels: int
    normalization: str
    position_embedding_type: str
    model_name: str = ""  # CloudAI-computed

    # MoE
    num_experts: Optional[int]
    moe_router_topk: Optional[int]
    moe_ffn_hidden_size: Optional[int]
    moe_grouped_gemm: Optional[bool]

    # Profiling
    profiling_enabled: bool = False
    # Depends on: profiling_enabled
    profiling_start_step: Optional[int] = None
    # Depends on: profiling_enabled
    profiling_stop_step: Optional[int] = None

    # Aggregation window (steps dropped before computing the top-level aggregation)
    exclude_start_steps: int = 5
    # Depends on: profiling_enabled, profiling_stop_step
    exclude_post_profiling_steps: int = 2


@dataclass(kw_only=True)
class TrainingResults:
    """Container for parsed training output."""

    schema_version: str = SCHEMA_VERSION
    config: TrainingConfig
    steps: List[TrainingStep]
    aggregation: Optional[StepAggregation] = None  # None when no steps remain after exclusions
