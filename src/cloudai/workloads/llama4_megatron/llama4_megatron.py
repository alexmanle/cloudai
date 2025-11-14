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

from typing import Optional

from cloudai.core import Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class Llama4MegatronCmdArgs(CmdArgs):
    """Llama4 Megatron workload command arguments."""

    docker_image_url: str = ""
    script_path: str = ""
    num_gpus: int = 64
    precision: str = "bf16"
    dispatcher: str = "nccl"
    ep: int = 1
    etp: int = 1
    pp: int = 1
    tp: int = 1
    vp: int = 1
    deepep_commit: str = ""
    profile: bool = False


class Llama4MegatronTestDefinition(TestDefinition):
    """Test definition for llama4_megatron workload."""

    cmd_args: Llama4MegatronCmdArgs

    @property
    def installables(self) -> list[Installable]:
        return []

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        """
        Check if the Llama4 Megatron run was successful.
        
        Args:
            tr: TestRun object containing output path and test information
            
        Returns:
            JobStatusResult indicating success or failure with error message
        """
        # Construct the expected output file path based on the pattern
        # Pattern: llama4-${precision}-ep${ep}etp${etp}pp${pp}tp${tp}vp${vp}-${dispatcher}-${SLURM_JOB_ID}.out
        # Since we don't know the exact SLURM_JOB_ID, we'll look for files matching the pattern
        output_files = list(tr.output_path.glob("llama4-*.out"))
        
        if not output_files:
            # Also check for stdout.txt as a fallback
            stdout_path = tr.output_path / "stdout.txt"
            if stdout_path.is_file():
                output_file = stdout_path
            else:
                return JobStatusResult(
                    is_successful=False,
                    error_message=(
                        f"No output files found in {tr.output_path}. "
                        "Expected to find files matching pattern 'llama4-*.out' or 'stdout.txt'. "
                        "Please ensure the Llama4 Megatron workload ran to completion and generated output files. "
                        "If the issue persists, check the job logs and contact the system administrator."
                    ),
                )
        else:
            # Use the first matching output file
            output_file = output_files[0]
        
        # Read and check the output file for success indicators
        with output_file.open("r") as file:
            content = file.read()
            
            # Check for error patterns first
            error_patterns = []
            if error_patterns:
                for pattern in error_patterns:
                    if pattern in content:
                        return JobStatusResult(
                            is_successful=False,
                            error_message=(
                                f"Error pattern '{pattern}' detected in {output_file}. "
                                "Please review the output file for detailed error information. "
                                "Common issues include configuration errors, out of memory errors, or "
                                "communication failures. If the issue persists, contact the system administrator."
                            ),
                        )
            
            # Check for success keywords
            success_keyword = "Configuration: "
            if success_keyword in content:
                return JobStatusResult(is_successful=True)
            
            # If we reach here, success keyword was not found
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Success keyword '{success_keyword}' not found in {output_file}. "
                    "This keyword is expected to be present in the output, usually at the beginning of the run. "
                    "Please review the output file to determine if the workload started properly. "
                    "Ensure that the Llama4 Megatron script is correctly configured and all dependencies are available. "
                    "If the issue persists, contact the system administrator."
                ),
            )

