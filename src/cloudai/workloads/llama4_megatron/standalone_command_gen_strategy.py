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

from typing import cast

from cloudai.core import CommandGenStrategy

from .llama4_megatron import Llama4MegatronCmdArgs, Llama4MegatronTestDefinition


class Llama4MegatronTestStandaloneCommandGenStrategy(CommandGenStrategy):
    """Command generation strategy for llama4_megatron on Standalone systems."""

    def store_test_run(self) -> None:
        pass

    def gen_exec_command(self) -> str:
        """Generate the command to run the llama4_megatron workload."""
        tdef: Llama4MegatronTestDefinition = cast(
            Llama4MegatronTestDefinition, self.test_run.test
        )
        tdef_cmd_args: Llama4MegatronCmdArgs = tdef.cmd_args
        
        # Build the command using the entry point template
        # Entry point: "llama4-megatron-{num_gpus}gpu.sh"
        entry_point = f"{tdef_cmd_args.script_path}/llama4-megatron-{tdef_cmd_args.num_gpus}gpu.sh"
        
        # Build command arguments
        cmd_parts = ["sbatch --parsable ", entry_point]
        
        # Add command arguments
        cmd_parts.append(f"--precision {tdef_cmd_args.precision}")
        cmd_parts.append(f"--dispatcher {tdef_cmd_args.dispatcher}")
        cmd_parts.append(f"--output_log {self.test_run.output_path}")
        cmd_parts.append(f"--project_path {tdef_cmd_args.script_path}")
        cmd_parts.append(f"--ep {tdef_cmd_args.ep}")
        cmd_parts.append(f"--etp {tdef_cmd_args.etp}")
        cmd_parts.append(f"--pp {tdef_cmd_args.pp}")
        cmd_parts.append(f"--tp {tdef_cmd_args.tp}")
        cmd_parts.append(f"--vp {tdef_cmd_args.vp}")
        
        if tdef_cmd_args.deepep_commit:
            cmd_parts.append(f"--deepep-commit {tdef_cmd_args.deepep_commit}")
        
        if tdef_cmd_args.profile:
            cmd_parts.append("--profile")
        
        return " ".join(cmd_parts)

