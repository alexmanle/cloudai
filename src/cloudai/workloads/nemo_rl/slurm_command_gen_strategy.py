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

import stat
from pathlib import Path
from typing import List, cast

import toml

from cloudai.models.scenario import TestRunDetails
from cloudai.systems.slurm import SlurmCommandGenStrategy

from .nemo_rl import NeMoRLCmdArgs, NeMoRLTestDefinition


class NeMoRLSlurmCommandGenStrategy(SlurmCommandGenStrategy):
    """
    Command generation strategy for NeMo RL on Slurm systems.
    
    Similar to Megatron Bridge: execute the run_nemo_rl.sh script on the submit node,
    which in turn submits the actual job via sbatch.
    """

    def _container_mounts(self) -> List[str]:
        return []

    def image_path(self) -> str | None:
        tdef: NeMoRLTestDefinition = cast(NeMoRLTestDefinition, self.test_run.test)
        return str(tdef.docker_image.installed_path)

    def gen_exec_command(self) -> str:
        """Generate bash command to execute run_nemo_rl.sh script."""
        # Create the bash scripts in the output directory
        exp_script_path = self._create_run_experiment_script()
        self._create_run_nemo_rl_script(exp_script_path)
        
        # Generate a simple bash command that runs run_nemo_rl.sh
        run_script_path = self.test_run.output_path / "run_nemo_rl.sh"
        launcher_cmd = f"bash {run_script_path.absolute()}"
        
        # Wrap the launcher to capture and parse job ID
        full_cmd = self._wrap_launcher_for_job_id_and_quiet_output(launcher_cmd)
        
        self._write_command_to_file(full_cmd, self.test_run.output_path)
        return full_cmd

    def store_test_run(self) -> None:
        """Store test run details to file."""
        test_cmd = self.gen_exec_command()
        trd = TestRunDetails.from_test_run(self.test_run, test_cmd=test_cmd, full_cmd=test_cmd)
        with (self.test_run.output_path / self.TEST_RUN_DUMP_FILE_NAME).open("w") as f:
            toml.dump(trd.model_dump(), f)

    def _write_command_to_file(self, command: str, output_path: Path) -> None:
        """Write the generated command to a file for debugging/tracking."""
        log_file = output_path / "generated_command.sh"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("w") as f:
            f.write(f"{command}\n")

    def _wrap_launcher_for_job_id_and_quiet_output(self, launcher_cmd: str) -> str:
        """
        Run the NeMo RL launcher quietly and ensure CloudAI can parse a job ID.

        CloudAI's SlurmRunner expects stdout to include "Submitted batch job <id>".
        This writes a readable wrapper script (with section breaks) into the test output directory, then runs it.
        """
        output_dir = self.test_run.output_path.absolute()
        output_dir.mkdir(parents=True, exist_ok=True)

        wrapper_path = output_dir / "nemo_rl_submit_and_parse_jobid.sh"
        log_path = output_dir / "nemo_rl_launcher.log"

        script_lines = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            f'LOG="{log_path}"',
            "",
            # Launch NeMo RL (log stdout/stderr to file)
            "",
            ': >"$LOG"',
            f'{launcher_cmd} >>"$LOG" 2>&1',
            "",
            # Parse job id from sbatch output (format: 'Submitted batch job <num>')
            "",
            'JOB_ID=""',
            'JOB_ID=$(grep -Eio "Submitted batch job[: ]+[0-9]+" "$LOG" | tail -n1 | grep -Eo "[0-9]+" | tail -n1 || true)',
            "",
            # Emit a canonical line for CloudAI to parse
            "",
            'if [ -n "${JOB_ID}" ]; then',
            '  echo "Submitted batch job ${JOB_ID}"',
            "else",
            '  echo "Failed to retrieve job ID." >&2',
            '  tail -n 200 "$LOG" >&2 || true',
            "  exit 1",
            "fi",
            "",
        ]

        wrapper_path.write_text("\n".join(script_lines))
        wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR)

        return f"bash {wrapper_path}"

    def _create_run_experiment_script(self) -> Path:
        """Create the run_experiment.sh script in the output directory."""
        tdef: NeMoRLTestDefinition = cast(NeMoRLTestDefinition, self.test_run.test)
        tdef_cmd_args: NeMoRLCmdArgs = tdef.cmd_args
        
        # Get command arguments
        common_path = Path("/opt/nemo-rl/tests/test_suites/llm/performance/common.env")
        model_name = tdef_cmd_args.model_name
        max_steps = tdef_cmd_args.max_steps
        steps_per_run = tdef_cmd_args.steps_per_run

        # Get command arguments
        model_name = tdef_cmd_args.model_name
        num_nodes = self.test_run.nnodes
        num_gpus = self.system.gpus_per_node
        exp_name = f"grpo-{model_name}-{num_nodes}n{num_gpus}g"
        
        script_content = f'''#!/bin/bash
source {common_path}

# ===== BEGIN CONFIG =====
NUM_NODES={self.test_run.nnodes}
STEPS_PER_RUN={steps_per_run}
MAX_STEPS={max_steps}
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))  # Round up
NUM_MINUTES=100
# ===== END CONFIG =====

exit_if_max_steps_reached

# Run the experiment
cd /opt/nemo-rl
uv run examples/run_grpo_math.py \\
    --config $CONFIG_PATH \\
    grpo.max_num_steps=$MAX_STEPS \\
    logger.log_dir=/cloudai_run_results/logs \\
    logger.wandb_enabled=False \\
    logger.wandb.project=nemo-rl \\
    logger.wandb.name={exp_name} \\
    logger.monitor_gpus=True \\
    logger.tensorboard_enabled=True \\
    checkpointing.enabled=True \\
    checkpointing.checkpoint_dir=/cloudai_run_results/ckpts \\
    $@ \\
    2>&1 | tee /cloudai_run_results/run.log

uv run tests/json_dump_tb_logs.py /cloudai_run_results/logs --output_path /cloudai_run_results/metrics.json
'''
        
        script_path = self.test_run.output_path / f"{exp_name}.sh"
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)
        return script_path

    def _create_run_nemo_rl_script(self, exp_script_path: Path) -> Path:
        """Create the run_nemo_rl.sh script."""
        tdef: NeMoRLTestDefinition = cast(NeMoRLTestDefinition, self.test_run.test)
        
        # Get command arguments
        num_nodes = self.test_run.nnodes
        num_gpus = self.system.gpus_per_node

        # Use the generated experiment script
        image_path = self.image_path()
        ray_path = tdef.ray_sub_path
        
        script_content = f'''#!/bin/bash

echo "Setting CONTAINER to {image_path}"
export CONTAINER="{image_path}"

echo "Setting COMMAND to: uv run bash {exp_script_path.absolute()}"
export COMMAND="uv run bash {exp_script_path.absolute()}"

echo "Setting MOUNTS to: /lustre:/lustre:ro"
export MOUNTS="/lustre:/lustre:ro,{self.test_run.output_path.absolute()}:/cloudai_run_results"

echo "Submitting sbatch with:"
echo "  --nodes={num_nodes}"
echo "  --account={self.system.account}"
echo "  --job-name={self.job_name_prefix()}"
echo "  --partition={self.system.default_partition}"
echo "  --time={self.test_run.time_limit}"
echo "  --gres=gpu:{num_gpus}"
echo "  Script: {ray_path}"

sbatch --nodes={num_nodes} --account={self.system.account} --job-name={self.job_name_prefix()} --partition={self.system.default_partition} --time={self.test_run.time_limit} --gres=gpu:{num_gpus} {ray_path}
'''
        
        script_path = self.test_run.output_path / "run_nemo_rl.sh"
        script_path.write_text(script_content, encoding="utf-8")
        script_path.chmod(0o755)
        return script_path
