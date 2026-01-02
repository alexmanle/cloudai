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

import json
from pathlib import Path
from typing import List, Optional, Union

from cloudai.core import DockerImage, GitRepo, Installable, JobStatusResult, TestRun
from cloudai.models.workload import CmdArgs, TestDefinition


class NeMoRLCmdArgs(CmdArgs):
    """NeMo RL workload command arguments."""

    docker_image_url: str
    model_name: Union[str, List[str]] = "llama3.1-8b-instruct"
    max_steps: Union[int, List[int]] = 10
    steps_per_run: Union[int, List[int]] = 10


class NeMoRLTestDefinition(TestDefinition):
    """Test definition for NeMo RL workload."""

    cmd_args: NeMoRLCmdArgs
    _docker_image: Optional[DockerImage] = None

    @property
    def docker_image(self) -> DockerImage:
        if not self._docker_image:
            self._docker_image = DockerImage(url=self.cmd_args.docker_image_url)
        return self._docker_image

    @property
    def nemo_rl_repo(self) -> Optional[GitRepo]:
        """Get the NeMo RL git repository from git_repos if it exists."""
        for repo in self.git_repos:
            if "RL" in repo.repo_name:
                return repo
        return None

    @property
    def ray_sub_path(self) -> Path:
        """Get the path to ray.sub from the installed NeMo RL repository."""
        repo = self.nemo_rl_repo
        assert repo is not None and repo.installed_path is not None, (
            "NeMo RL git repository not found in git_repos. "
            "Please ensure the NeMo RL git repository is installed."
        )
        return repo.installed_path / "ray.sub"

    @property
    def installables(self) -> list[Installable]:
        """Get list of installable objects."""
        return [self.docker_image, *self.git_repos]

    def was_run_successful(self, tr: TestRun) -> JobStatusResult:
        """
        Check if the NeMo RL workload test run completed successfully.

        Args:
            tr: TestRun object containing test run information including output_path.

        Returns:
            JobStatusResult: Object with is_successful (bool) and optional error_message (str).
        """
        output_file = tr.output_path / "metrics.json"

        if not output_file.is_file():
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"metrics.json file not found in the specified output directory {tr.output_path}. "
                    "This file is expected to be created as a result of the NeMo RL test run. "
                    "Please ensure the NeMo RL test was executed properly and that metrics.json is generated. "
                    f"You can run the generated test command manually and verify the creation of {output_file}. "
                    "If the issue persists, contact the system administrator."
                ),
            )

        # Validate JSON file
        try:
            with output_file.open("r") as file:
                data = json.load(file)
                # If we can parse the JSON, consider it successful
                if data:
                    return JobStatusResult(is_successful=True)
                else:
                    return JobStatusResult(
                        is_successful=False,
                        error_message=(
                            f"metrics.json file in {tr.output_path} is empty. "
                            "The NeMo RL test should generate non-empty metrics. "
                            "Please review the test execution logs to understand why no metrics were generated. "
                            "Ensure that the NeMo RL test environment is correctly set up and configured. "
                            "If the issue persists, contact the system administrator."
                        ),
                    )
        except json.JSONDecodeError as e:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Failed to parse metrics.json in {tr.output_path}: {str(e)}. "
                    "The file appears to be corrupted or contains invalid JSON. "
                    "Please review the test execution logs and ensure the NeMo RL test completed properly. "
                    "If the issue persists, contact the system administrator."
                ),
            )
        except Exception as e:
            return JobStatusResult(
                is_successful=False,
                error_message=(
                    f"Unexpected error while reading metrics.json in {tr.output_path}: {str(e)}. "
                    "Please contact the system administrator."
                ),
            )
