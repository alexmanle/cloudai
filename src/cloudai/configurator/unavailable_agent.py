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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict

from cloudai._core.exceptions import TestScenarioParsingError

from .base_agent import BaseAgent, BaseAgentConfig
from .base_gym import BaseGym

if TYPE_CHECKING:
    from cloudai._core.test_scenario import TestScenario


class UnavailableAgentConfig(BaseAgentConfig):
    """
    Configuration for an unavailable agent.

    Validates common agent settings and accepts other fields without validation.
    """

    model_config = ConfigDict(extra="allow")


class UnavailableAgent(BaseAgent):
    """
    Placeholder for an agent with missing optional dependencies.

    Direct use raises ``ImportError`` with ``reason``. Override ``get_config_class()``
    to return the real agent's configuration model if it can be imported without
    the missing dependency.
    """

    reason: ClassVar[str] = "The agent implementation is unavailable."

    def __init__(self, env: BaseGym, config: BaseAgentConfig) -> None:
        raise ImportError(self.reason)

    @staticmethod
    def get_config_class() -> type[BaseAgentConfig]:
        return UnavailableAgentConfig

    def configure(self, config: dict[str, Any]) -> None:
        raise ImportError(self.reason)

    def select_action(self, observation: list[float] | None = None) -> tuple[int, dict[str, Any]] | None:
        raise ImportError(self.reason)

    def update_policy(self, _feedback: dict[str, Any]) -> None:
        raise ImportError(self.reason)

    def run(self) -> int:
        raise ImportError(self.reason)


def validate_available_agents(test_scenario: TestScenario) -> None:
    """
    Check that DSE agents in the scenario and its hooks are available.

    Args:
        test_scenario (TestScenario): The scenario with final test configurations.

    Raises:
        TestScenarioParsingError: A DSE test requires an unavailable agent.
    """
    from cloudai._core.registry import Registry

    registry = Registry()
    for tr in test_scenario.test_runs:
        if tr.is_dse_job and registry.has_agent(tr.test.agent):
            agent = registry.get_agent(tr.test.agent)
            if issubclass(agent, UnavailableAgent):
                raise TestScenarioParsingError(f"Test '{tr.name}' requires agent '{tr.test.agent}': {agent.reason}")
        for hook in (tr.pre_test, tr.post_test):
            if hook is not None:
                validate_available_agents(hook)
