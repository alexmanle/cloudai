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

import argparse
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import toml
from pydantic import ValidationError

import cloudai.cli.handlers as handlers
from cloudai.configurator.env_params import EnvParamSpec, validate_domain_randomization_active
from cloudai.configurator.unavailable_agent import validate_available_agents
from cloudai.core import (
    BaseAgentConfig,
    Parser,
    Registry,
    TestRun,
    TestScenario,
    TestScenarioParsingError,
    UnavailableAgent,
)
from cloudai.models.workload import CmdArgs, TestDefinition


class MissingAgent(UnavailableAgent):
    reason = "Install the 'optional-agent' package."


@pytest.fixture
def missing_agent(monkeypatch: pytest.MonkeyPatch) -> str:
    name = "test_unavailable_agent"
    monkeypatch.setitem(Registry().agents_map, name, MissingAgent)
    return name


def test_placeholder_preserves_custom_settings_and_validates_common_settings() -> None:
    config_class = MissingAgent.get_config_class()
    config = config_class.model_validate({"start_action": "first", "custom_option": {"choices": [1, 2]}})
    assert config.start_action == "first"
    assert config.model_dump()["custom_option"] == {"choices": [1, 2]}
    with pytest.raises(ValidationError, match="start_action"):
        config_class.model_validate({"start_action": "invalid"})


def test_placeholder_can_use_dependency_independent_config_schema(
    base_tr: TestRun, missing_agent: str, monkeypatch: pytest.MonkeyPatch
):
    class CustomConfig(BaseAgentConfig):
        trials: int

    monkeypatch.setattr(MissingAgent, "get_config_class", staticmethod(lambda: CustomConfig))
    data = {**base_tr.test.model_dump(), "agent": missing_agent, "agent_config": {"trials": "bad"}}
    with pytest.raises(ValidationError, match="trials"):
        TestDefinition.model_validate(data)


def test_placeholder_fails_on_instantiation_without_touching_environment() -> None:
    env = Mock()
    with pytest.raises(ImportError, match="optional-agent"):
        MissingAgent(env, MissingAgent.get_config_class()())
    assert not env.mock_calls


@pytest.fixture
def configs(tmp_path: Path, missing_agent: str) -> tuple[Path, Path, Path]:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    for name, agent in (("working", "grid_search"), ("optional", missing_agent)):
        (tests_dir / f"{name}.toml").write_text(
            toml.dumps(
                {
                    "name": name,
                    "description": name,
                    "test_template_name": "Sleep",
                    "agent": agent,
                    "agent_config": {"start_action": "first"},
                    "extra_env_vars": {"SWEEP": ["1", "2"]},
                    "cmd_args": {"seconds": 1},
                }
            )
        )
    scenario = tmp_path / "scenario.toml"
    scenario.write_text(toml.dumps({"name": "s", "Tests": [{"id": "selected", "test_name": "working"}]}))
    system = Path("conf/common/system/standalone_system.toml")
    return system, tests_dir, scenario


def test_unused_placeholder_does_not_block_scenario(configs: tuple[Path, Path, Path], caplog):
    system, tests_dir, scenario = configs
    _, tests, selected = Parser(system, tests_dir / "no-hooks").parse(tests_dir, scenario)
    assert [t.name for t in tests] == ["working"]
    assert selected is not None
    validate_available_agents(selected)
    assert "optional-agent" in caplog.text
    assert "agent-specific validation is deferred" in caplog.text


def test_verify_configs_accepts_placeholder_and_warns(configs: tuple[Path, Path, Path], caplog):
    _, tests_dir, scenario = configs
    assert handlers.verify_test_configs(list(tests_dir.glob("*.toml"))) == 0
    assert handlers.verify_test_scenarios([scenario], list(tests_dir.glob("*.toml")), [], []) == 0
    assert "is unavailable" in caplog.text


@pytest.mark.parametrize(
    "mode,single_sbatch,override",
    [("run", False, False), ("dry-run", True, True)],
)
def test_selected_placeholder_fails_before_side_effects(
    configs: tuple[Path, Path, Path],
    missing_agent: str,
    monkeypatch: pytest.MonkeyPatch,
    caplog,
    mode: str,
    single_sbatch: bool,
    override: bool,
):
    system, tests_dir, scenario = configs
    test: dict[str, Any] = {"id": "selected"}
    if override:
        test.update(test_name="working", agent=missing_agent, agent_config={"custom_option": True})
    else:
        test["test_name"] = "optional"
    scenario.write_text(toml.dumps({"name": "s", "Tests": [test]}))
    prepare_output = Mock()
    installation = Mock()
    runner = Mock()
    monkeypatch.setattr(handlers, "prepare_output_dir", prepare_output)
    monkeypatch.setattr(handlers, "_check_installation", installation)
    monkeypatch.setattr(handlers, "Runner", runner)
    args = argparse.Namespace(
        system_config=system,
        tests_dir=tests_dir,
        test_scenario=scenario,
        hook_dir=tests_dir / "no-hooks",
        output_dir=None,
        mode=mode,
        single_sbatch=single_sbatch,
    )
    assert handlers.handle_dry_run_and_run(args) == 1
    assert "selected" in caplog.text and missing_agent in caplog.text and "optional-agent" in caplog.text
    prepare_output.assert_not_called()
    installation.assert_not_called()
    runner.assert_not_called()


def test_selected_hooks_are_checked(base_tr: TestRun, missing_agent: str):
    hook_test = TestDefinition(
        name="hook", description="d", test_template_name="test", agent=missing_agent, cmd_args=CmdArgs()
    )
    hook_run = TestRun(name="hook-run", test=hook_test, num_nodes=[1, 2], nodes=[])
    base_tr.post_test = TestScenario(name="hook", test_runs=[hook_run])
    with pytest.raises(TestScenarioParsingError, match=r"hook-run.*optional-agent"):
        validate_available_agents(TestScenario(name="s", test_runs=[base_tr]))


def test_non_dse_placeholder_is_not_required(base_tr: TestRun, missing_agent: str):
    base_tr.test.agent = missing_agent
    validate_available_agents(TestScenario(name="s", test_runs=[base_tr]))


def test_placeholder_defers_capability_checks(base_tr: TestRun, missing_agent: str):
    base_tr.test.agent = missing_agent
    base_tr.test.env_params = {"sample": EnvParamSpec()}
    base_tr.num_nodes = [1, 2]
    scenario = TestScenario(name="s", test_runs=[base_tr])
    validate_domain_randomization_active(scenario)
    with pytest.raises(TestScenarioParsingError, match="optional-agent"):
        validate_available_agents(scenario)
