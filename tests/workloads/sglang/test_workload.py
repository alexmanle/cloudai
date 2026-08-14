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

from cloudai.workloads.sglang import SglangArgs, SglangCmdArgs, SglangTestDefinition


def test_sglang_serve_args_exclude_internal_fields() -> None:
    assert SglangArgs(gpu_ids="0", disaggregation_transfer_backend="nccl").serve_args == []


def test_local_model_is_not_installed_from_hugging_face() -> None:
    tdef = SglangTestDefinition(
        name="test",
        description="test",
        test_template_name="sglang",
        cmd_args=SglangCmdArgs(
            docker_image_url="test_url",
            model="custom-model",
            model_path="/models/custom-model",
        ),
    )

    assert tdef.installables == [tdef.docker_image]
