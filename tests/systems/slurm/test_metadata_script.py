# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
from pathlib import Path

import toml

from cloudai.systems import slurm
from cloudai.systems.slurm import SlurmSystemMetadata

METADATA_SCRIPT = Path(slurm.__file__).parent / "slurm-metadata.sh"


def _write_command(bin_dir: Path, name: str, body: str) -> None:
    command = bin_dir / name
    command.write_text(f"#!/usr/bin/env bash\n{body}\n")
    command.chmod(0o755)


def _run_collector(tmp_path: Path, commands: dict[str, str], mode: str = "all") -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in commands.items():
        _write_command(bin_dir, name, body)

    os_release = tmp_path / "os-release"
    os_release.write_text('ID=ubuntu\nVERSION="24.04 LTS"\n')
    infiniband_sysfs = tmp_path / "infiniband" / "mlx5_0"
    infiniband_sysfs.mkdir(parents=True)
    (infiniband_sysfs / "fw_ver").write_text("28.43.2026\n")

    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "USER": 'metadata"user',
        "CLOUDAI_OS_RELEASE_FILE": str(os_release),
        "CLOUDAI_INFINIBAND_SYSFS": str(infiniband_sysfs.parent),
        "CLOUDAI_CUDA_ROOT": str(tmp_path / "missing-cuda"),
        "CLOUDAI_DOCA_ROOT": str(tmp_path / "missing-doca"),
        "SLURM_JOBID": "12345",
    }
    return subprocess.run(
        ["/bin/bash", str(METADATA_SCRIPT), mode], env=env, text=True, capture_output=True, check=False
    )


def test_collects_requested_host_metadata(tmp_path: Path) -> None:
    result = _run_collector(
        tmp_path,
        {
            "lscpu": """cat <<'EOF'
Architecture:                         x86_64
Vendor ID:                            GenuineIntel
Model name:                           Intel(R) Xeon(R) Platinum 8573C
EOF""",
            "nvidia-smi": """case "$1" in
    --query-gpu=name) printf 'NVIDIA H100 80GB HBM3\\nNVIDIA H100 80GB HBM3\\n' ;;
    --query-gpu=driver_version) printf '570.124.06\\n570.124.06\\n' ;;
    *) printf '| NVIDIA-SMI 570.124.06 Driver Version: 570.124.06 CUDA Version: 12.8 |\\n' ;;
esac""",
            "nvcc": "printf 'Cuda compilation tools, release 12.6, V12.6.85\\n'",
            "lspci": """cat <<'EOF'
0000:17:00.0 Ethernet controller [0200]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
0000:31:00.0 Infiniband controller [0207]: Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]
0000:65:00.0 VGA compatible controller [0300]: NVIDIA Corporation Device [10de:2330]
EOF""",
            "dpkg-query": """case "$*" in
    *doca-ofed) printf '2.9.2-0.1.0\\n' ;;
    *) exit 1 ;;
esac""",
            "ofed_info": "printf 'MLNX_OFED_LINUX-24.10-1.1.4.0:\\n'",
            "fi_info": "printf 'libfabric: 1.22.0\\n'",
            "lldpctl": """cat <<'EOF'
lldp.eth0.chassis.name=eos-leaf-01
lldp.eth0.chassis.descr=NVIDIA Spectrum-4
EOF""",
            "mpirun": "printf 'mpirun (Open MPI) 4.1.7a1\\n'",
        },
    )

    assert result.returncode == 0
    assert result.stderr == ""
    metadata = toml.loads(result.stdout)
    assert metadata["user"] == 'metadata"user'
    assert metadata["system"]["linux_kernel_version"]
    assert metadata["system"] == {
        "os_type": "ubuntu",
        "os_version": "24.04 LTS",
        "linux_kernel_version": metadata["system"]["linux_kernel_version"],
        "gpu_arch_type": "NVIDIA H100 80GB HBM3",
        "gpu_count": 2,
        "gpu_inventory": "NVIDIA H100 80GB HBM3 x2",
        "cpu_model_name": "Intel(R) Xeon(R) Platinum 8573C",
        "cpu_arch_type": "x86_64",
        "cpu_vendor": "GenuineIntel",
    }
    assert metadata["cuda"] == {
        "cuda_build_version": "null",
        "cuda_runtime_version": "12.8",
        "cuda_driver_version": "570.124.06",
        "nvidia_driver_version": "570.124.06",
        "cuda_toolkit_version": "12.6",
    }
    assert metadata["network"] == {
        "nics": "Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021]",
        "nic_count": 2,
        "nic_inventory": "Mellanox Technologies MT2910 Family [ConnectX-7] [15b3:1021] x2",
        "hca_firmware_versions": "mlx5_0=28.43.2026",
        "switch_type": "NVIDIA Spectrum-4 x1",
        "network_name": "eos-leaf-01 x1",
        "mofed_version": "MLNX_OFED_LINUX-24.10-1.1.4.0",
        "doca_host_version": "2.9.2-0.1.0",
        "libfabric_version": "1.22.0",
    }


def test_missing_subcommands_do_not_fail_or_corrupt_output(tmp_path: Path) -> None:
    failing_commands = {
        command: "exit 42"
        for command in ("lscpu", "nvidia-smi", "nvcc", "lspci", "dpkg-query", "rpm", "mpirun", "ofed_info", "fi_info")
    }
    result = _run_collector(tmp_path, failing_commands)

    assert result.returncode == 0
    metadata = toml.loads(result.stdout)
    assert metadata["system"]["gpu_count"] == 0
    assert metadata["system"]["gpu_arch_type"] == "null"
    assert metadata["cuda"]["cuda_toolkit_version"] == "null"
    assert metadata["cuda"]["nvidia_driver_version"] == "null"
    assert metadata["network"]["nic_count"] == 0
    assert metadata["network"]["doca_host_version"] == "null"


def test_collects_container_metadata_in_legacy_sections(tmp_path: Path) -> None:
    result = _run_collector(
        tmp_path,
        {
            "nvidia-smi": "printf 'NVIDIA-SMI CUDA Version: 13.1\\n'",
            "nvcc": "printf 'Cuda compilation tools, release 13.0, V13.0.1\\n'",
            "mpirun": "printf 'mpirun (Open MPI) 4.1.9a1\\n'",
        },
        mode="runtime",
    )

    assert result.returncode == 0
    assert result.stderr == ""
    metadata = toml.loads(result.stdout)
    assert "runtime" not in metadata
    assert "system" not in metadata
    assert metadata["network"] == {"libfabric_version": "null"}
    assert metadata["mpi"] == {"mpi_type": "openmpi", "mpi_version": "4.1.9a1", "hpcx_version": "null"}
    assert metadata["cuda"]["cuda_runtime_version"] == "13.1"
    assert metadata["cuda"]["cuda_toolkit_version"] == "13.0"


def test_collects_only_host_owned_sections_in_host_mode(tmp_path: Path) -> None:
    result = _run_collector(tmp_path, {}, mode="host")

    assert result.returncode == 0
    assert result.stdout.startswith("\nnics = ")
    assert "\n[system]\n" in result.stdout
    assert "\n[mpi]\n" not in result.stdout
    assert "\n[nccl]\n" not in result.stdout
    assert "libfabric_version" not in result.stdout


def test_runtime_and_host_outputs_form_one_valid_metadata_document(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    host_dir = tmp_path / "host"
    runtime_dir.mkdir()
    host_dir.mkdir()

    runtime_result = _run_collector(runtime_dir, {}, mode="runtime")
    host_result = _run_collector(host_dir, {}, mode="host")
    metadata = toml.loads(f"{runtime_result.stdout}\n{host_result.stdout}")

    parsed = SlurmSystemMetadata.model_validate(metadata)
    assert parsed.system.os_type == "ubuntu"
    assert parsed.mpi.mpi_type == "null"
