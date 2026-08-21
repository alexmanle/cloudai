#!/usr/bin/env bash

export LC_ALL=C
. "${CLOUDAI_OS_RELEASE_FILE:-/etc/os-release}" 2>/dev/null || true

inventory() {
    awk 'NF { count[$0]++; if (count[$0] == 1) order[++n] = $0 }
         END { for (i = 1; i <= n; i++) printf "%s%s x%d", (i > 1 ? ", " : ""), order[i], count[order[i]] }'
}

first_or_null() {
    awk 'NF { print; found = 1; exit } END { if (!found) print "null" }'
}

ofed_version() {
    command -v ofed_info >/dev/null || { echo null; return; }
    ofed_info -s 2>/dev/null | sed 's/:$//' | first_or_null
}

libfabric_version() {
    command -v fi_info >/dev/null || { echo null; return; }
    fi_info --version 2>/dev/null | awk 'tolower($0) ~ /libfabric/ { print $2; exit }' | first_or_null
}

cuda_toolkit_version() {
    local version
    version="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -n1)"
    [ -n "$version" ] || version="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
        "${CLOUDAI_CUDA_ROOT:-/usr/local/cuda}/version.json" 2>/dev/null | head -n1)"
    [ -n "$version" ] || version="$(sed -n 's/[^0-9]*\([0-9][0-9.]*\).*/\1/p' \
        "${CLOUDAI_CUDA_ROOT:-/usr/local/cuda}/version.txt" 2>/dev/null | head -n1)"
    printf '%s' "${version:-null}"
}

doca_host_version() {
    local doca_root="${CLOUDAI_DOCA_ROOT:-/opt/mellanox/doca}"
    local doca_info version

    doca_info="$(command -v doca-info 2>/dev/null)"
    doca_info="${doca_info:-$doca_root/tools/doca-info}"
    if [ -x "$doca_info" ]; then
        version="$("$doca_info" 2>/dev/null | awk '
            /^DOCA:/ { found = 1; next }
            found && /^- / { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+\.[0-9]+/) { print $i; exit } }
        ')"
    fi
    if [ -z "$version" ] && command -v dpkg-query >/dev/null 2>&1; then
        version="$(dpkg-query -W -f='${db:Status-Abbrev} ${Version}\n' 'doca-*' 2>/dev/null |
            awk '$1 ~ /^ii/ { print $2; exit }')"
    fi
    if [ -z "$version" ] && command -v rpm >/dev/null 2>&1; then
        version="$(rpm -qa --qf '%{VERSION}-%{RELEASE}\n' 'doca-*' 2>/dev/null | head -n1)"
    fi
    if [ -z "$version" ]; then
        version="$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' \
            "$doca_root/version.json" 2>/dev/null | head -n1)"
    fi
    [ -n "$version" ] || version="$(awk 'NF { print; exit }' "$doca_root/version.txt" 2>/dev/null)"
    printf '%s' "${version:-null}"
}

system_metadata() {
    local gpu_names gpu_model gpu_count gpu_inventory
    local cpu_model cpu_arch cpu_vendor

    gpu_names="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
    gpu_model="$(printf '%s\n' "$gpu_names" | awk 'NF { print; exit }')"
    gpu_count="$(printf '%s\n' "$gpu_names" | awk 'NF { count++ } END { print count + 0 }')"
    gpu_inventory="$(printf '%s\n' "$gpu_names" | inventory)"
    cpu_model="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n1)"
    cpu_arch="$(lscpu 2>/dev/null | sed -n 's/^Architecture:[[:space:]]*//p' | head -n1)"
    cpu_vendor="$(lscpu 2>/dev/null | sed -n 's/^Vendor ID:[[:space:]]*//p' | head -n1)"

    cat <<EOF
[system]
os_type = "${ID:-null}"
os_version = "${VERSION:-${VERSION_ID:-null}}"
linux_kernel_version = "$(uname -r 2>/dev/null || echo null)"
gpu_arch_type = "${gpu_model:-null}"
gpu_count = $gpu_count
gpu_inventory = "${gpu_inventory:-null}"
cpu_model_name = "${cpu_model:-null}"
cpu_arch_type = "${cpu_arch:-null}"
cpu_vendor = "${cpu_vendor:-null}"
EOF
}

physical_network_metadata() {
    local nic_lines nic_models nics nic_count nic_inventory firmware
    local fw_file device

    nic_lines="$(lspci -Dnn 2>/dev/null | awk '
        tolower($0) ~ /ethernet controller|infiniband controller/ &&
        (tolower($0) ~ /mellanox|nvidia/ || tolower($0) ~ /\[15b3:/)
    ')"
    nic_models="$(printf '%s\n' "$nic_lines" | sed -E 's/^[^ ]+ [^:]+: //')"
    nics="$(printf '%s\n' "$nic_models" | awk 'NF { print; exit }')"
    nic_count="$(printf '%s\n' "$nic_lines" | awk 'NF { count++ } END { print count + 0 }')"
    nic_inventory="$(printf '%s\n' "$nic_models" | inventory)"

    for fw_file in "${CLOUDAI_INFINIBAND_SYSFS:-/sys/class/infiniband}"/*/fw_ver; do
        [ -r "$fw_file" ] || continue
        device="${fw_file%/fw_ver}"
        firmware="${firmware:+$firmware, }${device##*/}=$(head -n1 "$fw_file")"
    done

    cat <<EOF
nics = "${nics:-null}"
nic_count = $nic_count
nic_inventory = "${nic_inventory:-null}"
hca_firmware_versions = "${firmware:-null}"
switch_type = "${SWITCH:-null}"
network_name = "${NETWORK:-null}"
mofed_version = "$(ofed_version)"
doca_host_version = "$(doca_host_version)"
EOF
}

runtime_metadata() {
    local driver_version hpcx_version=${HPCX_DIR##*/}
    driver_version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)"

    cat <<EOF
user = "${USER:-$(whoami)}"

[mpi]
mpi_type = "$(mpirun --version 2>/dev/null | grep -qi 'open mpi' && echo openmpi || echo null)"
mpi_version = "$(mpirun --version 2>/dev/null | awk 'tolower($0) ~ /open mpi/ { print $NF; exit }' | first_or_null)"
hpcx_version = "${hpcx_version:-null}"

[cuda]
cuda_build_version = "${CUDA_BUILD_VERSION:-null}"
cuda_runtime_version = "$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | first_or_null)"
cuda_driver_version = "${driver_version:-null}"
nvidia_driver_version = "${driver_version:-null}"
cuda_toolkit_version = "$(cuda_toolkit_version)"

[nccl]
version = "${NCCL_VERSION:-null}"
commit_sha = "${NCCL_COMMIT_SHA:-null}"

[slurm]
cluster_name = "${SLURM_CLUSTER_NAME:-null}"
node_list = "${SLURM_NODELIST:-null}"
num_nodes = "${SLURM_NNODES:-null}"
ntasks_per_node = "${SLURM_NTASKS_PER_NODE:-null}"
ntasks = "${SLURM_NTASKS:-null}"
job_id = "${SLURM_JOBID:-null}"

[network]
libfabric_version = "$(libfabric_version)"
EOF
}

case "${1:-all}" in
    runtime)
        runtime_metadata
        ;;
    host)
        echo
        physical_network_metadata
        echo
        system_metadata
        ;;
    *)
        runtime_metadata
        echo
        physical_network_metadata
        echo
        system_metadata
        ;;
esac
