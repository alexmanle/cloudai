#!/usr/bin/env bash

# Metadata is best effort: a missing utility or an unsupported query must not
# fail the Slurm step or prevent the remaining fields from being collected.
set +e
set +u
set +o pipefail 2>/dev/null || true
export LC_ALL=C

readonly UNKNOWN="null"

safe_probe() {
    local output
    output="$("$@" 2>/dev/null)" || output=""
    if [ -z "$output" ]; then
        output="$UNKNOWN"
    fi
    printf '%s' "$output"
    return 0
}

bounded_command() {
    local seconds="$1"
    shift
    if command -v timeout >/dev/null 2>&1; then
        timeout "$seconds" "$@"
    else
        "$@"
    fi
}

toml_escape() {
    local value="${1:-$UNKNOWN}"
    value=${value//$'\r'/ }
    value=${value//$'\n'/, }
    value=${value//\\/\\\\}
    value=${value//\"/\\\"}
    printf '%s' "$value"
    return 0
}

emit_string() {
    printf '%s = "%s"\n' "$1" "$(toml_escape "$2")"
    return 0
}

emit_integer() {
    local value="$2"
    case "$value" in
        '' | *[!0-9]*) value=0 ;;
    esac
    printf '%s = %s\n' "$1" "$value"
    return 0
}

lscpu_field() {
    local label="$1"
    lscpu | awk -F: -v label="$label" '$1 == label {sub(/^[[:space:]]+/, "", $2); print $2; exit}'
}

gpu_names_probe() {
    nvidia-smi --query-gpu=name --format=csv,noheader
}

gpu_driver_probe() {
    nvidia-smi --query-gpu=driver_version --format=csv,noheader | awk 'NF {print; exit}'
}

driver_cuda_compat_probe() {
    nvidia-smi | awk '
        match($0, /CUDA Version: [0-9.]+/) {
            value = substr($0, RSTART, RLENGTH)
            sub(/^CUDA Version: /, "", value)
            print value
            exit
        }
    '
}

inventory_from_lines() {
    awk '
        NF {
            if (!seen[$0]++) {
                order[++count] = $0
            }
        }
        END {
            for (i = 1; i <= count; i++) {
                if (i > 1) printf ", "
                printf "%s x%d", order[i], seen[order[i]]
            }
        }
    '
}

nonempty_line_count() {
    awk 'NF {count++} END {print count + 0}'
}

cuda_toolkit_version_probe() {
    local cuda_root="${CLOUDAI_CUDA_ROOT:-/usr/local/cuda}"
    local version_file
    local version

    if command -v nvcc >/dev/null 2>&1; then
        version="$(nvcc --version 2>/dev/null | awk '
            match($0, /release [0-9.]+/) {
                value = substr($0, RSTART, RLENGTH)
                sub(/^release /, "", value)
                print value
                exit
            }
        ')"
        if [ -n "$version" ]; then
            printf '%s' "$version"
            return 0
        fi
    fi

    version_file="$cuda_root/version.json"
    if [ -r "$version_file" ]; then
        version="$(awk -F'"' '/"version"[[:space:]]*:/ {print $4; exit}' "$version_file" 2>/dev/null)"
        if [ -n "$version" ]; then
            printf '%s' "$version"
            return 0
        fi
    fi

    version_file="$cuda_root/version.txt"
    if [ -r "$version_file" ]; then
        version="$(awk 'match($0, /[0-9]+\.[0-9]+([.][0-9]+)?/) {print substr($0, RSTART, RLENGTH); exit}' \
            "$version_file" 2>/dev/null)"
        if [ -n "$version" ]; then
            printf '%s' "$version"
            return 0
        fi
    fi

    return 1
}

nic_lines_probe() {
    lspci -Dnn | awk '
        {
            line = tolower($0)
            is_network = line ~ /ethernet controller|infiniband controller/
            is_nvidia = line ~ /\[15b3:/ || line ~ /mellanox/ || line ~ /nvidia/
            if (is_network && is_nvidia) print
        }
    '
}

nic_models_from_lines() {
    awk '
        NF {
            sub(/^[^[:space:]]+[[:space:]]+/, "")
            sub(/^[^:]+:[[:space:]]*/, "")
            print
        }
    '
}

hca_firmware_probe() {
    local fw_file
    local device_path
    local device_name
    local firmware
    local separator=""
    local sysfs_root="${CLOUDAI_INFINIBAND_SYSFS:-/sys/class/infiniband}"

    for fw_file in "$sysfs_root"/*/fw_ver; do
        [ -r "$fw_file" ] || continue
        device_path=${fw_file%/fw_ver}
        device_name=${device_path##*/}
        firmware="$(awk 'NF {print; exit}' "$fw_file" 2>/dev/null)"
        [ -n "$firmware" ] || continue
        printf '%s%s=%s' "$separator" "$device_name" "$firmware"
        separator=", "
    done
    [ -n "$separator" ]
}

doca_host_version_probe() {
    local doca_root="${CLOUDAI_DOCA_ROOT:-/opt/mellanox/doca}"
    local package
    local version
    local version_file

    if command -v doca-info >/dev/null 2>&1; then
        version="$(bounded_command 10 doca-info 2>/dev/null | awk '
            /^DOCA:/ {in_doca = 1; next}
            in_doca && /^- / {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^[0-9]+\.[0-9]+/) {
                        print $i
                        exit
                    }
                }
            }
            in_doca && /^[^ -]/ {exit}
        ')"
        if [ -n "$version" ]; then
            printf '%s' "$version"
            return 0
        fi
    fi

    if command -v dpkg-query >/dev/null 2>&1; then
        for package in \
            doca-host doca-all doca-networking doca-ofed doca-roce doca-host-basic \
            doca-runtime doca-sdk doca-tools doca-extra doca-cx-runtime doca-cx-sdk doca-cx-tools; do
            version="$(dpkg-query -W -f='${Version}' "$package" 2>/dev/null)"
            if [ -n "$version" ]; then
                printf '%s' "$version"
                return 0
            fi
        done
    fi

    if command -v rpm >/dev/null 2>&1; then
        for package in \
            doca-host doca-all doca-networking doca-ofed doca-roce doca-host-basic \
            doca-runtime doca-sdk doca-tools doca-extra doca-cx-runtime doca-cx-sdk doca-cx-tools; do
            version="$(rpm -q --qf '%{VERSION}-%{RELEASE}' "$package" 2>/dev/null)"
            if [ -n "$version" ]; then
                printf '%s' "$version"
                return 0
            fi
        done
    fi

    for version_file in "$doca_root/version.json" "$doca_root/version.txt"; do
        [ -r "$version_file" ] || continue
        version="$(awk -F'"' '/"version"[[:space:]]*:/ {print $4; exit}' "$version_file" 2>/dev/null)"
        if [ -z "$version" ]; then
            version="$(awk 'NF {print; exit}' "$version_file" 2>/dev/null)"
        fi
        if [ -n "$version" ]; then
            printf '%s' "$version"
            return 0
        fi
    done

    return 1
}

mpi_type_probe() {
    mpirun --version | awk 'tolower($0) ~ /open mpi/ {print "openmpi"; exit}'
}

mpi_version_probe() {
    mpirun --version | awk 'tolower($0) ~ /open mpi/ {print $NF; exit}'
}

ofed_version_probe() {
    command -v ofed_info >/dev/null 2>&1 || return 1
    ofed_info -s | sed 's/:$//'
}

libfabric_version_probe() {
    local version

    if command -v fi_info >/dev/null 2>&1; then
        version="$(bounded_command 10 fi_info --version 2>/dev/null | awk '
            tolower($0) ~ /libfabric|fi_info/ {
                for (i = 1; i <= NF; i++) {
                    if ($i ~ /^[0-9]+\.[0-9]+/) {
                        print $i
                        exit
                    }
                }
            }
        ')"
        [ -z "$version" ] || { printf '%s' "$version"; return 0; }
    fi

    if command -v pkg-config >/dev/null 2>&1; then
        version="$(pkg-config --modversion libfabric 2>/dev/null)"
        [ -z "$version" ] || { printf '%s' "$version"; return 0; }
    fi

    if command -v dpkg-query >/dev/null 2>&1; then
        for package in libfabric1 libfabric-dev; do
            version="$(dpkg-query -W -f='${Version}' "$package" 2>/dev/null)"
            [ -z "$version" ] || { printf '%s' "$version"; return 0; }
        done
    fi

    if command -v rpm >/dev/null 2>&1; then
        version="$(rpm -q --qf '%{VERSION}-%{RELEASE}' libfabric 2>/dev/null)"
        [ -z "$version" ] || { printf '%s' "$version"; return 0; }
    fi

    return 1
}

lldp_field_probe() {
    local field="$1"
    local output

    if command -v lldpctl >/dev/null 2>&1; then
        output="$(lldpctl -f keyvalue 2>/dev/null)"
    elif command -v lldpcli >/dev/null 2>&1; then
        output="$(lldpcli show neighbors -f keyvalue 2>/dev/null)"
    else
        return 1
    fi

    printf '%s\n' "$output" | awk -F= -v field="$field" '
        index($1, field) {print $2}
    ' | inventory_from_lines
}

switch_type_probe() {
    local value

    value="$(lldp_field_probe '.chassis.descr')"
    [ -z "$value" ] || { printf '%s' "$value"; return 0; }

    if command -v ibswitches >/dev/null 2>&1; then
        value="$(bounded_command 10 ibswitches 2>/dev/null | awk -F'"' 'NF >= 2 {print $2}' | inventory_from_lines)"
        [ -z "$value" ] || { printf '%s' "$value"; return 0; }
    fi

    return 1
}

network_name_probe() {
    local value

    value="$(lldp_field_probe '.chassis.name')"
    [ -z "$value" ] || { printf '%s' "$value"; return 0; }

    if command -v fi_info >/dev/null 2>&1; then
        value="$(bounded_command 10 fi_info 2>/dev/null | awk -F': ' '$1 ~ /^[[:space:]]*fabric$/ {print $2}' | inventory_from_lines)"
        [ -z "$value" ] || { printf '%s' "$value"; return 0; }
    fi

    return 1
}

os_release_file="${CLOUDAI_OS_RELEASE_FILE:-/etc/os-release}"
if [ -r "$os_release_file" ]; then
    # shellcheck disable=SC1090
    . "$os_release_file" 2>/dev/null || true
fi

user="${USER:-$(safe_probe whoami)}"
kernel_version="$(safe_probe uname -r)"
cpu_model="$(safe_probe lscpu_field "Model name")"
cpu_arch="$(safe_probe lscpu_field "Architecture")"
cpu_vendor="$(safe_probe lscpu_field "Vendor ID")"

gpu_names="$(gpu_names_probe 2>/dev/null)" || gpu_names=""
gpu_arch_type="$(printf '%s\n' "$gpu_names" | awk 'NF {print; exit}' 2>/dev/null)"
gpu_arch_type="${gpu_arch_type:-$UNKNOWN}"
gpu_count="$(printf '%s\n' "$gpu_names" | nonempty_line_count 2>/dev/null)" || gpu_count=0
gpu_inventory="$(printf '%s\n' "$gpu_names" | inventory_from_lines 2>/dev/null)"
gpu_inventory="${gpu_inventory:-$UNKNOWN}"

nic_lines="$(nic_lines_probe 2>/dev/null)" || nic_lines=""
nic_models="$(printf '%s\n' "$nic_lines" | nic_models_from_lines 2>/dev/null)"
nics="$(printf '%s\n' "$nic_models" | awk 'NF {print; exit}' 2>/dev/null)"
nics="${nics:-$UNKNOWN}"
nic_count="$(printf '%s\n' "$nic_lines" | nonempty_line_count 2>/dev/null)" || nic_count=0
nic_inventory="$(printf '%s\n' "$nic_models" | inventory_from_lines 2>/dev/null)"
nic_inventory="${nic_inventory:-$UNKNOWN}"

emit_system_metadata() {
    emit_string os_type "${ID:-$UNKNOWN}"
    emit_string os_version "${VERSION:-${VERSION_ID:-$UNKNOWN}}"
    emit_string linux_kernel_version "$kernel_version"
    emit_string gpu_arch_type "$gpu_arch_type"
    emit_integer gpu_count "$gpu_count"
    emit_string gpu_inventory "$gpu_inventory"
    emit_string cpu_model_name "$cpu_model"
    emit_string cpu_arch_type "$cpu_arch"
    emit_string cpu_vendor "$cpu_vendor"
}

emit_physical_network_metadata() {
    emit_string nics "$nics"
    emit_integer nic_count "$nic_count"
    emit_string nic_inventory "$nic_inventory"
    emit_string hca_firmware_versions "$(safe_probe hca_firmware_probe)"
    emit_string switch_type "${SWITCH:-$(safe_probe switch_type_probe)}"
    emit_string network_name "${NETWORK:-$(safe_probe network_name_probe)}"
    emit_string mofed_version "$(safe_probe ofed_version_probe)"
    emit_string doca_host_version "$(safe_probe doca_host_version_probe)"
}

mode="${1:-all}"
if [ "$mode" != "host" ]; then
    emit_string user "$user"
fi

if [ "$mode" = "all" ]; then
    printf '\n[system]\n'
    emit_system_metadata

    printf '\n[network]\n'
    emit_physical_network_metadata
    emit_string libfabric_version "$(safe_probe libfabric_version_probe)"
fi

if [ "$mode" != "host" ]; then
    printf '\n[mpi]\n'
    emit_string mpi_type "$(safe_probe mpi_type_probe)"
    emit_string mpi_version "$(safe_probe mpi_version_probe)"
    hpcx_version=${HPCX_DIR##*/}
    emit_string hpcx_version "${hpcx_version:-$UNKNOWN}"

    printf '\n[cuda]\n'
    emit_string cuda_build_version "${CUDA_BUILD_VERSION:-$UNKNOWN}"
    emit_string cuda_runtime_version "$(safe_probe driver_cuda_compat_probe)"
    driver_version="$(safe_probe gpu_driver_probe)"
    emit_string cuda_driver_version "$driver_version"
    emit_string nvidia_driver_version "$driver_version"
    emit_string cuda_toolkit_version "$(safe_probe cuda_toolkit_version_probe)"

    printf '\n[nccl]\n'
    emit_string version "${NCCL_VERSION:-$UNKNOWN}"
    emit_string commit_sha "${NCCL_COMMIT_SHA:-$UNKNOWN}"

    printf '\n[slurm]\n'
    emit_string cluster_name "${SLURM_CLUSTER_NAME:-$UNKNOWN}"
    emit_string node_list "${SLURM_NODELIST:-$UNKNOWN}"
    emit_string num_nodes "${SLURM_NNODES:-$UNKNOWN}"
    emit_string ntasks_per_node "${SLURM_NTASKS_PER_NODE:-$UNKNOWN}"
    emit_string ntasks "${SLURM_NTASKS:-$UNKNOWN}"
    emit_string job_id "${SLURM_JOBID:-$UNKNOWN}"

    if [ "$mode" = "runtime" ]; then
        printf '\n[network]\n'
        emit_string libfabric_version "$(safe_probe libfabric_version_probe)"
    fi
else
    # The runtime collector leaves [network] open so the host can append the
    # physical fabric fields without declaring the TOML table a second time.
    printf '\n'
    emit_physical_network_metadata

    printf '\n[system]\n'
    emit_system_metadata
fi

exit 0
