#!/usr/bin/env bash

set -o pipefail

config_suffix=""
dataset=""
silent=false

display_help() {
    echo "Usage: $0 -c <config-suffix> [OPTIONS] <DATASET_DIRECTORY>"
    echo "Options:"
    echo "  -c, --config SUFFIX  Use config_SUFFIX/ and hashes_SUFFIX/"
    echo "  -t SUFFIX            Backward-compatible alias for -c"
    echo "  -s, --silent         Suppress progress messages"
    echo "  -h, --help           Show this help message"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            display_help
            exit 0
            ;;
        -c|--config|-t)
            if [[ $# -lt 2 ]]; then
                echo "Error: $1 requires a config suffix." >&2
                exit 1
            fi
            config_suffix=$2
            shift 2
            ;;
        -s|--silent)
            silent=true
            shift
            ;;
        -*)
            echo "Error: unknown option $1" >&2
            display_help >&2
            exit 1
            ;;
        *)
            if [[ -n $dataset ]]; then
                echo "Error: only one dataset directory may be supplied." >&2
                exit 1
            fi
            dataset=$1
            shift
            ;;
    esac
done

if [[ -z $config_suffix || -z $dataset ]]; then
    echo "Error: both a config suffix and dataset directory are required." >&2
    display_help >&2
    exit 1
fi

if [[ ! -d $dataset ]]; then
    echo "Error: dataset directory not found: $dataset" >&2
    exit 1
fi
dataset=$(cd "$dataset" && pwd -P)

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir"

config_dir="config_${config_suffix}"
hashes_dir="hashes_${config_suffix}"

if [[ ! -d $config_dir ]]; then
    echo "Error: config directory not found: $script_dir/$config_dir" >&2
    exit 1
fi
if [[ ! -x ./dedup.exe || ! -x ./measure-dedup.exe ]]; then
    echo "Error: dedup.exe and measure-dedup.exe must be built in $script_dir." >&2
    exit 1
fi

shopt -s nullglob
config_files=("$config_dir"/*.conf)
if [[ ${#config_files[@]} -eq 0 ]]; then
    echo "Error: no .conf files found in $script_dir/$config_dir" >&2
    exit 1
fi

$silent || echo "Running dedup.exe and measure-dedup.exe for each configuration file"
rm -f results.txt
rm -rf "$hashes_dir"
mkdir -p "$hashes_dir"
echo "Dataset path: $dataset" > results.txt

for config_path in "${config_files[@]}"; do
    config_file=${config_path##*/}
    filename=${config_file%.conf}
    output_file=$(sed -n 's/^[[:space:]]*output_file[[:space:]]*=[[:space:]]*//p' "$config_path" | tail -n 1 | tr -d '\r')
    if [[ -z $output_file ]]; then
        output_file="./${hashes_dir}/${filename}.out"
    fi
    mkdir -p "$(dirname "$output_file")"

    {
        echo "=================="
        echo "$config_file"
    } >> results.txt

    if ! run_output=$(./dedup.exe "$dataset" "$config_path"); then
        echo "Error: dedup.exe failed for $config_file" >&2
        exit 1
    fi
    printf '%s\n' "$run_output" | sed -n '/^Avg/,$p' >> results.txt

    if [[ ! -f $output_file ]]; then
        echo "Error: expected hash output was not created: $output_file" >&2
        exit 1
    fi
    if ! ./measure-dedup.exe "$output_file" >> results.txt; then
        echo "Error: measure-dedup.exe failed for $config_file" >&2
        exit 1
    fi
done

$silent || echo "Finished. Results are stored in $script_dir/results.txt"
