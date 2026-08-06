#!/usr/bin/env bash

set -e

repo_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$repo_dir"

is_windows=false
case "$(uname -s)" in
    MINGW*|MSYS*) is_windows=true ;;
esac

echo
echo "Installing dependencies...."
echo

if $is_windows; then
    if [[ ${MSYSTEM:-} != UCRT64 ]]; then
        echo "Error: Windows builds must be run from the MSYS2 UCRT64 terminal." >&2
        echo "Do not use Git Bash or the plain MSYS terminal; launch 'MSYS2 UCRT64' and rerun ./install.sh." >&2
        exit 1
    fi

    if ! command -v pacman >/dev/null 2>&1; then
        echo "Error: pacman was not found. Install MSYS2 and run this script from its UCRT64 terminal." >&2
        exit 1
    fi

    pacman -S --needed \
        make \
        mingw-w64-ucrt-x86_64-gcc \
        mingw-w64-ucrt-x86_64-openssl \
        mingw-w64-ucrt-x86_64-xxhash \
        mingw-w64-ucrt-x86_64-python \
        mingw-w64-ucrt-x86_64-python-matplotlib \
        mingw-w64-ucrt-x86_64-python-seaborn || {
        echo >&2
        echo "Dependency installation failed. Fully update MSYS2 with 'pacman -Suy'," >&2
        echo "restart the UCRT64 terminal if requested, and run ./install.sh again." >&2
        exit 1
    }

    compiler_target=$(g++ -dumpmachine)
    if [[ $compiler_target != *mingw32* ]]; then
        echo "Error: g++ targets '$compiler_target', not native MinGW-w64." >&2
        echo "Run this script from the MSYS2 UCRT64 terminal." >&2
        exit 1
    fi
else
    if ! command -v apt >/dev/null 2>&1; then
        echo "Error: automatic dependency installation currently supports Debian/Ubuntu and MSYS2 UCRT64." >&2
        exit 1
    fi
    sudo apt update
    sudo apt -y install build-essential libssl-dev libxxhash-dev python3 python3-pip python3-matplotlib python3-seaborn
fi

choice=0
while [[ $choice -eq 0 ]]; do
    echo
    echo "Choose a DedupBench build:"
    echo $'\t1. Unaccelerated (No SIMD)'
    echo $'\t2. SIMD-256 (SSE-128 and AVX-256)'
    echo $'\t3. SIMD-512 (Above and AVX-512)'
    if ! $is_windows; then
        echo $'\t4. ARM NEON-128'
        echo $'\t5. IBM Power VSX-128'
    fi
    read -r -p "Choice? " build_choice

    case "$build_choice" in
        1) build_target=all ;;
        2) build_target=simd_all ;;
        3) build_target=simd512_all ;;
        4) $is_windows && build_target="" || build_target=arm_neon128 ;;
        5) $is_windows && build_target="" || build_target=ibm_altivec128 ;;
        *) build_target="" ;;
    esac

    if [[ -n $build_target ]]; then
        make -C build clean
        make -C build "$build_target"
        if $is_windows && objdump -p build/dedup.exe | grep -Eiq 'DLL Name: (msys-2\.0|cygwin1)\.dll'; then
            echo "Error: dedup.exe unexpectedly depends on a POSIX emulation runtime." >&2
            echo "Ensure the build is using /ucrt64/bin/g++ from the UCRT64 terminal." >&2
            exit 1
        fi
        choice=1
    else
        echo "Choose one of the available build options."
    fi
done

while true; do
    read -r -p "Build completed. Do you want to (re)create the random dataset? Y/N: " yn_choice
    case "$yn_choice" in
        [yY]|[yY][eE][sS]) break ;;
        [nN]|[nN][oO]) exit 0 ;;
        *) echo "Please choose Y for yes or N for no." ;;
    esac
done

echo
echo "Creating random dataset..."

rm -rf "$repo_dir/build/random_dataset"
mkdir -p "$repo_dir/build/random_dataset"
for file_number in 1 2 3; do
    base64 /dev/urandom | head -c 1000000000 > "$repo_dir/build/random_dataset/random_${file_number}.txt"
done

echo
echo "Installation and random dataset creation completed in build/."
