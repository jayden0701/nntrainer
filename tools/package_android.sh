#!/usr/bin/env bash

set -e

TARGET=$1
[ -z $1 ] && TARGET=$(pwd)
echo $TARGET


if [ ! -d $TARGET ]; then
    if [[ $1 == -D* ]] || [[ $1 == --arm-arch* ]]; then
	TARGET=$(pwd)
	echo $TARGET
    else
	echo $TARGET is not a directory. please put project root of nntrainer
	exit 1
    fi
fi

ANDROID_NDK="${ANDROID_NDK:-${NDK_ROOT:-}}"
if [[ -z "$ANDROID_NDK" ]]; then
  ndk_build_path="$(command -v ndk-build 2>/dev/null || \
    command -v ndk-build.cmd 2>/dev/null || true)"
  if [[ -n "$ndk_build_path" ]]; then
    ANDROID_NDK="$(dirname "$ndk_build_path")"
  fi
fi
if [[ -z "$ANDROID_NDK" || ! -d "$ANDROID_NDK" ]]; then
  echo "Error: cannot determine the Android NDK" >&2
  exit 1
fi
ANDROID_NDK="$(cd "$ANDROID_NDK" && pwd -P)"
if [[ ! -f "$ANDROID_NDK/ndk-build" && \
      ! -f "$ANDROID_NDK/ndk-build.cmd" ]]; then
  echo "Error: invalid Android NDK root: $ANDROID_NDK" >&2
  exit 1
fi
NDK_ROOT="$ANDROID_NDK"
export ANDROID_NDK NDK_ROOT
export PATH="$ANDROID_NDK:$PATH"

pushd $TARGET

filtered_args=()
arm_arch=""

record_android_ndk_fingerprint() {
  local ndk_root="$ANDROID_NDK"
  local source_properties="$ndk_root/source.properties"
  local ndk_revision=""
  if [[ -f "$source_properties" ]]; then
    ndk_revision="$(
      awk '
        /^[[:space:]]*Pkg\.Revision[[:space:]]*=/ {
          revision = substr($0, index($0, "=") + 1)
          gsub(/^[[:space:]]+|[[:space:]\r]+$/, "", revision)
          print revision
          exit
        }
      ' "$source_properties"
    )"
  fi
  if [[ -z "$ndk_revision" ]]; then
    echo "Error: Pkg.Revision is missing from $source_properties" >&2
    return 1
  fi

  local fingerprint_path="$ndk_root"
  case "$(uname -s)" in
    CYGWIN*|MINGW*|MSYS*)
      if command -v cygpath >/dev/null 2>&1; then
        fingerprint_path="$(cygpath -m "$ndk_root")"
      fi
      ;;
  esac
  printf 'revision=%s\npath=%s\n' "$ndk_revision" "$fingerprint_path" \
    > android_build_result/nntrainer-ndk-fingerprint
}

for arg in "$@"; do
    if [[ $arg == -D* ]]; then
	filtered_args+=("$arg")
    fi
    # Handle --arm-arch=<version> argument
    if [[ $arg == --arm-arch=* ]]; then
        arm_arch="${arg#*=}"
    fi
done

# If --arm-arch specified, read configuration from JSON file
if [[ -z "$arm_arch" ]]; then
    arm_arch="armv8.2-a"
fi

if [[ -n "$arm_arch" ]]; then
    # Convert dots to dashes for filename (e.g., armv8.2-a -> armv8-2-a)
    arch_filename=$(echo "$arm_arch" | sed 's/\./-/g')
    json_file="${TARGET}/tools/cross/android_${arch_filename}.json"
    if [[ -f "$json_file" ]]; then
        echo "Using ARM architecture config from: $json_file"
        # Read values from JSON using Python (single invocation, portable, no jq dependency)
        eval "$(python3 -c "
import json, sys
try:
    data = json.load(open('$json_file'))
    print(f'enable_fp16={data.get(\"enable_fp16\", \"True\")}')
    print(f'arm_march=\"{data.get(\"arm_march\", \"\")}\"')
except Exception as e:
    print(f'echo \"Error reading JSON: {e}\" >&2', file=sys.stderr)
    sys.exit(1)
")"
        # Add arm-arch and arm-march to meson args
        filtered_args+=("-Darm-arch=${arm_arch}")
        filtered_args+=("-Darm-march=-march=${arm_march}")
        # Handle enable_fp16 based on JSON boolean
        if [[ "$enable_fp16" == "False" ]]; then
            filtered_args+=("-Denable-fp16=false")
        fi
    else
        echo "Warning: JSON config file not found: $json_file"
        echo "Available configurations:"
        ls -1 "${TARGET}/tools/cross/"*.json 2>/dev/null || echo "  No configurations found in tools/cross/"
    fi
fi


if [ ! -f builddir/build.ninja ]; then
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
  meson builddir -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Dnntr-num-threads=4 -Dhgemm-experimental-kernel=false ${filtered_args[@]}
else
  echo "warning: $TARGET/builddir has already been taken, this script tries to reconfigure and try building"
  pushd builddir
    #default value of openblas num threads is 1 for android
    #enable-tflite-interpreter=false is just temporally until ci system is stabel  
    #enable-opencl=true will compile OpenCL related changes or remove this option to exclude OpenCL compilations.
    meson configure -Dplatform=android -Dopenblas-num-threads=1 -Denable-tflite-interpreter=false -Denable-tflite-backbone=false -Denable-fp16=true -Dnntr-num-threads=4 -Dhgemm-experimental-kernel=false ${filtered_args[@]}
    meson --wipe
  popd
fi

pushd builddir
ninja install
record_android_ndk_fingerprint

tar -czvf $TARGET/nntrainer_for_android.tar.gz --directory=android_build_result .

popd
popd

