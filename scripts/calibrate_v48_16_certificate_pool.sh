#!/usr/bin/env bash
set -euo pipefail
# Compatibility wrapper around the corrected dedicated certificate controller.
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/calibrate_v48_14_certificate_pool.sh" "$@"
