#!/bin/bash
# Diff two ib_snapshot.sh JSONs on their per-port /sys counter values. Any delta
# on error-class counters (symbol_error, link_downed, port_xmit_discards,
# port_rcv_errors, port_rcv_remote_physical_errors, etc.) during a benchmark
# window is a signal that the fabric hiccupped under load.
#
# Usage:
#   fabric/counter_delta.sh <before.json> <after.json>
#
# Exit codes: 0 if no error-class counters advanced; 1 if any did.

set -euo pipefail

BEFORE="${1:?usage: counter_delta.sh <before.json> <after.json>}"
AFTER="${2:?after.json required}"

python3 - "${BEFORE}" "${AFTER}" <<'PY'
import json
import sys
from pathlib import Path

# Counter names whose non-zero delta indicates a real problem. Benign traffic
# counters (port_xmit_data, port_rcv_data) are skipped — we expect them to move.
_ERROR_COUNTERS = {
    "symbol_error",
    "link_downed",
    "port_rcv_errors",
    "port_rcv_remote_physical_errors",
    "port_rcv_switch_relay_errors",
    "port_xmit_discards",
    "port_xmit_constraint_errors",
    "port_rcv_constraint_errors",
    "local_link_integrity_errors",
    "excessive_buffer_overrun_errors",
    "VL15_dropped",
    "port_xmit_wait",
    "port_rcv_switch_relay_errors",
    "link_error_recovery",
}


def _counters_by_hca_port(snap):
    out = {}
    for hca in snap["ib"]["hcas"]:
        for port in hca["ports"]:
            key = f"{hca['name']}/port{port['port']}"
            out[key] = port.get("counters", {})
    return out


def main():
    before = json.loads(Path(sys.argv[1]).read_text())
    after = json.loads(Path(sys.argv[2]).read_text())
    a = _counters_by_hca_port(before)
    b = _counters_by_hca_port(after)

    print(f"{'port':<20} {'counter':<40} {'before':>12} {'after':>12} {'delta':>12}")
    print("-" * 100)

    any_error = False
    for port in sorted(set(a) | set(b)):
        a_c = a.get(port, {})
        b_c = b.get(port, {})
        for counter in sorted(set(a_c) | set(b_c)):
            before_v = a_c.get(counter, 0) or 0
            after_v = b_c.get(counter, 0) or 0
            delta = after_v - before_v
            if delta == 0:
                continue
            is_err = counter in _ERROR_COUNTERS
            tag = " ERROR" if is_err and delta > 0 else ""
            print(f"{port:<20} {counter:<40} {before_v:>12} {after_v:>12} {delta:>+12}{tag}")
            if is_err and delta > 0:
                any_error = True

    if any_error:
        print("\n[counter_delta] FAIL: error-class counters advanced during the window")
        sys.exit(1)
    print("\n[counter_delta] OK: no error-class counter deltas")
    sys.exit(0)


main()
PY
