"""
pod_monitor.py — background GPU-utilization watchdog for RunPod scripts.

Shared across every PCL sub-project (chat1_protocol, pcl-legacy2, deadpcl, ...).
Import from any script regardless of which sub-project directory it lives in;
this file sits at the repo root next to data/ and venv/ for exactly that reason.

What it does: starts a daemon thread that polls `nvidia-smi` every few seconds
for the life of the process. When it sees a SUSTAINED (not momentary) mismatch
between what you're paying for and what's actually running, it prints a loud,
impossible-to-miss banner:

  * GPU sits near-idle for IDLE_MIN minutes straight -> you're almost
    certainly downloading/parsing/preprocessing data, not training, and
    you're paying GPU-pod price for CPU work. Tells you to switch DOWN to
    a cheap pod.
  * After that, GPU utilization comes back up and STAYS up for ACTIVE_MIN
    minutes -> real training resumed. Tells you to switch BACK UP to the
    GPU pod, in case you moved down for the data-load phase and forgot.

Both directions matter (that's the point): it should stop you overspending
on an idle expensive pod, AND stop you wasting wall-clock time training on
an underpowered one.

Usage — one line, near the top of main(), before the expensive part starts:

    from pod_monitor import watch_pod
    watch_pod()   # non-blocking, spawns a daemon thread, returns immediately

Safe everywhere: if `nvidia-smi` isn't found (local dev machine, CPU-only
box, sandboxed CI), it disables itself silently after one failed probe and
costs nothing. There is no harm in leaving the call in scripts that never
run on a GPU pod at all.

Tunable via env vars if the default cadence doesn't fit a given run:
    POD_MONITOR_POLL_SEC    (default 15)   how often to poll nvidia-smi
    POD_MONITOR_IDLE_MIN    (default 5)    sustained idle minutes before alert
    POD_MONITOR_ACTIVE_MIN  (default 2)    sustained active minutes before alert
    POD_MONITOR_REMINDER_MIN (default 10)  re-print the banner if state persists
    POD_MONITOR_UTIL_PCT    (default 10)   GPU% threshold separating idle/active
"""
import os
import shutil
import subprocess
import threading
import time

POLL_SEC = float(os.environ.get("POD_MONITOR_POLL_SEC", 15))
IDLE_MIN = float(os.environ.get("POD_MONITOR_IDLE_MIN", 5))
ACTIVE_MIN = float(os.environ.get("POD_MONITOR_ACTIVE_MIN", 2))
REMINDER_MIN = float(os.environ.get("POD_MONITOR_REMINDER_MIN", 10))
UTIL_PCT = float(os.environ.get("POD_MONITOR_UTIL_PCT", 10))

_SWITCH_DOWN_MSG = """
{bar}
SWITCH PODS NOW — GPU HAS BEEN IDLE FOR {mins:.0f}+ MIN (UTIL < {util:.0f}%).
THIS LOOKS LIKE DATA LOADING / PARSING / PREPROCESSING, NOT TRAINING.
YOU ARE PAYING GPU-POD PRICE FOR CPU-ONLY WORK — SWITCH TO A CHEAP POD.
{bar}
"""

_SWITCH_UP_MSG = """
{bar}
SWITCH BACK TO THE GPU POD NOW — SUSTAINED GPU ACTIVITY FOR {mins:.0f}+ MIN
(UTIL >= {util:.0f}%). REAL TRAINING HAS RESUMED.
IF YOU MOVED DOWN TO A CHEAP POD FOR THE DATA-LOAD PHASE, MOVE BACK NOW.
{bar}
"""

_BAR = "=" * 70


def _read_gpu_util():
    """Max utilization.gpu (%) across all visible GPUs, or None if unreadable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    try:
        return max(float(line.strip()) for line in out.stdout.strip().splitlines() if line.strip())
    except ValueError:
        return None


def _monitor_loop(poll_sec, idle_min, active_min, reminder_min, util_pct, verbose):
    idle_streak_start = None
    active_streak_start = None
    state = None            # None -> "idle" -> "active" -> "idle" -> ...
    last_alert_ts = 0.0
    consecutive_read_failures = 0

    while True:
        time.sleep(poll_sec)
        util = _read_gpu_util()
        now = time.time()

        if util is None:
            consecutive_read_failures += 1
            if consecutive_read_failures >= 3:
                if verbose:
                    print("[pod_monitor] nvidia-smi unavailable — stopping monitor.", flush=True)
                return
            continue
        consecutive_read_failures = 0

        if util >= util_pct:
            active_streak_start = active_streak_start or now
            idle_streak_start = None
        else:
            idle_streak_start = idle_streak_start or now
            active_streak_start = None

        active_dur = (now - active_streak_start) if active_streak_start else 0.0
        idle_dur = (now - idle_streak_start) if idle_streak_start else 0.0

        if active_dur >= active_min * 60 and state != "active":
            print(_SWITCH_UP_MSG.format(bar=_BAR, mins=active_min, util=util_pct), flush=True)
            state, last_alert_ts = "active", now
        elif idle_dur >= idle_min * 60 and state != "idle":
            print(_SWITCH_DOWN_MSG.format(bar=_BAR, mins=idle_min, util=util_pct), flush=True)
            state, last_alert_ts = "idle", now
        elif state is not None and (now - last_alert_ts) >= reminder_min * 60:
            msg = _SWITCH_UP_MSG if state == "active" else _SWITCH_DOWN_MSG
            elapsed_min = (active_dur if state == "active" else idle_dur) / 60
            print(msg.format(bar=_BAR, mins=elapsed_min, util=util_pct), flush=True)
            last_alert_ts = now


_started = False
_lock = threading.Lock()


def watch_pod(poll_sec=None, idle_min=None, active_min=None, reminder_min=None,
              util_pct=None, verbose=False):
    """Start the background pod-switch watchdog. Idempotent: calling this more
    than once in the same process is a no-op after the first call. Returns
    immediately; does nothing if nvidia-smi isn't available (e.g. local dev)."""
    global _started
    with _lock:
        if _started:
            return
        if shutil.which("nvidia-smi") is None:
            if verbose:
                print("[pod_monitor] no nvidia-smi on this machine — monitor disabled.", flush=True)
            _started = True
            return
        t = threading.Thread(
            target=_monitor_loop,
            args=(poll_sec or POLL_SEC, idle_min or IDLE_MIN, active_min or ACTIVE_MIN,
                  reminder_min or REMINDER_MIN, util_pct or UTIL_PCT, verbose),
            daemon=True,
            name="pod-monitor",
        )
        t.start()
        _started = True
        if verbose:
            print(f"[pod_monitor] watching GPU util every {poll_sec or POLL_SEC:.0f}s "
                  f"(idle>={idle_min or IDLE_MIN:.0f}min / active>={active_min or ACTIVE_MIN:.0f}min "
                  f"triggers an alert).", flush=True)


if __name__ == "__main__":
    # Manual smoke test: run this file directly on a GPU pod to watch it fire.
    watch_pod(verbose=True)
    print("pod_monitor running standalone — Ctrl+C to stop.")
    while True:
        time.sleep(60)
