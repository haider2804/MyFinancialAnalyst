"""
Graceful Process Termination Utility (stop_analyst.py)

Uses psutil to reliably discover and terminate running instances
of the Quantitative Trading Analyst and MT5 bridge processes.
"""

import os
import sys
import psutil

PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.pid")


def stop_analyst() -> bool:
    """Finds and terminates active analyst processes."""
    stopped_any = False
    current_pid = os.getpid()

    # 1. Check PID recorded in file
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                target_pid = int(f.read().strip())
            if psutil.pid_exists(target_pid) and target_pid != current_pid:
                try:
                    proc = psutil.Process(target_pid)
                    print(f"[INFO] Terminating active analyst process (PID: {target_pid})...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    print(f"[SUCCESS] Analyst process {target_pid} terminated.")
                    stopped_any = True
                except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                    print(f"[NOTICE] Process {target_pid} already exited: {e}")
        except Exception as e:
            print(f"[WARN] Error reading pid file: {e}")
        finally:
            if os.path.exists(PID_FILE):
                try:
                    os.remove(PID_FILE)
                except Exception:
                    pass

    # 2. Safety scan for any python process running main.py or server.py
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if proc.info["pid"] != current_pid and proc.info["cmdline"]:
                cmd_str = " ".join(proc.info["cmdline"])
                if "main.py" in cmd_str or "server.py" in cmd_str:
                    pid = proc.info["pid"]
                    print(f"[INFO] Terminating active analyst/dashboard instance (PID: {pid})...")
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except psutil.TimeoutExpired:
                        proc.kill()
                    print(f"[SUCCESS] Process {pid} stopped.")
                    stopped_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not stopped_any:
        print("[INFO] No active analyst processes were found running.")

    return stopped_any


if __name__ == "__main__":
    stop_analyst()
