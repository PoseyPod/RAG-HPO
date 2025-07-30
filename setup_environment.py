#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
import ensurepip

STATE_FILE = ".setup_state.json"

def run_command(cmd, **kw):
    """Run a shell command; return True if it succeeds."""
    try:
        subprocess.check_call(cmd, **kw)
        return True
    except subprocess.CalledProcessError:
        return False

def load_state():
    """Load progress flags from disk (or initialize)."""
    if os.path.isfile(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "locked": False,
        "pip_upgraded": False,
        "env_created": False,
        "deps_installed": False,
        "verified": False
    }

def save_state(state):
    """Persist progress flags to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def lock_python_version(target="3.13.2"):
    """Ensure the script runs under exactly Python target, re-exec via pyenv if needed."""
    cur = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if cur == target:
        print(f"🔒 Python {target} detected.")
        return

    print(f"🔄 Switching Python from {cur} to {target} via pyenv…")
    if not shutil.which("pyenv"):
        print("❌ pyenv not found; please install pyenv and re-run.")
        sys.exit(1)

    run_command(["pyenv", "install", "-s", target])
    run_command(["pyenv", "local", target])

    new_py = subprocess.check_output(["pyenv", "which", "python"]).decode().strip()
    print(f"✅ Python {target} installed and set. Re-launching…")
    os.execv(new_py, [new_py] + sys.argv)

def ensure_pip():
    """Bootstrap pip if missing, then upgrade to the latest version."""
    try:
        import pip  # noqa: F401
    except ImportError:
        print("🔧 pip not found; bootstrapping with ensurepip…")
        ensurepip.bootstrap()
    print("🔄 Upgrading pip…")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

def ensure_virtualenv():
    """Decide between built-in venv or pip-install virtualenv."""
    try:
        import venv  # noqa: F401
        return "venv"
    except ImportError:
        print("🔧 venv module unavailable; installing virtualenv…")
        run_command([sys.executable, "-m", "pip", "install", "virtualenv"])
        return "virtualenv"

def create_env(env_name="venv"):
    """Create the virtual environment (or skip if it already exists)."""
    if os.path.isdir(env_name):
        print(f"ℹ️  '{env_name}' already exists; skipping creation.")
        return True

    tool = ensure_virtualenv()
    print(f"🏗  Creating '{env_name}' via {tool}…")
    if tool == "venv":
        ok = run_command([sys.executable, "-m", "venv", env_name])
        if not ok:
            print("⚠️ venv failed; trying virtualenv…")
            ok = run_command([sys.executable, "-m", "virtualenv", env_name])
    else:
        ok = run_command([sys.executable, "-m", "virtualenv", env_name])
        if not ok:
            print("⚠️ virtualenv failed; trying venv…")
            ok = run_command([sys.executable, "-m", "venv", env_name])
    return ok

def install_requirements(req_file="requirements.txt", env_name="venv"):
    """Install all packages from requirements.txt into the venv."""
    if not os.path.isfile(req_file):
        print(f"⚠️ '{req_file}' not found; skipping dependency installation.")
        return False

    pip_exe = os.path.join(env_name, "Scripts" if os.name=="nt" else "bin", "pip")
    if not os.path.isfile(pip_exe):
        print(f"⚠️ pip executable not found at '{pip_exe}'; skipping install.")
        return False

    print(f"📦 Installing dependencies from {req_file}…")
    ok = run_command([pip_exe, "install", "--upgrade", "pip"])
    ok &= run_command([pip_exe, "install", "-r", req_file])
    return ok

def verify_installation(env_name="venv"):
    """List the installed packages in the venv as a sanity check."""
    pip_exe = os.path.join(env_name, "Scripts" if os.name=="nt" else "bin", "pip")
    print("✅ Verifying installed packages:")
    run_command([pip_exe, "list"])

def main():
    state = load_state()

    # 1. Lock to Python 3.13.2
    if not state["locked"]:
        lock_python_version("3.13.2")
        state["locked"] = True
        save_state(state)

    # 2. Bootstrap & upgrade pip
    if not state["pip_upgraded"]:
        ensure_pip()
        state["pip_upgraded"] = True
        save_state(state)

    # 3. Create or resume virtual environment
    if not state["env_created"]:
        if create_env("venv"):
            state["env_created"] = True
        else:
            print("⚠️ Could not create virtual environment; retry on next run.")
        save_state(state)

    # 4. Install requirements
    if state["env_created"] and not state["deps_installed"]:
        if install_requirements("requirements.txt", "venv"):
            state["deps_installed"] = True
        else:
            print("⚠️ Dependency installation incomplete; retry on next run.")
        save_state(state)

    # 5. Verification
    if state["deps_installed"] and not state["verified"]:
        verify_installation("venv")
        state["verified"] = True
        save_state(state)

    # 6. Completion summary
    print("\n🎉 Setup complete (or resumed)!")
    print("To activate the environment:")
    if os.name=="nt":
        print(r"  .\venv\Scripts\activate")
    else:
        print("  source venv/bin/activate")

    # 7. Clean up state file when everything is done
    if all(state.values()):
        os.remove(STATE_FILE)

if __name__ == "__main__":
    main()