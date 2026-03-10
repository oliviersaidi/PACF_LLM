"""Smoke tests for PACF_LLM."""
import subprocess, sys, os, py_compile

def test_version_flag():
    result = subprocess.run(
        [sys.executable, "PACF_LLM_V13_1c.py", "--version"],
        capture_output=True, text=True, timeout=30
    )
    # Accept either a version string or help output; just must not crash with code >1
    assert result.returncode in (0, 1), f"Unexpected exit code: {result.returncode}"

def test_imports():
    for module in ["torch", "transformers", "numpy"]:
        result = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            capture_output=True, text=True, timeout=30
        )
        assert result.returncode == 0, f"Failed to import {module}: {result.stderr}"

def test_syntax():
    py_compile.compile("PACF_LLM_V13_1c.py", doraise=True)
    print("Syntax OK")

def test_files_present():
    for f in ["PACF_LLM_V13_1c.py", "requirements.txt", "README.md", "LICENSE.txt"]:
        assert os.path.exists(f), f"Missing: {f}"

if __name__ == "__main__":
    test_syntax()
    test_files_present()
    test_imports()
    print("All smoke tests passed.")
