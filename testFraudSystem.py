"""Backward-compatible test entry point. Run: pytest tests/ -v"""
import pytest

if __name__ == "__main__":
    pytest.main(["tests/", "-v", "--cov=src", "--cov-report=term-missing"])
