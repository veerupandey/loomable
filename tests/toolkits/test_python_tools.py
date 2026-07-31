# Feature: built-in-toolkits, Property 11: Python execution captures output
# Feature: built-in-toolkits, Property 12: Python non-zero exit includes return code and stderr
"""Property 11: Python execution captures output.

For any Python code (inline string or file) that prints a deterministic string
to stdout, PythonTools SHALL return that string in the result. For code that
exceeds the timeout, PythonTools SHALL return a timeout error.

Property 12: Python non-zero exit includes return code and stderr.

For any Python code that exits with a non-zero return code, the result SHALL
contain both the numeric return code and the stderr content.

**Validates: Requirements 6.1, 6.2, 6.4, 6.6**
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from loomable.toolkits.python_tools import PythonTools


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: ASCII-safe strings that can be safely printed and captured on any OS.
# Restricted to ASCII letters, digits, and basic punctuation to avoid encoding
# issues with Windows cp1252 codec in subprocess output.
safe_text_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters=" !@#%^&*()-_=+[]{}|;:,.<>?/~",
        max_codepoint=127,
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())  # Ensure non-whitespace-only strings

# Strategy: non-zero exit codes (1-127)
exit_code_st = st.integers(min_value=1, max_value=127)

# Strategy: error messages for stderr (ASCII-safe)
error_message_st = st.text(
    alphabet=st.characters(
        whitelist_categories=("L", "N"),
        whitelist_characters=" !@#%^&*()-_=+[]{}|;:,.<>?/~",
        max_codepoint=127,
    ),
    min_size=1,
    max_size=50,
).filter(lambda s: s.strip())


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


class TestPythonExecutionCapturesOutput:
    """Property 11: Python execution captures output."""

    @settings(max_examples=20, deadline=None)
    @given(output_text=safe_text_st)
    async def test_run_python_captures_stdout(self, output_text: str) -> None:
        """For any deterministic string printed to stdout via run_python,
        the result SHALL contain that string."""
        tools = PythonTools(timeout=10)
        code = f"print({repr(output_text)})"

        result = await tools._run_python(code)

        assert output_text in result

    @settings(max_examples=20, deadline=None)
    @given(output_text=safe_text_st)
    async def test_run_python_file_captures_stdout(self, output_text: str) -> None:
        """For any deterministic string printed to stdout via run_python_file,
        the result SHALL contain that string."""
        tools = PythonTools(timeout=10)
        code = f"print({repr(output_text)})"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            result = await tools._run_python_file(tmp_path)
            assert output_text in result
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    async def test_run_python_timeout_returns_error(self) -> None:
        """For code that exceeds the timeout, PythonTools SHALL return a
        timeout error."""
        tools = PythonTools(timeout=1)
        code = "import time; time.sleep(10)"

        result = await tools._run_python(code)

        assert "timed out" in result.lower() or "timeout" in result.lower()


class TestPythonNonZeroExitIncludesReturnCodeAndStderr:
    """Property 12: Python non-zero exit includes return code and stderr."""

    @settings(max_examples=20, deadline=None)
    @given(error_msg=error_message_st, exit_code=exit_code_st)
    async def test_run_python_nonzero_exit_contains_code_and_stderr(
        self, error_msg: str, exit_code: int
    ) -> None:
        """For any Python code that exits with a non-zero return code,
        the result SHALL contain both the numeric return code and stderr."""
        tools = PythonTools(timeout=10)
        code = (
            f"import sys; "
            f"sys.stderr.write({repr(error_msg)}); "
            f"sys.exit({exit_code})"
        )

        result = await tools._run_python(code)

        # Result must contain the numeric return code
        assert str(exit_code) in result
        # Result must contain the stderr content
        assert error_msg in result

    @settings(max_examples=20, deadline=None)
    @given(error_msg=error_message_st, exit_code=exit_code_st)
    async def test_run_python_file_nonzero_exit_contains_code_and_stderr(
        self, error_msg: str, exit_code: int
    ) -> None:
        """For any Python file that exits with a non-zero return code,
        the result SHALL contain both the numeric return code and stderr."""
        tools = PythonTools(timeout=10)
        code = (
            f"import sys\n"
            f"sys.stderr.write({repr(error_msg)})\n"
            f"sys.exit({exit_code})\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            result = await tools._run_python_file(tmp_path)
            # Result must contain the numeric return code
            assert str(exit_code) in result
            # Result must contain the stderr content
            assert error_msg in result
        finally:
            Path(tmp_path).unlink(missing_ok=True)
