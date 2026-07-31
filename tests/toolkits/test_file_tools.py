# Feature: built-in-toolkits, Property 5: File write/read round-trip
# Feature: built-in-toolkits, Property 6: list_directory completeness
# Feature: built-in-toolkits, Property 7: Path traversal rejection
# Feature: built-in-toolkits, Property 8: Missing file returns error
"""Property-based tests for FileTools.

Property 5: For any valid file path (within base_dir) and any string content,
writing via write_file(path, content) followed by read_file(path) SHALL return
the original content.

Property 6: For any set of files and directories created within a directory,
list_directory(path) SHALL return all their names.

Property 7: For any path that resolves outside the configured base_dir (via ../
traversal or absolute path outside base_dir), FileTools operations SHALL return
an error and not access the target.

Property 8: For any path that does not exist on the filesystem, read_file(path)
SHALL return a result containing a "not found" error message rather than raising
an exception.

**Validates: Requirements 3.1, 3.3, 3.4, 3.5, 3.7**
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from loomable.toolkits.file_tools import FileTools


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy: simple filenames that are valid on the filesystem
# Avoid special characters, null bytes, reserved names, and extensions that
# trigger format auto-detection (json, csv) which transforms content on read.
_safe_filename_chars = st.sampled_from(
    "abcdefghijklmnopqrstuvwxyz0123456789_-"
)
safe_filenames = st.text(
    alphabet=_safe_filename_chars, min_size=1, max_size=20
).map(lambda s: s + ".txt")  # .txt extension ensures no format transformation

# Strategy: arbitrary text content for file round-trip
# Exclude lone \r characters because Python's text-mode I/O normalizes
# line endings (\r -> \n on read), which is expected platform behavior.
# Exclude surrogates (\ud800-\udfff) which can't be encoded to UTF-8.
file_content = st.text(
    alphabet=st.characters(
        blacklist_characters="\r",
        blacklist_categories=("Cs",),  # Exclude surrogates
    ),
    min_size=0,
    max_size=500,
)

# Strategy: lists of unique filenames for directory listing tests
filename_lists = st.lists(
    st.text(alphabet=_safe_filename_chars, min_size=1, max_size=15),
    min_size=1,
    max_size=10,
    unique=True,
)

# Strategy: path traversal attempts
traversal_paths = st.one_of(
    # Relative traversal with ../
    st.text(alphabet=_safe_filename_chars, min_size=1, max_size=10).map(
        lambda s: f"../{s}"
    ),
    # Double traversal
    st.text(alphabet=_safe_filename_chars, min_size=1, max_size=10).map(
        lambda s: f"../../{s}"
    ),
    # Traversal into specific system paths
    st.just("../../../etc/passwd"),
    st.just("../../windows/system32/config"),
    # Just parent directory reference
    st.just(".."),
)

# Strategy: random filenames that are unlikely to exist
nonexistent_filenames = st.text(
    alphabet=_safe_filename_chars, min_size=5, max_size=30
).map(lambda s: f"nonexistent_{s}.txt")


# ---------------------------------------------------------------------------
# Property 5: File write/read round-trip
# ---------------------------------------------------------------------------


class TestFileWriteReadRoundTrip:
    """Property 5: write_file followed by read_file returns original content."""

    @settings(max_examples=20, deadline=None)
    @given(filename=safe_filenames, content=file_content)
    @pytest.mark.asyncio
    async def test_write_then_read_returns_original_content(
        self, filename: str, content: str
    ) -> None:
        """For any valid path and content, write then read is identity."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # Write the file
            write_result = await tools._write_file(filename, content)
            assert "Successfully wrote" in write_result

            # Read back the file
            read_result = await tools._read_file(filename)

            # Round-trip: content should be identical
            assert read_result == content

    @settings(max_examples=20, deadline=None)
    @given(content=file_content)
    @pytest.mark.asyncio
    async def test_write_read_nested_path(self, content: str) -> None:
        """Write/read works for files in nested subdirectories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)
            nested_path = "subdir/nested/file.txt"

            write_result = await tools._write_file(nested_path, content)
            assert "Successfully wrote" in write_result

            read_result = await tools._read_file(nested_path)
            assert read_result == content


# ---------------------------------------------------------------------------
# Property 6: list_directory completeness
# ---------------------------------------------------------------------------


class TestListDirectoryCompleteness:
    """Property 6: list_directory returns all file and directory names."""

    @settings(max_examples=20, deadline=None)
    @given(filenames=filename_lists)
    @pytest.mark.asyncio
    async def test_list_directory_returns_all_files(
        self, filenames: list[str]
    ) -> None:
        """For any set of created files, list_directory includes all names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # Create all files
            for name in filenames:
                Path(tmp_dir, name).write_text(
                    f"content of {name}", encoding="utf-8"
                )

            # List the directory
            result = await tools._list_directory(".")

            # Every created file should appear in the listing
            for name in filenames:
                assert name in result

    @settings(max_examples=20, deadline=None)
    @given(dirnames=filename_lists)
    @pytest.mark.asyncio
    async def test_list_directory_returns_all_directories(
        self, dirnames: list[str]
    ) -> None:
        """For any set of created directories, list_directory includes all names."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # Create all directories
            for name in dirnames:
                Path(tmp_dir, name).mkdir(exist_ok=True)

            # List the directory
            result = await tools._list_directory(".")

            # Every created directory should appear in the listing
            for name in dirnames:
                assert name in result


# ---------------------------------------------------------------------------
# Property 7: Path traversal rejection
# ---------------------------------------------------------------------------


class TestPathTraversalRejection:
    """Property 7: Paths escaping base_dir are rejected with an error."""

    @settings(max_examples=20, deadline=None)
    @given(traversal_path=traversal_paths)
    @pytest.mark.asyncio
    async def test_read_file_rejects_traversal(
        self, traversal_path: str
    ) -> None:
        """read_file with a traversal path returns an error string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            result = await tools._read_file(traversal_path)

            assert "Error" in result
            assert "traversal" in result.lower() or "not allowed" in result.lower()

    @settings(max_examples=20, deadline=None)
    @given(traversal_path=traversal_paths)
    @pytest.mark.asyncio
    async def test_write_file_rejects_traversal(
        self, traversal_path: str
    ) -> None:
        """write_file with a traversal path returns an error string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            result = await tools._write_file(traversal_path, "malicious content")

            assert "Error" in result
            assert "traversal" in result.lower() or "not allowed" in result.lower()

    @settings(max_examples=20, deadline=None)
    @given(traversal_path=traversal_paths)
    @pytest.mark.asyncio
    async def test_list_directory_rejects_traversal(
        self, traversal_path: str
    ) -> None:
        """list_directory with a traversal path returns an error string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            result = await tools._list_directory(traversal_path)

            assert "Error" in result
            assert "traversal" in result.lower() or "not allowed" in result.lower()

    @settings(max_examples=20, deadline=None)
    @given(
        filename=st.text(
            alphabet=_safe_filename_chars, min_size=1, max_size=10
        )
    )
    @pytest.mark.asyncio
    async def test_absolute_path_outside_base_rejected(
        self, filename: str
    ) -> None:
        """Absolute paths outside base_dir are rejected."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # Construct an absolute path outside base_dir
            if os.name == "nt":
                outside_path = f"C:\\Windows\\Temp\\{filename}"
            else:
                outside_path = f"/tmp/{filename}"

            # Only test if it's truly outside base_dir
            resolved = Path(outside_path).resolve()
            assume(not str(resolved).startswith(str(Path(tmp_dir).resolve())))

            result = await tools._read_file(outside_path)

            assert "Error" in result


# ---------------------------------------------------------------------------
# Property 8: Missing file returns error
# ---------------------------------------------------------------------------


class TestMissingFileReturnsError:
    """Property 8: read_file on non-existent path returns error, not exception."""

    @settings(max_examples=20, deadline=None)
    @given(filename=nonexistent_filenames)
    @pytest.mark.asyncio
    async def test_read_nonexistent_file_returns_error_message(
        self, filename: str
    ) -> None:
        """read_file for a non-existent file returns an error string."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # Ensure the file truly doesn't exist
            assert not Path(tmp_dir, filename).exists()

            result = await tools._read_file(filename)

            # Should return an error message, not raise an exception
            assert isinstance(result, str)
            assert "Error" in result
            assert "not found" in result.lower()

    @settings(max_examples=20, deadline=None)
    @given(filename=nonexistent_filenames)
    @pytest.mark.asyncio
    async def test_read_nonexistent_does_not_raise(
        self, filename: str
    ) -> None:
        """read_file never raises an exception for missing files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tools = FileTools(base_dir=tmp_dir)

            # This should NOT raise - it should return an error string
            try:
                result = await tools._read_file(filename)
                # Verify we got a string result (not an exception)
                assert isinstance(result, str)
            except FileNotFoundError:
                pytest.fail(
                    "read_file raised FileNotFoundError instead of returning error string"
                )
            except Exception as e:
                pytest.fail(
                    f"read_file raised {type(e).__name__} instead of returning error string"
                )
