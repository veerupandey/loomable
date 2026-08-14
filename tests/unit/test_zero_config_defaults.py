"""Tests for zero-configuration defaults on all toolkit classes."""
from pathlib import Path

import pytest

from loomable.toolkits import FileTools, PythonTools, SQLTools, WebSearchTools, PDFTools, URLTools


class TestZeroConfigDefaults:
    def test_web_search_tools_default_provider(self):
        tools = WebSearchTools()
        assert tools._provider == "duckduckgo"
        assert tools._api_key is None
        # Should have one tool registered
        assert len(tools.tools()) == 1
        assert tools.tools()[0].name == "web_search"

    def test_file_tools_defaults_to_cwd(self):
        tools = FileTools()
        assert tools._base_dir == Path.cwd().resolve()
        # Should have 7 tools (edit/glob/grep added for deep-agent parity)
        assert len(tools.tools()) == 7
        names = {t.name for t in tools.tools()}
        assert names == {
            "read_file",
            "write_file",
            "write_json",
            "list_directory",
            "edit_file",
            "glob_files",
            "grep_files",
        }

    def test_python_tools_default_timeout(self):
        tools = PythonTools()
        assert tools._timeout == 30
        assert tools._working_dir is None
        # Should have 2 tools
        assert len(tools.tools()) == 2
        names = {t.name for t in tools.tools()}
        assert names == {"run_python", "run_python_file"}

    def test_sql_tools_defaults_to_read_only(self):
        tools = SQLTools()
        assert tools._read_only is True
        # Should have 3 tools
        assert len(tools.tools()) == 3
        names = {t.name for t in tools.tools()}
        assert names == {"run_sql", "list_tables", "describe_table"}

    def test_pdf_tools_instantiates_no_args(self):
        pytest.importorskip("pypdf")
        tools = PDFTools()
        assert len(tools.tools()) == 2
        names = {t.name for t in tools.tools()}
        assert names == {"read_pdf", "search_pdf"}

    def test_url_tools_default_timeout(self):
        pytest.importorskip("bs4")
        tools = URLTools()
        assert tools._timeout == 30
        assert tools._max_length is None
        assert len(tools.tools()) == 2
        names = {t.name for t in tools.tools()}
        assert names == {"fetch_url", "extract_text"}
