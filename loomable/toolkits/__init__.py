"""loomable.toolkits - Built-in toolkit classes for the loomable agent framework.

This package provides production-ready toolkits that integrate with the
``Agent(tools=[...])`` API. Each toolkit groups related FunctionTool instances.

Usage::

    from loomable.toolkits import Toolkit, FileTools, SQLTools, WebSearchTools

    agent = Agent(
        model=...,
        tools=[FileTools(), SQLTools(read_only=True)],
    )
"""

from loomable.toolkits._base import Toolkit
from loomable.toolkits.citation_tools import CitationStore, CitationTools
from loomable.toolkits.file_tools import FileTools
from loomable.toolkits.image_tools import ImageTools
from loomable.toolkits.python_tools import PythonTools
from loomable.toolkits.sql_tools import SQLTools
from loomable.toolkits.todo_tools import TodoStore, TodoTools
from loomable.toolkits.web_search import WebSearchTools
from loomable.toolkits.workspace_tools import WorkspaceStore, WorkspaceTools

# Conditional imports for toolkits with optional dependencies
try:
    from loomable.toolkits.pdf_tools import PDFTools
except ImportError:  # pragma: no cover
    PDFTools = None  # type: ignore[misc, assignment]

try:
    from loomable.toolkits.ppt_tools import PPTTools
except ImportError:  # pragma: no cover
    PPTTools = None  # type: ignore[misc, assignment]

try:
    from loomable.toolkits.url_tools import URLTools
except ImportError:  # pragma: no cover
    URLTools = None  # type: ignore[misc, assignment]

__all__ = [
    "Toolkit",
    "CitationTools",
    "CitationStore",
    "FileTools",
    "ImageTools",
    "PDFTools",
    "PPTTools",
    "PythonTools",
    "SQLTools",
    "URLTools",
    "WebSearchTools",
    "TodoTools",
    "TodoStore",
    "WorkspaceTools",
    "WorkspaceStore",
]
