# Simple use cases

Minimal Loomable agents for everyday questions.

| File | What it shows |
|------|----------------|
| `01_news_india.py` | Web-search Q&A: India news + Modi government |
| `02_research_topic.py` | Research an unfamiliar topic (`kageha mactha`) |
| `03_structured_brief.py` | Structured JSON via `response_model` |
| `04_document_io.py` | Markdown + PDF + PPTX input → markdown output |
| `05_tool_calling.py` | Verify tool calls actually happen |
| `QUESTIONS.md` | Larger question bank for demos |

## Setup

```bash
# Provider (Z.AI example)
export ZAI_API_KEY="..."
export ZAI_BASE_URL="https://api.z.ai/api/coding/paas/v4"
export ZAI_MODEL="glm-5.2"

# Or Azure OpenAI via .env (AZURE_OPENAI_*)

pip install -e ".[web,pdf,ppt]"
# or: pip install duckduckgo-search pypdf python-pptx
```

## Run

```bash
python examples/simple_use_cases/05_tool_calling.py
python examples/simple_use_cases/01_news_india.py
python examples/simple_use_cases/02_research_topic.py
python examples/simple_use_cases/03_structured_brief.py
python examples/simple_use_cases/04_document_io.py
```

Run from `examples/simple_use_cases/` or with `PYTHONPATH` so `_provider.py` imports resolve:

```bash
cd examples/simple_use_cases && python 01_news_india.py
```
