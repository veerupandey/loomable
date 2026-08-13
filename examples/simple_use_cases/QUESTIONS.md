# Simple Loomable agent — question bank

Starter questions for demos and smoke tests. Keep them short.

## A. Plain Q&A (no tools required)

1. What is the capital of India? Answer in one sentence.
2. Explain GST in India like I’m new to it.
3. Summarize the difference between Lok Sabha and Rajya Sabha.
4. What is matcha? Keep it under 5 bullets.
5. Translate to Hindi: “The meeting is at 3pm tomorrow.”

## B. Current affairs / web search

1. What is the news in India today?
2. How is the Modi government doing — economy, politics, public mood?
3. Top 5 business headlines in India this week.
4. What changed recently in India’s digital payments / UPI story?
5. Bring me info about **kageha mactha** (try spelling variants if needed).

## C. Structured output

1. Return a JSON brief: `{headline, summary, mood, key_points[], sources[]}` on Modi government performance.
2. Return a topic card for kageha/matcha: `{name, what_it_is, origin_or_context, why_it_matters, confidence}`.
3. Extract action items from a meeting note into `{owner, task, due}` objects.
4. Classify a customer email: `{intent, urgency, sentiment, suggested_reply}`.
5. Score a product idea 1–5 on `{clarity, demand, feasibility, risk}` with one-line reasons.

## D. Tool calling

1. What is `(12 + 8) * 3`? Use calculator tools.
2. Look up a canned `city_fact` for Delhi and Mumbai.
3. Convert 2500 INR to USD with a rates tool (stub or live).
4. List files in a folder, then read `readme.md`.
5. Search the web for “India manufacturing AI scheduling” and cite 3 links.

## E. Document I/O (md / pdf / pptx)

1. Read `notes.md` and summarize in 5 bullets.
2. Read a PDF market snapshot and extract buyer pains.
3. Read a PPT ops update and list risks + next actions.
4. Combine md + pdf + ppt into one `summary.md` written back to disk.
5. Turn a markdown brief into slide bullet points (output markdown that mimics slides).

## F. More simple product use cases (ideas)

| Use case | Example question |
|----------|------------------|
| FAQ bot | “What is your refund policy?” (with knowledge docs) |
| Lead qualifier | “From this form reply, is the lead hot/warm/cold?” |
| Meeting notes | “Turn this transcript into decisions + owners.” |
| Support triage | “Which queue should get this ticket?” |
| Price explainer | “Explain our Starter vs Pro plan simply.” |
| Competitor sniff | “Who competes with us in Indian MES/APS?” |
| Hiring helper | “Rewrite this JD for a backend engineer in Bengaluru.” |
| Travel lite | “Plan a 2-day Delhi work trip checklist.” |
| Compliance lite | “What does SOC2 mean for a SaaS startup?” |
| Content draft | “Write a 120-word LinkedIn post about factory AI.” |

## Suggested demo order

1. `05_tool_calling.py` — prove tools fire  
2. `01_news_india.py` — web Q&A  
3. `02_research_topic.py` — unfamiliar topic  
4. `03_structured_brief.py` — typed JSON  
5. `04_document_io.py` — md/pdf/pptx in and markdown out  
