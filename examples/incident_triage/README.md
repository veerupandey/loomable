# Incident triage agent

The agent reads an incident packet and its runbook through two local tools, then returns a typed handoff. It checks that evidence IDs belong to the packet and that the handoff cites its runbook. The output proposes a mitigation for human review.

## Run a fixture

Install the example dependencies with `pip install -e '.[examples]'`. Set `OPENAI_API_KEY` or `GEMINI_API_KEY` for the model you select.

```bash
python -m examples.incident_triage --model openai:gpt-5.4-mini
python -m examples.incident_triage --model gemini:gemini-3.8-flash --packet examples/incident_triage/fixtures/search.json
```

The included packets cover a payments release regression and a search slowdown during a scheduled reindex. To try another incident, pass a JSON file with the same structure as the fixtures.

## Two-worker staging

Run from the repository root with Python 3.12 and PostgreSQL. Supply
`OPENAI_API_KEY`, `TRIAGE_DATABASE_URL`, and `TRIAGE_ACCESS_TOKEN` through your
deployment's secret configuration. Use a random token of at least 32 bytes.
The database role needs permission to create the checkpoint table and indexes.

```bash
pip install -e '.[examples,serve,postgres]'
uvicorn examples.incident_triage.staging:app --host 127.0.0.1 --port 8874 --workers 2
```

Send the incident packet to `POST /triage` with a UUID in the additional
`request_id` field and an `Authorization: Bearer` header. Staging pins
`gpt-5.4-mini`. Each worker admits two runs, bounds request bodies to 128 KiB,
and applies a 25-second deadline that includes model retries. `GET /health`
reports the ready worker's PID.

```bash
jq --arg id "$(uuidgen)" '. + {request_id: $id}' examples/incident_triage/fixtures/payments.json \
  | curl --fail-with-body http://127.0.0.1:8874/triage \
      -H "Authorization: Bearer $TRIAGE_ACCESS_TOKEN" \
      -H 'Content-Type: application/json' --data-binary @-
```

Use the same UUID and packet when retrying. Workers share a checkpoint
database and return the saved handoff for a completed request. Changing
the packet creates a new session.

| HTTP response | What to do |
|---|---|
| `429`, `503`, or `409` with `Retry-After` | Wait as directed, then retry the same request. |
| `409` without `Retry-After` | Reconcile the interrupted tool batch before retrying. |
| `502` | Investigate the provider or handoff failure before starting a new request. |

### Retention and deployment

Completed sessions are kept for seven days by default. Set
`TRIAGE_RETENTION_SECONDS` and `TRIAGE_RETENTION_INTERVAL_SECONDS` to change the
retention age and sweep interval, respectively. The default interval is 60
seconds per worker and each sweep checks up to 100 sessions. After deletion,
resending a packet starts a new session. Incomplete runs remain available
for reconciliation. Keep the service private or terminate TLS at your staging
ingress. Incident evidence is sent to OpenAI and stored in the checkpoint
database when used by the tools.

## Verify behavior

Install the test dependencies with `pip install -e '.[dev,postgres]'`.

The fixture evaluation checks severity, escalation owner, mitigation,
evidence citations, and the recovery threshold. Set
`LOOMABLE_LIVE_PROVIDER=openai` and `OPENAI_MODEL=gpt-5.4-mini`, or
`LOOMABLE_LIVE_PROVIDER=gemini` and `GEMINI_MODEL=gemini-3.8-flash`, along
with the corresponding API key.

```bash
python -m pytest --live tests/integration/test_incident_triage_live.py
```

To rerun the fault exercise against GPT-5.4 mini, point
`LOOMABLE_TEST_POSTGRES_URL` at a disposable PostgreSQL database and provide
`OPENAI_API_KEY`. The test relay injects HTTP 503 and 429 on selected requests
and forwards healthy calls to OpenAI.

```bash
python -m pytest --live tests/integration/test_incident_triage_staging_live.py tests/integration/test_postgres_retention.py
```

## Measure sustained load

Run the sustained benchmark against an otherwise idle staging database and
service. It submits distinct payments and search sessions, checks each handoff,
and records successful response percentiles, HTTP status counts, and checkpoint
rows and allocated PostgreSQL bytes. It reads `TRIAGE_ACCESS_TOKEN` and
`TRIAGE_DATABASE_URL` from the environment.

```bash
python -m tests.benchmarks.incident_triage_load --url http://127.0.0.1:8874 --seconds 600 --rate 0.75
```
