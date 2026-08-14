"""Shared provider + constants for the Escalation War Room exam."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

# Repo-root .env (examples may be run from this subfolder)
load_dotenv(Path(__file__).resolve().parents[2] / ".env")
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _provider import make_provider, require_provider  # noqa: E402, F401

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
OUTPUT = ROOT / "output"


ESCALATION_EMAIL = """\
From: priya.nair@bharatnova.bank
To: support@acmepay.io
Subject: URGENT — UPI settlement batches stuck since 18:40 IST (prod)

AcmePay team,

Our BharatNova production settlement window is failing. Approximately 42,000
merchant payouts for the 18:00 IST cut-over are in RETRYING / FAILED state in
your console. Our treasury desk is blocked from closing books.

Impact:
- Partner: BharatNova Bank (contract tier: Strategic, ARR ~$2.4M)
- Product: AcmePay Settlement Rail v3 (region: ap-south-1)
- First detected: 18:42 IST by our NOC
- Customer-visible: Yes — merchants seeing "payout delayed" in BharatNova app
- Our ticket with you: already opened as INC-88421 (P1)

Please confirm:
1) Is this an AcmePay-side outage or a bank connector issue?
2) Are we in SLA breach for Strategic partners?
3) Who is the war-room bridge / next update ETA?

— Priya Nair, Head of Digital Payments Ops, BharatNova
"""
