"""Shared provider + constants for the Escalation War Room exam."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures"
OUTPUT = ROOT / "output"


def make_provider():
    gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if gemini_key:
        from loomable.providers.gemini import GeminiProvider

        return GeminiProvider(
            model=os.environ.get("GEMINI_MODEL", "gemini-flash-latest"),
            api_key=gemini_key,
            timeout=180.0,
        )

    zai = os.environ.get("ZAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if zai:
        from loomable.providers.openai import OpenAIProvider

        return OpenAIProvider(
            model=os.environ.get("ZAI_MODEL", os.environ.get("OPENAI_MODEL", "glm-5.2")),
            api_key=zai,
            base_url=os.environ.get(
                "ZAI_BASE_URL",
                os.environ.get("OPENAI_BASE_URL", "https://api.z.ai/api/coding/paas/v4"),
            ),
            timeout=180.0,
        )

    from loomable.providers.openai import AzureOpenAIProvider

    return AzureOpenAIProvider()


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
