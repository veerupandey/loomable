"""Amazon Bedrock Agent — run Loomable on any Bedrock model.

USE WHEN: You want to back an agent with Amazon Bedrock (Amazon Nova, Anthropic
Claude, Meta Llama, Mistral, ...) through the unified Converse API.

Authentication uses the standard AWS credential chain — environment variables,
a shared profile, or SSO — so this works for anyone with Bedrock access.

Configure with environment variables (all optional, sensible defaults shown):
  BEDROCK_MODEL   model id or inference-profile id (default: amazon.nova-lite-v1:0)
  AWS_REGION      region to call (default: us-east-1)
  AWS_PROFILE     named profile; unset -> default credential chain

Many newer models are only served via cross-region inference profiles, whose
ids carry a geo prefix: "us." (Americas), "eu." (Europe), "apac." (Asia Pacific).
Pick the one matching your AWS_REGION, e.g.:
  BEDROCK_MODEL=us.amazon.nova-lite-v1:0            AWS_REGION=us-east-1
  BEDROCK_MODEL=eu.anthropic.claude-sonnet-4-6      AWS_REGION=eu-west-1
List what your account can use:
  aws bedrock list-inference-profiles --region <region>

Prereqs:
  pip install "loomable[bedrock]"
  # Grant the caller bedrock:InvokeModel and enable model access for the model
  # in your target region (AWS console -> Bedrock -> Model access).

Run:
  python examples/agents/10_bedrock.py
"""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

from loomable import Agent, tool
from loomable.providers import BedrockProvider
from loomable.providers.errors import PermanentProviderError, TransientProviderError

MODEL = os.environ.get("BEDROCK_MODEL", "amazon.nova-lite-v1:0")
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PROFILE = os.environ.get("AWS_PROFILE")  # None -> default credential chain


@tool
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b


async def main() -> None:
    provider = BedrockProvider(
        MODEL,
        region_name=REGION,
        profile_name=PROFILE,
        max_tokens=512,
    )

    agent = Agent(
        model=provider,
        tools=[add],
        instructions="Use the add tool for arithmetic, then answer in one sentence.",
    )

    print(f"Model: {MODEL}   Region: {REGION}")
    result = await agent.arun("What is 21 + 21? Use the tool, then tell me the total.")
    print(result.output.text())


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (PermanentProviderError, TransientProviderError) as exc:
        # Most first-run failures are AWS setup, not code. Give actionable hints.
        raise SystemExit(
            f"\nBedrock call failed for '{MODEL}' in '{REGION}':\n  {exc}\n\n"
            "Common causes:\n"
            "  - Credentials: run `aws sso login` or set AWS_PROFILE / AWS_* env vars.\n"
            "  - Permissions: the caller needs bedrock:InvokeModel (and\n"
            "    bedrock:InvokeModelWithResponseStream for streaming) on the model ARN.\n"
            "  - Model access: enable the model in the AWS console\n"
            "    (Bedrock -> Model access) for your region.\n"
            "  - Wrong id/region: newer models need a cross-region inference profile\n"
            "    (us./eu./apac. prefix). List them with:\n"
            f"      aws bedrock list-inference-profiles --region {REGION}\n"
        ) from exc
