# Settlement Rail v3 — SEV Runbook

## Symptoms
- Partner payouts stuck in RETRYING/FAILED
- `settlement.batch.submit` error spike
- Connector pool wait time > 2s

## Immediate actions
1. Confirm region health (`ap-south-1` settlement + connector)
2. Check latest cert/config change in last 6h
3. If connector saturation: scale workers, drain queue, notify partner
4. Strategic tier: update within **15 minutes**, bridge within **30 minutes**

## Communication
- Use customer-safe language (no internal hostnames)
- State ETA + mitigation clearly
