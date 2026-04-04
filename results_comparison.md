# ID3QNE Sepsis OpenEnv Results

| Policy | Mean Score | Density | Steps | Safety |
|--------|------------|---------|-------|--------|
| Heuristic | 0.9867 | 1.00 | 9.7 | 100% |
| LLM (gpt-4o-mini) | 0.9867 | 1.00 | 9.7 | 100% |
| ID3QNE | 0.9867 | 1.00 | 9.7 | 100% |

## Statistical Validation

- LLM 10-episode mean score: `0.9867`
- LLM 10-episode score std across episode means: `0.0`
- LLM global reward density: `1.0`
- LLM safety violation rate: `0.0`

## Key Result

All verified policies achieved dense reward performance with zero safety violations in the local OpenEnv sepsis benchmark.

## Notes

- The OpenAI-backed policy was constrained to the environment action schema and guarded against unsupported outputs.
- In this environment, the observed performance ceiling is `0.9867`, and both the LLM-controlled run and ID3QNE matched that ceiling.
