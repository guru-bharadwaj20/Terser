# 2.2.2 Token counts

`data/interim/token_counts.parquet` (DVC stage `token_counts`): one row per (prompt, candidate, target)
for the 1,800 labelled prompts and their 10,800 candidates (rejected ones included). **37,800 rows =
12,600 texts × 3 tokenizers, 0 missing** (`check_complete`); 10,117 unique texts; 1.4 s on CPU.

Targets: `gpt-4o` (tiktoken), `qwen2.5`, `llama-3.1`. `claude` is counted only through the Anthropic
token-counting endpoint, and no `ANTHROPIC_API_KEY` is configured on this machine. Rerunning the stage
with the key set adds it (counts are cached per text).

Median original prompt: 80 / 84 / 81 tokens (gpt-4o / qwen2.5 / llama-3.1).

Median token savings vs. the original prompt (all candidates, before any quality filter):

| Method | gpt-4o | llama-3.1 | qwen2.5 |
|:------------|---------:|------------:|----------:|
| llm_high | 0.620 | 0.617 | 0.607 |
| llm_medium | 0.548 | 0.550 | 0.543 |
| llm_low | 0.455 | 0.457 | 0.447 |
| llmlingua2 | 0.518 | 0.519 | 0.512 |
| ops_CHFIDSL | 0.016 | 0.016 | 0.016 |
| ops_FHC | 0.004 | 0.004 | 0.004 |

The three tokenizers agree to within ~1.5 pp per method (consistent with C1.5). Operator stacks barely
shorten casual prompts (as in C1.4), so savings must come from the generative candidates.
Whether those savings preserve quality is measured in 2.3.3 and 2.4.
