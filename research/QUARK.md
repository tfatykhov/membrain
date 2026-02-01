# QUARK: Robust Retrieval under Non-Faithful Queries

**Paper:** [arXiv:2601.21049](https://arxiv.org/abs/2601.21049)
**Conference:** SIGIR 2026 (Melbourne, July 20-24)
**Authors:** Rita Qiuran Lyu, Michelle Manqiao Wang, Lei Shi (Microsoft)

## Summary

QUARK addresses retrieval failures caused by noisy, incomplete, or distorted queries.

### Problem
User queries are often "non-faithful" — they don't accurately represent the user's true intent due to:
- Memory limitations
- Ambiguity
- Language mismatch
- Incomplete information

Standard retrievers (BM25, DPR) fail when key semantics are missing.

### Solution: Query-Anchored Aggregation

1. **Generate recovery hypotheses**: Multiple plausible interpretations of the noisy query
2. **Retrieve with each**: Run retrieval for original query + all hypotheses
3. **Anchor to original**: Weight original query higher to prevent drift
4. **Aggregate scores**: `final_score = α × score(Q) + (1-α) × mean(scores(hypotheses))`

### Key Insights

- **Training-free**: No model fine-tuning required
- **Retriever-agnostic**: Works with BM25, DPR, LLM2Vec, ColBERT
- **Prevents hypothesis hijacking**: Bad hypotheses can't dominate due to anchoring
- **Robust to hypothesis count**: Performance stable across 3-10 hypotheses

### Results

Tested on BEIR benchmarks (FIQA, SciFact, NFCorpus):
- Improved Recall@M, MRR@M, nDCG@M over base retrievers
- Never degraded performance (safe to apply)
- Anchored aggregation outperforms max/mean/median pooling

## Relevance to Membrain

### Conceptual Mapping

| QUARK Concept | Membrain Equivalent |
|---------------|---------------------|
| Noisy query | Corrupted input embedding |
| Recovery hypotheses | Attractor basin samples |
| Query-anchored aggregation | Energy-weighted combination |
| Semantic anchor | Original input activation |

### Application to Attractor Dynamics (PR-010)

QUARK's approach suggests a strategy for noise-robust recall:

1. **Generate candidates**: Instead of single recall, generate K nearest neighbors
2. **Anchor to input**: Weight original input embedding in combination
3. **Energy-based filtering**: Use SNN energy landscape to filter bad candidates
4. **Aggregated recall**: Combine signals from multiple attractor basins

### Potential Implementation

```python
def robust_recall(self, query: np.ndarray, k_hypotheses: int = 5) -> np.ndarray:
    """QUARK-inspired robust recall with hypothesis generation."""
    # 1. Get K nearest neighbors as "recovery hypotheses"
    hypotheses = self.get_nearest_neighbors(query, k=k_hypotheses)
    
    # 2. Run attractor dynamics on each
    settled_states = [self.settle_attractor(h) for h in hypotheses]
    
    # 3. Anchor-weighted aggregation
    alpha = 0.6  # Anchor weight
    original_settled = self.settle_attractor(query)
    
    # Weighted combination with original as anchor
    aggregated = alpha * original_settled
    aggregated += (1 - alpha) * np.mean(settled_states, axis=0)
    
    return aggregated
```

### Research Questions

1. **Optimal α (anchor weight)**: Paper uses 0.5-0.7, but SNN dynamics may prefer different
2. **Hypothesis generation**: Use FlyHash neighbors vs. LLM-generated paraphrases?
3. **Energy filtering**: Can we use SNN energy to reject bad hypotheses before aggregation?
4. **Temporal aspects**: Does QUARK extend to temporal binding (PR-011)?

## Code Status

**No public implementation yet** (paper is from Jan 28, 2026).

Monitor:
- GitHub: [leirocks](https://github.com/leirocks) (Lei Shi)
- arXiv updates: [2601.21049](https://arxiv.org/abs/2601.21049)

## References

```bibtex
@inproceedings{lyu2026quark,
  title={QUARK: Robust Retrieval under Non-Faithful Queries via Query-Anchored Aggregation},
  author={Lyu, Rita Qiuran and Wang, Michelle Manqiao and Shi, Lei},
  booktitle={Proceedings of SIGIR 2026},
  year={2026}
}
```
