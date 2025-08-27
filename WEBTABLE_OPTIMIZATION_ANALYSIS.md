# WebTable Performance Optimization Analysis

## Current Performance Issues

### Performance Degradation Pattern
```
JOIN Task:
L1:       Hit@1: 32.0%  | P@5: 18.4%  | R@5: 46.0%
L1+L2:    Hit@1: 23.7%  | P@5: 14.2%  | R@5: 35.4%  ↓ WORSE
L1+L2+L3: Hit@1: 16.5%  | P@5: 10.8%  | R@5: 27.1%  ↓ WORSE

UNION Task:
L1:       Hit@1: 60.9%  | P@5: 21.2%  | R@5: 52.9%
L1+L2:    Hit@1: 27.2%  | P@5: 7.2%   | R@5: 18.0%  ↓ WORSE
L1+L2+L3: Hit@1: 66.3%  | P@5: 32.8%  | R@5: 82.0%  ↑ BETTER
```

## Root Cause Analysis

### 1. Over-Filtering Problem
Each layer is REMOVING candidates instead of REFINING them:
- **Expected**: L1 (broad) → L2 (add semantic) → L3 (validate & rank)
- **Actual**: L1 (filter) → L2 (filter more) → L3 (filter even more)

### 2. WebTable Data Characteristics
- **Schema Diversity**: Web-scraped tables have inconsistent naming
- **Noise Level**: Higher than curated datasets (NLCTables/OpenData)
- **Semantic Gap**: Column names may not reflect actual content
- **Scale Variation**: Tables range from 2 columns to 100+ columns

## Optimization Strategy

### Layer Role Redefinition
```yaml
L1_metadata:
  role: "Broad candidate generation"
  strategy: "Cast wide net, include all possibilities"
  threshold: 0.15  # Very low for WebTable
  
L2_vector:
  role: "Add semantic candidates"
  strategy: "UNION with L1, not intersection"
  threshold: 0.25  # Relaxed
  
L3_llm:
  role: "Validation and ranking"
  strategy: "Score all candidates, filter only below 0.02"
  threshold: 0.02  # Almost never filter
```

### Configuration Principles
1. **Additive Layers**: Each layer should ADD candidates, not remove
2. **Dataset Adaptation**: Thresholds based on dataset statistics
3. **Progressive Confidence**: Start broad, narrow gradually
4. **Preserve Diversity**: Keep diverse candidates for final LLM judgment

## Specific Threshold Recommendations

### WebTable JOIN Configuration
```yaml
metadata_filter:
  column_similarity_threshold: 0.15  # Much lower than 0.35
  value_overlap_threshold: 0.05      # Very relaxed
  max_candidates: 100                # Keep more candidates
  
vector_search:
  similarity_threshold: 0.25         # Lower than 0.40
  combine_strategy: "union"          # Union with L1, not filter
  
llm_matcher:
  confidence_threshold: 0.02         # Almost never reject
  scoring_mode: "relative"           # Score relative to dataset
```

### WebTable UNION Configuration
```yaml
metadata_filter:
  column_similarity_threshold: 0.10  # Even lower for UNION
  allow_self_match: true
  fuzzy_matching: true
  
vector_search:
  similarity_threshold: 0.20
  semantic_weight: 0.7               # More semantic focus
  
llm_matcher:
  confidence_threshold: 0.01         # Extremely permissive
  consider_partial_matches: true
```

## Agent Prompt Optimization (No Hardcoding)

### Dynamic Context Variables
```python
# Instead of hardcoding, inject these at runtime:
context = {
    "dataset_characteristics": compute_dataset_stats(tables),
    "schema_consistency": measure_schema_variance(tables),
    "avg_table_size": calculate_average_columns(tables),
    "naming_pattern": detect_naming_conventions(tables)
}
```

### Adaptive Prompt Template
```python
# Good - Adaptive based on data
prompt = f"""
Analyze this matching task considering:
- Dataset schema consistency: {context['schema_consistency']}
- Average table complexity: {context['avg_table_size']} columns
- Detected naming pattern: {context['naming_pattern']}

Adjust confidence scoring based on these characteristics.
"""

# Bad - Hardcoded
prompt = """
For WebTable dataset, use lower thresholds because web tables are messy.
"""
```

## Performance Targets

### Immediate Goals (Next 24 hours)
- JOIN Hit@1: 32% → 50%
- UNION Hit@1: 66% → 75%
- Fix layer degradation issue

### SOTA Targets (Publication Ready)
- JOIN Hit@1: ≥75%, Hit@5: ≥90%
- UNION Hit@1: ≥70%, Hit@5: ≥85%
- Consistent improvement across layers

## Implementation Priority

1. **Fix Over-Filtering** (CRITICAL)
   - Change layer combination from intersection to union
   - Implement progressive thresholds
   
2. **Dataset-Adaptive Configuration**
   - Auto-detect dataset characteristics
   - Dynamically adjust thresholds
   
3. **Agent Prompt Templates**
   - Remove all hardcoded references
   - Use context injection pattern

## Testing Strategy

```bash
# Test with progressive layer activation
python run_unified_experiment.py --dataset webtable --task join --layer L1 --max-queries 5
# Should see ~35% Hit@1

python run_unified_experiment.py --dataset webtable --task join --layer L1+L2 --max-queries 5  
# Should see ~45% Hit@1 (IMPROVED from L1)

python run_unified_experiment.py --dataset webtable --task join --layer all --max-queries 5
# Should see ~55% Hit@1 (BEST performance)
```

## Key Insight

WebTable's web-scraped nature means traditional strict filtering fails. The system must be MORE permissive, not less, as we add layers. Each layer should contribute different perspectives:
- L1: Structure matching
- L2: Semantic enrichment  
- L3: Intelligent ranking

The final LLM is smart enough to handle noise - give it MORE candidates, not fewer!