# Paper Writing Guide

## LaTeX Document Information

### File Location
- **Directory**: `paper_draft/` (git-ignored for privacy)
- **Main Paper**: `paper_draft/paper_ieee_experimental.tex`
- **Status**: Draft focusing on experimental sections
- **Template**: IEEE Conference Format

### Important: Paper Privacy
The `paper_draft/` directory is completely excluded from git to protect unpublished research. This ensures:
- No accidental commits of draft papers
- Protection of intellectual property
- Compliance with conference submission policies

### Compilation Instructions

```bash
# Navigate to paper directory
cd paper_draft/

# Method 1: Use the compilation script
./compile.sh

# Method 2: Manual compilation
pdflatex paper_ieee_experimental.tex
pdflatex paper_ieee_experimental.tex

# If using BibTeX
bibtex paper_ieee_experimental
pdflatex paper_ieee_experimental.tex
pdflatex paper_ieee_experimental.tex
```

### Document Structure

The current document includes:

1. **Section: The Proposed Multi-Agent Framework**
   - Framework Overview (Three-layer architecture)
   - Agent Design and Responsibilities
     - PlannerAgent
     - ColumnDiscoveryAgent
     - TableAggregationAgent
     - TableDiscoveryAgent
     - TableMatchingAgent
   - Optimization Mechanisms
     - Multi-level Caching
     - Parallel Processing

2. **Section: Experimental Evaluation**
   - Experimental Setup
     - Datasets (100-table subset, 1,534-table complete)
     - Implementation Details
     - Evaluation Metrics
   - Performance Results
     - Effectiveness Metrics
     - Efficiency Metrics
     - Scalability Analysis
   - Ablation Study
     - Three-layer Architecture Impact
     - Multi-Agent Benefits
     - Caching Effectiveness

### Key Academic Improvements Made

1. **Formal Language**: 
   - Replaced informal phrases with academic terminology
   - Used passive voice where appropriate
   - Maintained consistent technical vocabulary

2. **Structure**:
   - Clear subsection hierarchy
   - Logical flow from architecture to results
   - Comprehensive ablation studies

3. **Technical Accuracy**:
   - Based on actual implementation
   - Real performance metrics (1.08s latency, 75% recall)
   - Honest discussion of limitations (30% precision)

### Tables and Figures Needed

The following figures/tables are referenced but need to be created:

1. `images/three_layer_architecture.pdf` - System architecture diagram
2. `images/scalability.pdf` - Scalability analysis chart

Current tables included:
- Table I: Effectiveness Metrics
- Table II: Query Latency Comparison
- Table III: Ablation Study Results
- Table IV: Multi-Agent Performance

### Writing Style Guidelines

1. **Tense Usage**:
   - Present tense for system description
   - Past tense for experimental procedures
   - Present tense for results discussion

2. **Technical Terms**:
   - Define acronyms on first use
   - Consistent terminology throughout
   - Standard IR metrics (Precision, Recall, F1)

3. **Academic Phrases Used**:
   - "We propose..." / "Our system employs..."
   - "Experimental results demonstrate..."
   - "The findings indicate..."
   - "Future work will focus on..."

### Next Steps for Paper Completion

1. **Add Missing Sections**:
   - Abstract
   - Introduction with contributions
   - Related Work
   - Conclusion

2. **Create Figures**:
   - Architecture diagrams
   - Performance charts
   - Example queries and results

3. **Add References**:
   - Cite LangGraph, FAISS, Sentence-BERT
   - Include related work on schema matching
   - Reference benchmark datasets

4. **Performance Improvements**:
   - Address the 30% precision limitation
   - Propose concrete solutions
   - Add comparative evaluation with SOTA

### Important Notes

1. The paper focuses on **completed work only** as requested
2. Current precision (30%) is honestly reported with explanations
3. The three-layer architecture is the actual implementation
4. All performance numbers are from real experiments

### Citation Format

When ready to add references, use BibTeX format:

```bibtex
@inproceedings{yourlastname2025multiagent,
  title={Multi-Agent System for Large-Scale Data Lake Schema Matching and Discovery},
  author={Your Name},
  booktitle={Proceedings of IEEE Conference Name},
  year={2025},
  organization={IEEE}
}
```