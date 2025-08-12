# Paper Draft Directory

This directory contains LaTeX source files for academic paper drafts. 

## ⚠️ Important Notice

**This directory is excluded from version control to protect unpublished research.**

## Contents

- `paper_ieee_experimental.tex` - Main paper focusing on experimental sections
- LaTeX temporary files (automatically generated during compilation)

## Compilation Instructions

```bash
# Standard compilation
pdflatex paper_ieee_experimental.tex
pdflatex paper_ieee_experimental.tex

# With bibliography
bibtex paper_ieee_experimental
pdflatex paper_ieee_experimental.tex
pdflatex paper_ieee_experimental.tex

# Clean temporary files
rm -f *.aux *.log *.bbl *.blg *.out *.toc
```

## Directory Structure

```
paper_draft/
├── paper_ieee_experimental.tex    # Main paper file
├── figures/                       # Figure files (create as needed)
├── tables/                        # Table data (create as needed)
├── bibliography.bib               # References (create as needed)
└── .gitignore                     # Local git ignore rules
```

## Best Practices

1. Keep all paper-related files in this directory
2. Use meaningful commit messages if tracking locally
3. Create backups before major changes
4. Consider using version numbers for different drafts

## Security Notes

- This directory is git-ignored to prevent accidental publication
- Do not share paper drafts without co-author consent
- Keep sensitive research data separate