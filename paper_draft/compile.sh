#!/bin/bash
# LaTeX compilation script for paper drafts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default file
TEX_FILE=${1:-paper_ieee_experimental.tex}
BASE_NAME="${TEX_FILE%.tex}"

echo -e "${GREEN}Compiling $TEX_FILE...${NC}"

# First compilation
pdflatex -interaction=nonstopmode "$TEX_FILE"
if [ $? -ne 0 ]; then
    echo -e "${RED}First compilation failed!${NC}"
    exit 1
fi

# Check if bibliography exists
if [ -f "${BASE_NAME}.aux" ] && grep -q "\\citation" "${BASE_NAME}.aux"; then
    echo -e "${YELLOW}Running BibTeX...${NC}"
    bibtex "$BASE_NAME"
    
    # Recompile twice for references
    echo -e "${GREEN}Recompiling for bibliography...${NC}"
    pdflatex -interaction=nonstopmode "$TEX_FILE"
    pdflatex -interaction=nonstopmode "$TEX_FILE"
else
    # Just compile once more for cross-references
    echo -e "${GREEN}Second compilation for cross-references...${NC}"
    pdflatex -interaction=nonstopmode "$TEX_FILE"
fi

# Check if PDF was created
if [ -f "${BASE_NAME}.pdf" ]; then
    echo -e "${GREEN}✓ Success! PDF created: ${BASE_NAME}.pdf${NC}"
    
    # Show PDF info
    if command -v pdfinfo &> /dev/null; then
        echo -e "\n${YELLOW}PDF Information:${NC}"
        pdfinfo "${BASE_NAME}.pdf" | grep -E "Pages:|File size:"
    fi
else
    echo -e "${RED}✗ Error: PDF was not created${NC}"
    exit 1
fi

# Optional: Clean temporary files
read -p "Clean temporary files? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Cleaning temporary files...${NC}"
    rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot *.nav *.snm *.vrb
    echo -e "${GREEN}✓ Cleaned${NC}"
fi