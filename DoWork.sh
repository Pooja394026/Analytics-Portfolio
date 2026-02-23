#!/bin/bash

echo "Running DoWork.sh..."

# Step 1: Run the Python script
echo "Running Python script..."
python Code/code.py  # Use lowercase 'python' and make sure it's the correct path

# Step 2: Compile the LaTeX document
cd Paper 
echo "Compiling LaTeX document..."

pdflatex paper.tex

# Use the .aux file, not the .bib file, with bibtex
bibtex paper

pdflatex paper.tex
pdflatex paper.tex

echo "Done. PDF saved as Paper.pdf"


cd ..

echo ""
echo "Finished building document"
echo ""
echo "#--------------------------------------------"
echo ""

echo ""
echo "#---------------------------------------------"
echo "Move Paper/Paper.pdf..."

mv Paper/paper.pdf FPoojaPaper.pdf 

echo "Done."
echo "#---------------------------------------------"