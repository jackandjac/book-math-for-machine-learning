# Math for Machine Learning — A Software Engineer's Guide

## Project Structure

```
book-math-for-machine-learning/
├── chapters/          # Markdown drafts (one file per chapter)
├── code/              # Python code examples (runnable with numpy/sympy)
├── latex/             # LaTeX source for final PDF
├── assets/            # Diagrams and figures
└── README.md
```

## Building the PDF

### Requirements
- `pandoc` (converts Markdown → LaTeX → PDF)
- `texlive` or `MacTeX` (LaTeX engine)
- `python3` with `numpy`, `matplotlib`, `sympy`

### Build
```bash
make pdf        # Build full PDF from LaTeX source
make chapters   # Build individual chapter PDFs
make clean      # Remove build artifacts
```

### Install tools (macOS)
```bash
# Via internal Artifactory or IT-approved channels
brew install pandoc
brew install --cask mactex
```

## Chapters

| # | File | Title |
|---|------|-------|
| 1 | chapters/01-why-math-for-ml.md | Why Math for ML? |
| 2 | chapters/02-vectors.md | Linear Algebra I: Vectors |
| 3 | chapters/03-matrices.md | Linear Algebra II: Matrices |
| 4 | chapters/04-decompositions.md | Linear Algebra III: Decompositions |
| 5 | chapters/05-derivatives.md | Calculus I: Derivatives |
| 6 | chapters/06-gradients-optimization.md | Calculus II: Gradients & Optimization |
| 7 | chapters/07-probability-foundations.md | Probability I: Foundations |
| 8 | chapters/08-distributions.md | Probability II: Key Distributions |
| 9 | chapters/09-statistics-for-ml.md | Statistics for ML |
| 10 | chapters/10-putting-it-together.md | Putting It All Together |
