CHAPTERS = chapters/01-why-math-for-ml.md \
           chapters/02-vectors.md \
           chapters/03-matrices.md \
           chapters/04-decompositions.md \
           chapters/05-derivatives.md \
           chapters/06-gradients-optimization.md \
           chapters/07-probability-foundations.md \
           chapters/08-distributions.md \
           chapters/09-statistics-for-ml.md \
           chapters/10-putting-it-together.md

LATEX_MASTER = latex/book.tex
PDF_OUT      = math-for-ml.pdf

.PHONY: pdf chapters clean lint

# Build full book PDF from LaTeX master
pdf: $(LATEX_MASTER)
	cd latex && xelatex book.tex && xelatex book.tex
	cp latex/book.pdf $(PDF_OUT)
	@echo "✓ PDF built: $(PDF_OUT)"

# Build PDF using md-to-pdf (via npx — requires node/npm, no pandoc/latex needed)
# Uses headless Chrome via Puppeteer. Run from project root.
md-pdf: $(CHAPTERS)
	cat $(CHAPTERS) > math-for-ml-book.md
	node $(shell npx --no-install -p md-to-pdf which md-to-pdf 2>/dev/null || echo "$$HOME/.npm/_npx/55158e48eb5c59f7/node_modules/md-to-pdf/dist/cli.js") \
	  --config-file .md-to-pdf.json \
	  math-for-ml-book.md \
	  < /dev/null
	cp math-for-ml-book.pdf $(PDF_OUT)
	rm math-for-ml-book.md
	@echo "✓ PDF built: $(PDF_OUT)"

# Build PDF directly from Markdown via pandoc (requires pandoc + texlive)
pandoc-pdf: $(CHAPTERS)
	pandoc $(CHAPTERS) \
	  --pdf-engine=xelatex \
	  --toc \
	  --toc-depth=3 \
	  --number-sections \
	  -V geometry:margin=1in \
	  -V fontsize=11pt \
	  -V mainfont="Libertinus Serif" \
	  -V monofont="Fira Code" \
	  --highlight-style=tango \
	  -o $(PDF_OUT)
	@echo "✓ Pandoc PDF built: $(PDF_OUT)"

clean:
	rm -f latex/*.aux latex/*.log latex/*.toc latex/*.out latex/*.pdf
	rm -f $(PDF_OUT)

# List all chapter Markdown files (quick sanity check)
lint:
	python3 -c "import glob; [print(f) for f in glob.glob('chapters/*.md')]"
