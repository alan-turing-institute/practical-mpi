all: tutorial-3-training.pdf

slides.aux: slides.tex slides.bib
	pdflatex slides.tex

#slides.bbl: slides.aux
#	bibtex slides.aux

#slides.pdf: slides.tex slides.bbl macros.tex
slides.pdf: slides.tex macros.tex
	pdflatex slides.tex
	pdflatex slides.tex

tutorial-3-training.pdf: slides.pdf
	cp slides.pdf tutorial-3-training.pdf

clean:
	rm -f *.blg *.log *.pdf *.bbl *.aux *.out *.nav *.snm *.toc *.vrb

