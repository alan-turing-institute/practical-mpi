all: practical-mpi.pdf

slides.aux: slides.tex slides.bib
	pdflatex slides.tex

#slides.bbl: slides.aux
#	bibtex slides.aux

#slides.pdf: slides.tex slides.bbl macros.tex
slides.pdf: slides.tex macros.tex
	pdflatex slides.tex
	pdflatex slides.tex

practical-mpi.pdf: slides.pdf
	cp slides.pdf practical-mpi.pdf

clean:
	rm -f *.blg *.log *.pdf *.bbl *.aux *.out *.nav *.snm *.toc *.vrb

