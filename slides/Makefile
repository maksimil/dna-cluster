OUTDIR=./out
OBJDIR=./obj
OVERLEAFDIR=./overleaf

BASENAME=slides
OUTNAME=slides

BASETEX=./$(BASENAME).tex
BASEOBJ=$(OBJDIR)/$(BASENAME).pdf
BASEOUT=$(OUTDIR)/$(OUTNAME).pdf

LOGSECTION=@echo "\033[32m==== $(1) =====\033[0m"

SET_PREVIEWER=-e '$$pdf_previewer=q[zathura %S];'
# SET_XELATEX=-e '$$pdflatex=q[xelatex %O %S];'
# SET_PATCHES=-e 'ensure_path(q[TEXINPUTS], q[./patches]);'

LATEX=latexmk -pdf -outdir=$(OBJDIR) $(SET_PREVIEWER) $(SET_XELATEX)

.PHONY: watch build filesys clean plots

filesys:
	$(call LOGSECTION, Creating filesystem)
	[ ! -d $(OUTDIR) ] && mkdir -p $(OUTDIR) || true
	[ ! -d $(OBJDIR) ] && mkdir -p $(OBJDIR) || true

plots:
	$(call LOGSECTION, Creating plots)
	rm -rf plots/eps plots/pdf
	mkdir plots/eps 
	mkdir plots/pdf
	cd plots; ./makeplots.sh

clean:
	rm -rf $(OUTDIR) $(OBJDIR)

watch: filesys
	$(call LOGSECTION, Starting watch)
	$(LATEX) -pvc $(BASETEX)

build: filesys
	$(call LOGSECTION, Starting build)
	$(LATEX) $(BASETEX)
	$(call LOGSECTION, Optimizing)
	# pdfsizeopt --do-require-image-optimizers=no --quiet $(BASEOBJ) $(BASEOUT)
	cp $(BASEOBJ) $(BASEOUT)

overleaf-package:
	rm -rf $(OVERLEAFDIR)
	mkdir -p $(OVERLEAFDIR)
	cp -r slides.tex speech.txt $(OVERLEAFDIR)
	mkdir -p $(OVERLEAFDIR)/plots
	cp -r plots/pdf $(OVERLEAFDIR)/plots
