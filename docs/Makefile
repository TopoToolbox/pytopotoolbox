# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

EXAMPLEDIR    = ../examples
DOCEXAMPLEDIR = _examples
AUTOSUMMARYDIR = _autosummary

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
 
.PHONY: help Makefile

# Target to build HTML documentation, including copying/removing examples
html: 
	@echo "Copying examples directory..."
	@cp -r $(EXAMPLEDIR) $(DOCEXAMPLEDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Cleaning up _examples directory..."
	@rm -rf $(DOCEXAMPLEDIR)/*
	@echo "Cleaning up _autosummary directory..."
	@rm -rf $(AUTOSUMMARYDIR)/*

clean:
	@echo "Removing everything under '_build'..."
	@rm -rf $(BUILDDIR)/*
	@echo "Cleaning up '_autosummary' directory..."
	@rm -rf $(AUTOSUMMARYDIR)/*
	@echo "Cleaning up '_examples' directory..."
	@rm -rf $(DOCEXAMPLEDIR)/*

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
