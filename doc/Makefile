# Minimal makefile for Sphinx documentation
#
# You can set these variables from the command line.
SPHINXOPTS    ?=
SPHINXBUILD   ?= python3 -m sphinx
SPHINXAPIDOC  ?= python3 -m sphinx.apidoc
SPHINXPROJ    ?= sourced
SOURCEDIR     ?= .
BUILDDIR      ?= _build

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
all: Makefile
	@$(SPHINXAPIDOC) -H "$(SPHINXPROJ) documentation" -o . "../$(SPHINXPROJ)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

rebuild: Makefile
	@rm -rf "$(BUILDDIR)"
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile
