# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# Serve the HTML output on a local webserver
.PHONY:
serve:
	sphinx-autobuild --host 0.0.0.0 --port 8888 "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)


# Publish the HTML output to Github Pages
github: html
	ghp-import -m 'Generated from "$(shell git symbolic-ref --short HEAD)" @ $(shell git rev-parse --short HEAD)' -n $(BUILDDIR)/html
	git push origin gh-pages
