PACKAGE = orientations
SAGE = sage

all: install test

docker:
	docker build . --file Dockerfile --tag orientations
	docker run -it orientations:latest sage tests.sage

install:
	$(SAGE) -pip install --upgrade --no-index -v .

uninstall:
	$(SAGE) -pip uninstall $(PACKAGE)

develop:
	$(SAGE) -pip install --upgrade -e .

test:
	$(SAGE) setup.py test

coverage:
	$(SAGE) -coverage $(PACKAGE)/*

doc:
	cd docs && $(SAGE) -sh -c "make html"

doc-pdf:
	cd docs && $(SAGE) -sh -c "make latexpdf"

clean: clean-doc

clean-doc:
	cd docs && $(SAGE) -sh -c "make clean"

.PHONY: all install develop test coverage clean clean-doc doc doc-pdf docker
