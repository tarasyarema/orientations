PACKAGE = orientations
SAGE = sage

all: install test

docker-build:
	docker build . --file Dockerfile --tag orientations

docker-jupyter:
	docker run -it -p 8888:8888 orientations:latest sage-jupyter --NotebookApp.token= --NotebookApp.password=

build:
	rm -rf dist/*
	git describe --abbrev=0 > VERSION
	python setup.py sdist bdist_wheel
	twine check dist/*

release: build
	twine upload dist/*

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

.PHONY: all install develop test coverage clean clean-doc doc doc-pdf docker-build docker-jupyter build release
