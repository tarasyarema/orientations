test:
	docker run -it -v ${PWD}:/tmp sagemath/sagemath:latest sage /tmp/src/lib_test.sage

docker:
	docker build . --file Dockerfile --tag orientations
	docker run -it orientations:latest sage tests.sage
