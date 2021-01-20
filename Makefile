docker:
	docker build . --file Dockerfile --tag orientations
	docker run -it orientations:latest sage tests.sage
