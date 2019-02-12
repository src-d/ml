.PHONY: check
check:
	! grep -R /tmp sourced/ml/tests
	flake8 --count
	pylint sourced.ml

.PHONY: docker-build
docker-build:
	docker build -t srcd/ml .

.PHONY: docker-build-core
docker-build-core:
	docker build -t srcd/ml-core --file Dockerfile.core .