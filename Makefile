PYTHON ?= python

.PHONY: all test report verify lint
all:
	$(PYTHON) -m credit_risk all
test:
	$(PYTHON) -m pytest -q
report:
	$(PYTHON) -m credit_risk report
verify:
	$(PYTHON) -m credit_risk verify
lint:
	$(PYTHON) -m ruff check src tests
