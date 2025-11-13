PYTHON=python
PIP=$(PYTHON) -m pip

init:
	$(PIP) install -r requirements.txt

install: init
	$(PIP) install -e .

test:
	$(PYTHON) -m pytest -v

build:
	rm -rf build *.egg-info dist
	$(PYTHON) setup.py sdist bdist_wheel

.PHONY: init install uninstall build
