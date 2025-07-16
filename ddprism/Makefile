# —————— Configuration ——————
PYTHON       ?= python3
TEST_PATTERN ?= '**_test.py'

# —————— Targets ——————
.PHONY: test clean

test:
	export JAX_ENABLE_X64=True
	@echo "Running all tests matching $(TEST_PATTERN)…"
	$(PYTHON) -m unittest discover -p $(TEST_PATTERN)

clean:
	unset JAX_ENABLE_X64
	find . -name '__pycache__' -type d -exec rm -rf {} +
	find . -name '*.py[co]' -delete
