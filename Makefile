# List of Python demos (without file extenion) from the repo `demo` to include in the jupyterbook. 
# These files should be listed in `docs/_toc.yml`
DEMOS = "assembly_strategies"

# We use --set-kernel with jupytext to make it possible for binder to pick it up
doc: # Generate Sphinx HTML documentation, including API docs 
	@for demo in ${DEMOS}; do \
	jupytext --to=ipynb --set-kernel=python3 demo/$$demo.py --output=docs/$$demo.ipynb ;\
	done
	jupyter book build -W docs

clean-pytest: # Remove output from pytest
	rm -rf .pytest_cache
	rm -rf test/__pycache__

clean-coverage: # Remove output from coverage
	rm -rf .coverage htmlcov

clean-oasix: # Remove output from installing package
	rm -rf oasisx.egg-info oasisx/__pycache__

clean-mypy: # Remove output from mypy
	rm -rf .mypy_cache

clean: clean-mypy clean-oasisx clean-coverage clean-pytest
