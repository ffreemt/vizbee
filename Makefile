# https://tech.davis-hansson.com/p/make/
ifeq ($(origin .RECIPEPREFIX), undefined)
  $(error This Make does not support .RECIPEPREFIX. Please use GNU Make 4.0 or later)
endif
.RECIPEPREFIX = >

# https://earthly.dev/blog/python-makefile/
# https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# .venv/bin/python:

# https://tech.davis-hansson.com/p/make/
# .RECIPEPREFIX = >

# > # python3 -m venv .venv

# .venv/.install.stamp: .venv/bin/python requirements.txt
		# .venv/bin/python -m pip install -r requirements.txt
		# touch .venv/.install.stamp

# test: .venv/.install.stamp
		# .venv/bin/python -m pytest tests/
# make run

.PHONY = help setup test run clean install

# https://blog.mathieu-leplatre.info/tips-for-your-makefile-with-python.html
# NAME := multipage-app-panel-sin-cos.py

# NAME := vizbee.py
NAME := pyvizbee/__main__.py
INSTALL_STAMP := .install.stamp

dev: $(INSTALL_STAMP)
# > panel serve --port 8088 --show --autoreload $(NAME)
> nodemon -w pyvizbee -e py -x python -m pyvizbee

.PHONY: dev
run:  $(INSTALL_STAMP)
> python -m pyvizbee

install: $(INSTALL_STAMP)
# $(INSTALL_STAMP): pyproject.toml poetry.lock
$(INSTALL_STAMP): $(NAME)
> touch $(INSTALL_STAMP)

.PHONY: lint
lint: $(INSTALL_STAMP)
> flake8 --ignore=W503,E501,F401,E722,F841 $(NAME)
> isort --profile=black --lines-after-imports=1 --check-only  $(NAME)
> black --check $(NAME) --diff
> pyright $(NAME)
> # $(POETRY) run bandit -r $(NAME) -s B608

.PHONY: format
format: $(INSTALL_STAMP)
> isort --profile=black --lines-after-imports=1 $(NAME)
> black  $(NAME)

.PHONY: pylint
pylint: $(INSTALL_STAMP)
> pylint $(NAME)

.PHONY: pyright
pyright: $(INSTALL_STAMP)
> pyright $(NAME)
