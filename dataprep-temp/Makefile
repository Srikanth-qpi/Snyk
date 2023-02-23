SHELL: /bin/bash
PYTHON: python3

help:
	@echo "QpiAI Project utility script management tool"
	@echo ""
	@echo "Refer to the commands given below to execute various functions"
	@echo ""
	@echo make help   - To view this help manual
	@echo make install - Installs the poetry and pre-commit tools
	@echo make init   - Initialize git and setup pre-commit git hooks
	@echo make start  - Start the RESTful API Application
	@echo make test   - Start the pytest framework

.ONESHELL:
install:
	# Install poetry
	curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | ${PYTHON} -
	echo export PATH="${HOME}/.poetry/bin:${PATH}" >> ${HOME}/.bashrc
	source ${HOME}/.poetry/env
	poetry --version

	# Precommit install and setup
	pip3 install pre-commit
	
.ONESHELL:
init:
	# Initialize Git
	git init
	git branch -m main # Rename Main branch
	
	@echo "Configuring local git user \n\n"
	@echo "Enter you name :"
	read user_name
	@echo "Enter you email id:"
	read user_email
	git config --local user.name $$user_name
	git config --local user.email $$user_email
	
	@echo "Adding initial commit"
	git add .
	git commit -m "Initialized from Pro Microservice Template"
	
	git checkout -b dev main # Create dev branch

	@echo "Initializing pre-commit hooks..."
	pre-commit install
	pre-commit autoupdate

	make help

.ONESHELL:
test:
	export DB_LOG=false
	export PYTHONPATH=$$(pwd)
	pytest

.ONESHELL:
start:
	export PYTHONPATH=$$(pwd)
	poetry run python app/main.py
