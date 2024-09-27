ENV_PATH = ./.venv
REQ_FILE = config/requirements.txt

.PHONY: clean debug env install tests validate_code

# Run any pipeline located in src/pipelines
# Example: make setup_system
.DEFAULT:
	$(ENV_PATH)/bin/python -m src.__main__ -n "$(@)"

# Generate environment through venv
env:
	python3 -m venv $(ENV_PATH)
	$(ENV_PATH)/bin/pip install -r $(REQ_FILE)

# Install a library in virtual environment and update requirements
# Example: make install LIB=pandas
install: env
	$(ENV_PATH)/bin/pip install $(LIB)
	$(ENV_PATH)/bin/pip freeze > $(REQ_FILE)

# Debug any pipeline located in src/pipelines
# Example: make debug NAME=setup_system
debug:
	$(ENV_PATH)/bin/python -m pdb -m src.__main__ -n $(NAME)

# Run all project tests
tests:
	$(ENV_PATH)/bin/python -m unittest

# Check code syntax
validate_code:
	flake8 --exclude=./$(ENV_PATH),./build

clean:
	rm -rf $(ENV_PATH)
