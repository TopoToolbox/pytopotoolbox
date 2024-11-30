# Contribution Guidelines

First off, thanks for taking the time to contribute!

The following is a set of guidelines for contributing to the TopoToolbox Python Library. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please report it by opening an issue. Please include:

- A clear and descriptive title.
- A detailed description of the problem.
- Steps to reproduce the issue.
- Any error messages you encountered.

### Suggesting Enhancements

If you have an idea for an enhancement or new feature, please open an issue to discuss it. Please include:

- A clear and descriptive title.
- A detailed description of the proposed enhancement.
- Any relevant examples or screenshots.

### Submitting Pull Requests

If you have a patch or new feature that you would like to contribute, please submit a [pull request (PR)](https://guides.github.com/introduction/flow/). Before you do, please ensure the following:

1. Fork the repository and create your branch from `main`.
2. If you have added code that should be tested, add tests.
3. If you have changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

Check the [developer documentation](dev.rst) for more details about setting up your development environment and running the tests.

### Code Style

Please follow the [PEP 8](https://pep8.org/) style guide for Python code. Make sure your code passes the linting tests (`pylint --rcfile=pyproject.toml src/topotoolbox/`, `mypy --ignore-missing-imports src/topotoolbox`) and the pytests (`python -m pytest`). Also add tests for new content you want to contribute.

It may also be a good idea to run pylint without specifying the rcfile to check for potential code smells that need to be addressed.

### Commit Messages

Use [clear and descriptive commit messages](https://cbea.ms/git-commit/). Follow these conventions:

- Use the present tense ("Analyze terrain" not "Analyzed terrain").
- Use the imperative mood ("Generate contour map" not "Generates contour map").
- Limit the first line to 50 characters or less. All other lines should not be longer than 72 characters.
- Reference issues and pull requests liberally.

### Documentation

Improvements to the documentation are always welcome. Please ensure that your changes are clear and concise.

Thank you for contributing!
