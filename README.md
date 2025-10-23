# SUAS-2026

Missouri S&amp;T Multirotor Design Team's code for the Association for Unmanned Vehicle Systems International's 2026 Student Unmanned Aerial Systems Competition (AUVSI SUAS 2026) hosted by RoboNation

## Table of contents

- [Installations](#installations)
    - [Git](#git)
    - [GitHub Credential Manager](#github-credential-manager)
    - [uv](#uv)
    - [Python](#python)
    - [Getting the Repo](#getting-the-repo)
    - [Installing Dependencies](#installing-dependencies)
    - [Pre-commit](#pre-commit)
- [Development](#development)
    - [Running Code](#running-code)
    - [Branches](#branches)
    - [Commits and Contributing](#commits-and-contributing)
    - [Pull Requests](#pull-requests)
- [License](#license)

## Installations

*Note: It is recommended that you develop for this repository on Ubuntu 22.xx . All steps from this point on will assume you are on Ubuntu.*

This guide will walk you through the process of getting set up with the repo and the tools you will need for development.

### Git

Make sure you have git installed with `git --version`. If you do not, you can install it with `sudo apt-get install git`.

### GitHub Credential Manager

GitHub handles credentials in a way that can be confusing to use at the command line. This can be remedied by using GitHub CLI. GitHub CLI will store your Git credentials for HTTPS Git operations.

First, install `curl` with `sudo apt-get install curl`.

Then, run the command (yes this is one command):

```
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
&& sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
&& sudo apt update \
&& sudo apt install gh -y
```

Run the command `gh auth login` and follow the prompts. For options, choose `GitHub.com`, `HTTPS`, `Yes`, `Login with a web browser`. Authorize the session in the web browser and your GitHub credentials will be saved.

### uv
We use [uv](https://docs.astral.sh/uv/) to manage Python versions and our dependencies. You can install it with these commands:

Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Python

This repository uses Python 3.12. You can install it easily through uv with the command `uv python install 3.12`.

### Getting the Repo

Go to your Documents folder and clone the repo with `git clone https://github.com/MissouriMRR/SUAS-2026.git`

### Installing Dependencies

Once you have the correct Python version installed, you can install all the required dependencies with the command `uv sync`. This will create a virtual environment in the `.venv` folder.

Every time the dependency list changes, you should run `uv sync` to update your virtual environment.

### Pre-commit

Pre-commit is a git hook that will check your code to make sure it is up to our standards before you make a commit. It is required that your code passes our pre-commit checks to be merged into the develop branch. That being said, when working on your own branch, you can add the `--no-verify` flag to your git commit command in order to bypass pre-commit. Just make sure you successfully run pre-commit before you submit a pull request.

uv will install pre-commit for you once you run `uv sync`.

Run the command `pre-commit install` to install the pre-commit hooks we use to check code.

You can test that this worked by running `pre-commit run`. This will run all of your code changes against the pre-commit hooks.

## Development

### Running Code

uv creates a virtual environment in the `.venv` folder that stores the correct Python versions and libraries for the codebase, but by default, it will not be used when running `python` commands.

You can run python files inside of the virtual environment using `uv run <filename>`.

You can also activate the virtual environment, allowing you to run `python` commands like normal, by running these commands:

Windows: `.venv\Scripts\activate`

Linux/MacOS: `source .venv/bin/activate`

There are other activation files for different shells, such as `source .venv/bin/activate.fish` for [Fish shell](https://fishshell.com/).

If you try to run a python file that is nested inside of other folders, you may run into issues with imports not resolving correctly. You can fix this by add the `-m` flag to your `uv run` or `python` commands, and replacing the file name with the import path to the file. For example: `uv run -m flight.test_files.connection_test` or `python -m flight.test_files.connection_test`.

### Branches

For each issue that you work on, you should create a new branch.

1. Run `git checkout -b feature/issue` to create a new branch. Replace issue with something descriptive.
2. Run `git push` to push your new branch to the repo.

### Commits and Contributing

***Note: Never directly commit to the `develop` branch! Make sure you are on a separate branch.***

1. Once you are on a new branch, you can start writing new code.
2. Add files to your next commit using `git add <filename>`.
3. Run `git commit -m "Description of changes"` to commit your code to the repo.
4. You should test your code in a poetry shell. Open the shell using `poetry shell`.
5. When you attempt to commit, pre-commit should automatically check your code to make sure it is up to our standards. If you fail the tests, go back and change what it requests. You can also use the command `pre-commit run` to run pre-commit checks without commiting.
6. Alternatively, if you just want to quickly push code to the repo, you can add the `--no-verify` flag to your git commit command to skip running pre-commit. For example, `git commit --no-verify -m "Updated readme"`. *Note, however, that your code will not be allowed to be merged into develop until it passes our pre-commit checks, so make sure you go back and fix any issues before submitting a pull request*.
7. Use `git push` to push your commits to the remote repo.


### Pull Requests

Once you have code that you think is ready to merge into develop, you can submit a pull request.

A template for your pull requests is available at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`

Your pull request should describe what changes you made and what issue you solved.

On the sidebar, request a review from your sublead, assign yourself, apply appropriate labels, add to your subteam's project board, and tie to an issue.

## License

We adopt the MIT License for our projects. Please read the [LICENSE](LICENSE) file for more info
