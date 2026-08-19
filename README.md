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
    - [Adding New Dependencies](#adding-new-dependencies)
    - [Pre-commit checks / `prek`](#pre-commit-checks--prek)
- [Repository Structure](#repository-structure)
- [Development](#development)
    - [Running Code](#running-code)
    - [Branches](#branches)
    - [Commits and Contributing](#commits-and-contributing)
    - [Pull Requests](#pull-requests)
    - [CI/CD](#cicd)
- [License](#license)

## Installations

> [!IMPORTANT]
> It is recommended that you develop for this repository on [Ubuntu 24.xx](https://mirror.math.princeton.edu/pub/ubuntu-iso/), or have it installed through [Windows Subsystem for Linux / WSL](https://missourimrr.github.io/docs/wsl/). All steps from this point on will assume you are on Ubuntu.

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

> [!TIP]
> In the future you will want to [set up an SSH key for your GitHub account.](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/about-ssh) This is the recommended way to log into GitHub/GitLab across campus and is standard in industry.

### uv
We use [uv](https://docs.astral.sh/uv/) to manage Python versions and our project's dependencies. You can install it with these commands:

macOS and Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`

Windows: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`

### Python

This repository uses Python 3.12. You can install it easily through uv with the command `uv python install 3.12`.

### Getting the Repo

Go to your Documents folder and clone the repo with `git clone <repository_link>`.

### Installing Dependencies

Once you have the correct Python version installed, you can install all the required dependencies with the command `uv sync`. This will create a virtual environment in the `.venv` folder.

Every time the dependency list changes, you should run `uv sync --dev` to update your virtual environment and install all required development tools.

This helps keep Python environments consistent across different members computers. Without it, it would be harder to find the core reason for bugs, as different people would be running different versions of libraries.

> [!TIP]
> We add tools like `prek` and `ruff` to the development group so that when computers do not need the full dependency list, like when we are downloading codebases onto the ground station computer, we can save space by not installing these dev tools.

### Adding New Dependencies

You can add new Python libraries as dependencies to the project by running `uv add <library_name>`. If it needs to be added to the development dependency list, add the `--dev` flag.

### Pre-commit checks / `prek`

`prek` is a Git hook that will check your code to make sure it is up to our standards before you make a commit. It is required that your code passes our pre-commit checks to be merged into the main branch. That being said, when working on your own branch, you can add the `--no-verify` flag to your git commit command in order to bypass pre-commit. Just make sure you successfully run `prek` before you submit a pull request.

uv will install `prek` for you once you run `uv sync --dev`.

Run the command `prek install` to install the pre-commit hooks we use to check code.

You can test that this worked by running `prek`. This will run all of your code changes against the pre-commit hooks.

## Repository Structure

- `main.py` - The entry point of the project. This should run all of the other code that your project provides in most cases.
- `pyproject.toml` - Project metadata, dependencies, and tool configurations for ruff, basedpyright, and uv.
- `uv.lock` - The `uv` lockfile that pins the exact versions of all dependencies and keeps environments consistent.
- `.python-version` - The Python version used by `uv` for this repository.
- `.pre-commit-config.yaml` - The configuration for the pre-commit hooks that `prek` installs.
- `.gitlint` - The rules that commit messages must follow.
- `.github/workflows/` - The GitHub Actions CI/CD workflows that run linting and commit message checks.
- `.venv/` - The virtual environment created by `uv sync`. It is not tracked by git.
- `.gitignore` - Lists the files and folders that git should not track.

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
3. Run `git commit -m "[#<issue_number>] Description of changes"` to commit your code to the repo.
4. You should test your code inside the virtual environment with `uv run` before committing.
5. When you attempt to commit, `prek` should automatically check your code to make sure it is up to our standards. If you fail the checks, go back and change what it requests. You can also run `prek` manually to run the checks without committing.
6. Alternatively, if you just want to quickly push code to the repo, you can add the `--no-verify` flag to your git commit command to skip running pre-commit. For example, `git commit --no-verify -m "[#42] Updated readme"`. *Note, however, that your code will not be allowed to be merged into develop until it passes our pre-commit checks, so make sure you go back and fix any issues before submitting a pull request*.

> [!NOTE]
> Commit titles must start with the issue number they address or relate to in brackets, e.g. `[#42] Fix flaky camera connection test`, and must be between 10 and 72 characters. These rules are enforced by [gitlint](https://jorisroovers.com/gitlint/) both in the pre-commit hook and in CI. No issue created for what you are working on? Make one yourself!
7. Use `git push` to push your commits to the remote repo.


### Pull Requests

Once you have code that you think is ready to merge into develop, you can submit a pull request.

A template for your pull requests is available at `.github/PULL_REQUEST_TEMPLATE/pull_request_template.md`

Your pull request should describe what changes you made and what issue you solved.

On the sidebar, request a review from your sublead, assign yourself, apply appropriate labels, add to your subteam's project board, and tie to an issue.

### CI/CD

This repository uses GitHub Actions to automatically check code quality. The workflows can be found in `.github/workflows/`.

- `lint.yml` runs on every push to `develop` and on every pull request. It runs `ruff check`, `ruff format --check`, and basedpyright, mirroring the checks that `prek` runs locally.
- `commitlint.yml` runs on every pull request. It runs `gitlint` against all of the commits in the pull request to enforce the commit message format.

These checks must pass before code can be merged into `develop`. Note that `--no-verify` does not affect CI results, so it is best to fix any issues locally before pushing.

## License

We adopt the MIT License for our projects. Please read the [LICENSE](LICENSE) file for more info.
