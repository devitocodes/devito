# Contributing to Devito
We welcome third-party contributions, and we would love you to become an active contributor!

Software contributions are made via GitHub pull requests to https://github.com/devitocodes/devito.
If you are planning a large contribution, we encourage you to engage with us frequently to ensure that your effort is well-directed.
See below for more details.

Devito is distributed under the MIT License, https://github.com/devitocodes/devito/blob/main/LICENSE.md.
The act of submitting a pull request or patch (with or without an explicit Signed-off-by tag) will be understood as an affirmation of the following:

 Developer's Certificate of Origin 1.1

 By making a contribution to this project, I certify that:

 (a) The contribution was created in whole or in part by me and I
   have the right to submit it under the open source license
   indicated in the file; or

 (b) The contribution is based upon previous work that, to the best
   of my knowledge, is covered under an appropriate open source
   license and I have the right under that license to submit that
   work with modifications, whether created in whole or in part
   by me, under the same open source license (unless I am
   permitted to submit under a different license), as indicated
   in the file; or

 (c) The contribution was provided directly to me by some other
   person who certified (a), (b) or (c) and I have not modified
   it.

 (d) I understand and agree that this project and the contribution
   are public and that a record of the contribution (including all
   personal information I submit with it, including my sign-off) is
   maintained indefinitely and may be redistributed consistent with
   this project or the open source license(s) involved.

### Reporting issues
There are several options:
* Talk to us. You can join our Slack team via this [link](https://join.slack.com/t/devitocodes/shared_invite/zt-2hgp6891e-jQDcepOWPQwxL5JJegYKSA). Should you have installation issues, or should you bump into something that appears to be a Devito-related bug, do not hesitate to get in touch. We are always keen to help out.
* File an issue on [our GitHub page](https://github.com/devitocodes/devito/issues).

### Making changes
First of all, read of [code of conduct](https://github.com/devitocodes/devito/blob/main/CODE_OF_CONDUCT.md) and make sure you agree with it.

The protocol to propose a patch is:
* [Recommended, but not compulsory] Talk to us on Slack about what you're trying to do. There is a great chance we can support you.
* As soon as you know what you need to do, [fork](https://help.github.com/articles/fork-a-repo/) Devito.
* Create a branch with a suitable name.
* Write code following the guidelines below. Commit your changes as small logical units.
* Commit messages must adhere to the format specified below. We may ask you to rebase the commit history if it looks too messy.
* Write tests to convince us and yourself that what you have done works as expected and commit them.
* Run **the entire test suite**, including the new tests, to make sure that you haven't accidentally broken anything else.
* Push everything to your Devito fork.
* Submit a Pull Request on our repository.
* Wait for us to provide feedback. This may require a few iterations.

Tip, especially for newcomers: prefer short, self-contained Pull Requests over lengthy, impenetrable, and thus difficult to review, ones.

#### Commit messages and pull request titles

Your commit message should follow the following format: `tag: Message`

Where `tag` should be one of the following:
* `arch`: JIT and architecture (basically anything in `devito/arch`)
* `bench`: Anything related to benchmarking and profiling
* `ci`: Continuous Integration (CI)
* `ckp`: Checkpointing related
* `compiler`: Compilation (`operator`, `ir`, `passes`, `symbolics`, ...)
* `docs`: Updates or changes to docstrings or the documentation
* `dsl`: A change related to Devito's Domain Specific Language _Note: `fd`, `differentiable`, etc -- all belong to dsl_
* `examples`: Updates or changes to the examples or tutorials
* `install`: Related to installation (`docker`, `conda`, `pip`, ...)
* `reqs`: Package dependence updates
* `sympy`: Changes related to `sympy`
* `tests`: Updates or changes to the test suite
* `misc`: tools, docstring/comment updates, linting fixes, etc

`Message` should:
* Start with an upper case letter
* Start with a verb in first person
* Be as short as possible

Examples:
* `compiler: Fix MPI optimization pass`
* `install: Update Dockerfiles to new NVidia SDK`

Your Pull Request (PR) should follow a similar format: `tag: Title`
Additionally, you should add labels to the PR so that it can be categorised and the new changes can be correctly auto-summarised in the changelog.
Optionally, you may wish to select a reviewer, especially if you have discussed the PR with a member of the Devito team already.

### Coding guidelines

To ease the process of contributing we use [pre-commit](https://pre-commit.com/), which runs a small set of formatting and linting tests before you create a new commit.
To use `pre-commit` with Devito simply run:
```bash
pip install pre-commit
pre-commit install
```
Now when you make a commit, a set of pre-defined steps will run to check your contributed code.

These checks will:
* Trim any trailing whitespace
* Fix ends of files
* Check YAML formatting
* Check for accidentally added large files
* Sort imports using `isort` *
* Lint the codebase using `ruff` *
* Lint the codebase again using `flake8` *
* Check the code and documentation for typos using `typos` *
* Lint GitHub Actions workflow files using actionlint *
* Lint Dockerfiles using hadolint *

(* these checks will not change the edited files, you must manually fix the files or run an automated tool eg: `ruff check --fix` see below for details)

If you absolutely must push "dirty" code, `pre-commit` can be circumvented using:
```bash
git commit --no-verify -m "misc: WIP very dirty code"
```
However, this will cause CI to fail almost immediately!

Some coding rules are "enforced" (and automatically checked by CI), some are "strongly recommended", others are "optional" but welcome.

* We _enforce_ [PEP8](https://www.python.org/dev/peps/pep-0008/) using `ruff` and `flake8` with [a few exceptions](https://github.com/devitocodes/devito/blob/main/pyproject.toml).
* We _enforce_ a maximum line length of 90 characters.
* We _enforce_ indentation via 4 spaces.
* We _enforce_ imports to be at the top of a module and logically grouped using `isort`.
* We _strongly recommend_ to document any new module, class, routine, with [numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html).
* We _strongly recommend_ to follow standard Python coding guidelines:
  - Use camel caps for class names, e.g. ``class FooBar``.
  - Method names must start with a small letter; use underscores to separate words, eg: `def my_method(...)`.
  - Private class attributes and methods must start with an underscore, eg: `def _my_private_method(...)`.
  - Variable names should be explicative (Devito prefers "long and clear" over "short and FORTRAN like").
  - Comment your code, and do not be afraid of being verbose. The first letter must be capitalized. Do not use punctuation (unless the comment consists of multiple sentences).
* We _like_ that blank lines are used to logically split blocks of code implementing different (possibly sequential) tasks.

#### Pre-commit

You can use `pre-commit` to apply automated fixes for line endings, ends of files, import sorting and ruff linting.
All of these steps can be run together on your changes by running:
```bash
pre-commit run --hook-stage manual
```
Adding the `-a` flag runs this on all files in the repository, not just the files that you have changed.
Adding the name of the stage will run just one check.
See the [pre-commit-config](https://github.com/devitocodes/devito/blob/main/.pre-commit.yaml) file for the names of stages.

Some fixes can be automatically applied by the ruff linter, but may change the code in undesirable ways.
This step can only be run manually:
```bash
ruff check --fix --unsafe-fixes
```

### Adding tutorials or examples

We always look forward to extending our [suite of tutorials and examples](https://www.devitoproject.org/devito/tutorials.html) with new Jupyter Notebooks.
Even something completely new, such as a new series of tutorials showing your work with Devito, would be a great addition.
