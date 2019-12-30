# Contributing to Devito

We always welcome third-party contributions. And we would love you to become an active contributor! There are a number of ways you can help us.

### Reporting issues

There are several options:
* Talk to us. You can join our Slack team via this [link](https://devito-slackin.now.sh/). Should you have installation issues, or should you bump into something that appears to be a Devito-related bug, do not hesitate to get in touch. We are always keen to help out.
* File an issue on [our GitHub page](https://github.com/devitocodes/devito/issues).

### Making changes

First of all, read of [code of conduct](https://github.com/devitocodes/devito/blob/master/CODE_OF_CONDUCT.md) and make sure you agree with it.

The protocol to propose a patch is:
* [Recommended, but not compulsory] Talk to us on Slack about what you're trying to do. There is a great chance we can support you.
* As soon as you know what you need to do, [fork](https://help.github.com/articles/fork-a-repo/) Devito.
* Create a branch with a suitable name.
* Write code following the guidelines below. Commit your changes as small logical units.
* Commit messages should adhere to the format `<tag>: <msg>`, where `<tag>` could be, for example, "ir" (if the commit impacts the intermediate representation), "operator", "tests", etc. We may ask you to rebase the commit history if it looks too messy.
* Write tests to convince us and yourself that what you've done works as expected. Commit them.
* Run **the entire test suite**, including the new tests, to make sure that you haven't accidentally broken anything else.
* Push everything to your Devito fork.
* Submit a Pull Request on our repository.
* Wait for us to provide feedback. This may require a few iterations.

Tip, especially for newcomers: prefer short, self-contained Pull Requests over lengthy, impenetrable, and thus difficult to review, ones.

### Coding guidelines

Some coding rules are "enforced" (and automatically checked by our Continuous Integration systems), some are "strongly recommended", others are "optional" but welcome.

* We _enforce_ [PEP8](https://www.python.org/dev/peps/pep-0008/), with a few exceptions, listed [here](https://github.com/devitocodes/devito/blob/master/setup.cfg#L3)
* We _enforce_ a maximum line length of 90 characters.
* We _enforce_ indentation via 4 spaces.
* We _suggest_ to use ``flake8`` to check the above points locally, before filing a Pull Request.
* We _strongly recommend_ to document any new module, class, routine, ... with [NumPy-like docstrings](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html#example-numpy) ("numpydoc").
* We _strongly recommend_ imports to be at the top of a module, logically grouped and, within each group, to be alphabetically ordered. As an example, condider our [__init__.py](https://github.com/devitocodes/devito/blob/master/devito/__init__.py): the first group is imports from the standard library; then imports from third-party dependencies; finally, imports from devito modules.
* We _strongly recommend_ to follow standard Python coding guidelines:
  - Use camel caps for class names, e.g. ``class FooBar``.
  - Method names must start with a small letter; use underscores to separate words, e.g. ``def _my_meth_...``.
  - Private class attributes and methods must start with an underscore.
  - Variable names should be explicative (Devito prefers "long and clear" over "short but unclear").
  - Comment your code, and do not be afraid of being verbose. The first letter must be capitalized. Do not use punctuation (unless the comment consists of multiple sentences).
* We _like_ that blank lines are used to logically split blocks of code implementing different (possibly sequential) tasks.

### Adding tutorials or examples

We always look forward to extending our [suite of tutorials and examples](https://www.devitoproject.org/devito/tutorials.html) with new Jupyter Notebooks. Even something completely new, such as a new series of tutorials showing your work with Devito, would be a great addition.
