# Neural CAPTCHA buster

### Prerequisites

#### [Python](https://docs.python-guide.org/starting/install3/osx/)

Mac OS X comes with Python 2.7 out of the box. However, this project uses Python 3.7 so we'll install it using [Homebrew](https://brew.sh/):

```shell
brew install python
```

You'll probably need to augment your PATH in `.bash_profile` or `.zshrc`, too:

```shell
export PATH="/usr/local/opt/python/libexec/bin:$PATH"
```

If you're using Linux or Windows, you'll need to work this bit out yourself.

#### [Poetry](https://poetry.eustace.io/docs/)

This project uses [Poetry](https://poetry.eustace.io/docs/) to manage packages and environments. Frankly, I couldn't find a viable alternative amongst the vast menagerie of whacky and incomplete Python package managers. [PyPI](https://pypi.org/) is still terrible so packages aren't any easier to find but at least we won't need to fight with Pip and Virtualenv:

```shell
curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python
```

#### [Direnv](https://direnv.net/)

[Direnv](https://direnv.net/) is just a local environment manager, it automatically loads environment variables from an `.envrc` file in to your local environment, making it easier to manage sensitive or environment-specific application configuration.  In this case we're using it to manage server keys and augment the PATH to avoid global dependencies.
