# Neural CAPTCHA buster
## An AWS Lambda for reading CAPTCHAs using a Convolutional Neural Network (CNN)

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

#### [Direnv](https://direnv.net/) (Optional)

[Direnv](https://direnv.net/) is just a local environment manager, it automatically loads environment variables from an `.envrc` file in to your local environment, making it easier to manage sensitive or environment-specific application configuration.  In this case we're using it to manage server keys and augment the PATH to avoid global dependencies.

### Configuring the development environment

To make life simpler, we won't deal with our virtual environments manually. To automatically configure a new virtual environment and install the required dependencies, just use [Poetry](https://poetry.eustace.io/docs/):

```shell
poetry install
```

### Building the training dataset

To correctly read the text from the captchas, we'll need some training data. In this case, we're going to create labelled and normalised images for each letter in lots of solved captcha.

Firstly, we'll need a directory full of many, many solved captchas that looks like this:

```
├─ solved_captchas
│ ├─ 60958917271_brarded.jpg
│ ├─ 60957005257_dowties.jpg
│ ├─ 60958918641_itsonia.jpg
│ └─ 60958919248_clotler.jpg
```

> The filenames are important, everything after the `_` is assumed to be the captcha solution.

Once we've got the solved captured in place we can create the training images by running our contour detection script:

```shell
poetry run python build_training_images.py
```

Afterwards we should see the `training_images` directory fill up with individual images for each possible letter divided in to separate directories:

```
├─ training_images
│ ├─ a
│ │ ├─ 1552589077.315438.png
│ │ ├─ 1552589077.367729.png
│ │ └─ 1552589077.403722.png
│ ├─ b
│ │ ├─ 1552589077.662636.png
│ │ ├─ 1552589077.663386.png
│ │ └─ 1552589077.688153.png
```

### Training the model

Once we have some training images we can build and train a model by running our training script:

```shell
poetry run python train_model.py
```

That will create a model and some weightings in the root directory:

```
├─ model.json
├─ model.h5
```

### Testing the service

Assuming the training went well, we can test out the AWS Lambda using `serverless`:

```shell
poetry run serverless invoke local --function readCaptcha --data '{ "image": "<BASE64_ENCODED_IMAGE_STRING>" }'
```
