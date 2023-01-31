# complexity-and-ml
Conceptual Complexity, Conceptual Analysis, and Machine Learning

## Setup
To recreate, we must first build our Conda environment. To achieve this, please run:
```
conda env create -f environment.yml
```
This should contain all of the packages you need to run our code.

Our notebooks will not run without embedding dictionaries that can be recreated using the scripts and instructions in `/vocab`. Once your environment is built, please head there and follow the instructions in the README in order to run the notebooks.

## Overview
This project contains several notebooks, each with a different purpose, containing different tests. Here is a short list:

* `simdef.ipynb` - contains exploratory tests grounded in angle and magnitude.
    * Contains a `define(positive: List[str])` which finds the closest word vector in `extended_vocab` to the positive of the vectors provided.
    * Contains definition verifications for each defined positive.
* `magnitudes.ipynb` 
    * Contains tests comparing and contrasting the magnitude of semantically similar and variant words
    * Compares magnitude and frequency; simplicity.
* `naive_def.ipynb`
    * Contains a naive approach to constructing the best two word positive as concerns cosine similartiy to the provided term.
    * Contains a revised approach to constructing positives that is more efficient and displays the `top_n` best definitions
* `definitions.ipynb`
    * Contains a `define(word: str)` which uses the revised approach from `naive_def` to construct the best two term positive as concerns cosine similarity to `word`.
    * Contains results not included/calculated in `naive_def`.