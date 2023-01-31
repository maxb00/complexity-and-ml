# Vocabulary
This folder contains scripts that can be used to reconstruct the embedding repositories that power the included notebooks. Each has a slightly different usage, with attempted standardization. By default, each script will reconstruct (in the same directory) the embedding dictionary that sourced most examples in the paper - either GPT-3 Curie, OPT-13B, or T5-3b. For simple recreation, no arguments need to be specified.

GPT-3: `openaidl.py`

OPT: `opt_squeeze.py`

T5: `t5rip.py`
### Arguments (optional)
* `-v, --version`:
    * Specifies the version/size of a specific release to switch to. Must be in:
        * GPT-3: 'ada', 'curie', 'davinci', or 'babbage'
        * OPT: '13b', '6.7b', '2.7b', '1.3b', '350m', '125m'
        * T5: '3b', '11b', 'large', 'base', 'small'
* `-c, --cache`: ! not available for `openaidl.py` !
    * Specifies cache location to be used by HuggingFace Transformers for temporarily storing model data during download.
* `-d, --dest`:
    * Specifies destination directory for dictionary file.