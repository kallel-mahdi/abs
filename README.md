# Augmented Bayesian Policy Search

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Description

This repository contains the official implementation for the paper [Augmented Bayesien Policy Search](https://openreview.net/forum?id=OvlcyABNQT) accepted at ICLR 2024.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Installation

Create new virtual environment : 

```python -m venv /path/to/new/virtual/environment```

Install requirements file :

```python -m venv /path/to/new/virtual/environment```

Install jax separately depending on your hardware (a GPU is definetely recommended):

CPU:
```pip install -U "jax[cpu]```

GPU:

```pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html```




## Usage

You need to specify a wandb api key for the environment variables at each script.

To run the ARS baseline just run :

```python run_ars.py ```

To run MPD :

```python run_mpd.py --algo_name mpd```


You can an algorithm with parameters different from the default configs e.g :

```python run_mpd.py --algo_name abs --reset_critic False --n_critics 10```




## License

This project is licensed under the [MIT License](LICENSE).
