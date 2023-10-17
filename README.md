# OSR
Code for NeurIPS 2023 paper "Recovering from Out-of-sample States via Inverse Dynamics in Offline Reinforcement Learning".

We have released the OSR demo v0.01.

## Installation

1. If you want to install [OSR] environment, you can use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate [OSR]
```

## Run Experiments
You can run OSR experiments using the following command:
```
python -m conservative_sac_main \
    --env [ENVIRONMENT]
```
If you want to run on CPU only, just add the `--device='cpu'` option.

## Credits
The CQL implementation is based on [CQL](https://github.com/young-geng/CQL). Thanks for their sharing codes.

## Code for OSR-v and OSR-10 is coming soon...
