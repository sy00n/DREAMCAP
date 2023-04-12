# DREAMCAP
DREAM: Debiasing Representation based Evidential open set Action recognition with Multi-modality

### TRAIN

```shell
sh train.sh
```
- `train_config.json` must be modified as you want to train.
    - `config`: Directory of Configuration Python File
    - `work_dir`: Directory to save Log & Checkpoint File
    - `resume_from`: Whether to do Resume
    - `validate`: Whether to validate during train
    - and so on,,,


### TEST

#### Closed Set Evaluation

```shell
sh test_closed_set.sh
```

#### Get Uncertainty Threshold & OpenSet Evaluation&Comparison

```shell
sh test_open_set.sh
```
