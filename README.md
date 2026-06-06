# Data augmented bootstrap (DAB)

Codebase for experiments in the paper ``Data augmented bootstrap: Unifying confidence interval
construction by approximate invariance".

The repository is organised as follows:

## DAB simulation scripts
The following scripts can be used to perform the DAB simulations reported in the paper:
- `run_bootstrap_experiments.py`
- `run_wildbootstrap_experiments.py`
- `run_conformal_experiments.py`
Note that other than simulations with 2d Gaussians / Rademachers / Centred Gammas, all other setups require either data download or generation from external code. See setup below.


## Visualisation notebooks
- `visualise_bootstrap.ipynb`
- `visualise_wild_bootstrap.ipynb`
- `visualise_conformal.ipynb`
The notebook `visualise_qm_syn_data.ipynb` is only used for visualising data from the `QM-sym-database`.


## Setup
The code in this repository can be run with the conda environment specified in `env_dab.yml`, which uses `jax`. One exception is that the GMP+SNN model used in [Hu et al. 22](https://arxiv.org/abs/2208.08337) is in PyTorch, for which we provide `env_amptorch.yml` for training the PyTorch model. 

Most external code bases have been linked in this repository as [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules). The specific scripts required for different setups are as follows:

### FermiNet experiments with Lithium in bootstrap
The FermiNet codebase has been linked as `ferminet` submodule, which contains the default configuration for training with Lithium. The samples produced at inference time can then be used by `settings/bootstrap.py`.

### MNIST
MNIST data needs to be downloaded and saved in the format `data/mnist/image_train.npy`, `data/mnist/label_train.npy`, `data/mnist/image_test.npy` and `data/mnist/label_test.npy`. 

### CIFAR
[CIFAR data](https://www.cs.toronto.edu/~kriz/cifar.html) needs to be downloaded to `data/cifar10_origin/data`. The notebook `visualise_wild_bootstrap,ipynb` provides the following code snippet for converting it to the format accepted by DAB:

```
import pickle
import numpy as np

with open('data/cifar10_origin/data', 'rb') as fo:
    output1 = pickle.load(fo, encoding='bytes')

CIFAR_SHAPE = [-1,3,32,32]

np.save('data/cifar10/image_train.npy', np.array(output1[b'data']).reshape(-1,3,32,32).transpose(0,2,3,1))
np.save('data/cifar10/label_train.npy', np.array(output1[b'labels']))

with open('data/cifar10_origin/data_batch_2', 'rb') as fo:
    output2 = pickle.load(fo, encoding='bytes')

np.save('data/cifar10/image_test.npy', np.array(output2[b'data']).reshape(-1,3,32,32).transpose(0,2,3,1))
np.save('data/cifar10/label_test.npy', np.array(output2[b'labels']))
```

### HIGGS
[Higgs data](https://archive.ics.uci.edu/ml/datasets/HIGGS) needs to be downloaded and converted to `.npy` files using e.g.~the following example script:

```
    from pathlib import Path
    import numpy as np
    import pandas as pd

    csv_path = Path("HIGGS.csv")
    out_dir = Path("higgs_npy_batches")
    out_dir.mkdir(exist_ok=True)

    rows_per_batch = 5000000

    for batch_idx, chunk in enumerate(
        pd.read_csv(
            csv_path,
            header=None,
            chunksize=rows_per_batch,
            dtype=np.float32,   # smaller than float64
        )
    ):
        if batch_idx >= 3:
            break
        arr = chunk.to_numpy(copy=False)
        out_path = out_dir / f"higgs_batch_{batch_idx:04d}.npy"
        np.save(out_path, arr)
        print(f"saved {out_path}  shape={arr.shape}  class1_count={np.sum(arr[:,0])}")
```

### GMP+SNN
We use the [QM-sym](https://www.nature.com/articles/s41597-019-0237-9) database, which is different from the data that the model used in [Hu et al. 22](https://arxiv.org/abs/2208.08337) was originally trained on. The model is found in the `amptorch` submodule and the data is found in the `QM-sym-database` submodule. Specifically we use `QM-sym-database/QM_sym_C4h_1.tar` for training and `QM-sym-database/QM_sym_C4h_2.tar` as the hold-out set for calibration and prediction.

**Pre-processing**. `scripts/qm_sym_to_ase.py` converts QM-sym data into the [ASE](https://ase-lib.org/ase/atoms.html) format. This produces the pickle files `data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl` and `data/qm_sym_c4h_1/qm_sym_c4h_2_ase_u0_ha.pkl`.

**Training**. The script to train the model is `scripts/train_qm_sym_c4h_1.py`.

**Load model in JAX for DAB and produce latents**. `scripts/load_amptorch_model.py` takes the trained pytorch model weights and load them in a JAX model for compatibility with DAB. It also produces the latent vectors for `data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl` and `data/qm_sym_c4h_1/qm_sym_c4h_2_ase_u0_ha.pkl`, via the following example commands:


```
    conda run -n amptorch python settings/load_amptorch_model.py \
    --checkpoint-dir logs/qm_sym_c4h_1_paper/checkpoints/DATETIME-qm_sym_c4h_1_gmp30_snn128_64_64 \
    --ase-path data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl \
    --element-bias logs/qm_sym_c4h_1_paper/qm_sym_c4h_1_element_bias.json \
    --out-dir data/qm_sym_c4h_1_jax \
    --dataset-name c4h_1 \
    --descriptor-cache logs/qm_sym_c4h_1_paper/processed/descriptors/GMPOrderNorm/dec49e45c583e80a9ec4a4ec2b858ccb \
    --batch-size 256
```

```
    conda run -n amptorch python settings/load_amptorch_model.py \
    --checkpoint-dir logs/qm_sym_c4h_1_paper/checkpoints/DATETIME-qm_sym_c4h_1_gmp30_snn128_64_64 \
    --ase-path data/qm_sym_c4h_2/qm_sym_c4h_2_ase_u0_ha.pkl \
    --element-bias logs/qm_sym_c4h_1_paper/qm_sym_c4h_1_element_bias.json \
    --out-dir data/qm_sym_c4h_2_jax \
    --dataset-name c4h_2 \
    --batch-size 256 \
    --overwrite
```

`scripts/check_amptorch_jax_parity.py` can be used to verify that the PyTorch model and the JAX model produce the same outputs.


The produced `*_energy_pred_ha.npy` and `*_latent.npy` files are then compatible with the DAB code in `settings/conformal.py`.

### Qwen3-14B + MMLU

The [MMLU](https://huggingface.co/datasets/cais/mmlu) dataset can be downloaded with `scripts/download_mmlu.py`. Qwen3-14B is installed via [Ollama](https://ollama.com/library/qwen3:14b), and `scripts/generate_mmlu_scores_local.py` is used to generate predictions. For augmentations, we use `scripts/augment_mmlu_options.py` to augment the text prompts in advance, run `scripts/generate_mmlu_scores_local.py` to store predictions, and randomly sample the augmented text-prediction pairs for DAB. 

An example code snippet for running `scripts/generate_mmlu_scores_local.py` is:


```
    conda run -n DAB python scripts/generate_mmlu_scores_local.py \
    --raw-dir data/mmlu/augmented_raw \
    --num-prompts 10 \
    --model qwen3:14b \
    --output-dir data/mmlu/local_scores_AUGMENTED \
    --progress-log-path data/mmlu/local_scores_AUGMENTED_progress.txt
```