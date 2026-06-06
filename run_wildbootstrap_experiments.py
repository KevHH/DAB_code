"""Sequential runner for the wild-bootstrap experiments in the notebook.

This script is intentionally not executed here. It mirrors the experiment cells
in `experiments_wild_bootstrap.ipynb`, including the added vary-n and vary-k
sections, so the user can launch the full workload from one place.
"""

import numpy as np

from settings.wild_bootstrap import mnist_rbf_kernel, simulate_rbf_kernel, cifar_rbf_kernel, higgs_rbf_kernel
from utils.dabstats import DABConfig


SIMULATED_COMPOSE = ["id", "random_ortho", "permute_coord"]
SIMULATED_AVERAGE = ["permute_coord", "random_ortho"]
IMAGE_COMPOSE = ["id", "rotate", "zoom", "rotate_zoom"]
IMAGE_AVERAGE = ["rotate", "zoom", "rotate_zoom"]
HIGGS_COMPOSE = ["id", "random_rotate"]
HIGGS_AVERAGE = ["random_rotate"]


ALPHA_GRID = list(np.arange(0.00, 0.31, 0.01))

SIMULATED_SPECS = [
    {
        "setting": "GaussianShift",
        "alpha_label": "WB2dGaussian",
        "shift_label": "WB2dGaussianShift",
    },
    {
        "setting": "RademacherShift",
        "alpha_label": "WB2dRademacher",
        "shift_label": "WB2dRademacherShift",
    },
    {
        "setting": "CentredGammaShift",
        "alpha_label": "WB2dCentredGamma",
        "shift_label": "WB2dCentredGammaShift",
    },
]


def run_simulated(cfg: DABConfig, setting: str, description: str):
    print(f"running {description}")
    simulate_rbf_kernel(cfg, setting=setting)


def run_mnist(cfg: DABConfig, description: str):
    print(f"running {description}")
    mnist_rbf_kernel(cfg)


def run_cifar(cfg: DABConfig, description: str):
    print(f"running {description}")
    cifar_rbf_kernel(cfg)

def run_higgs(cfg: DABConfig, description: str):
    print(f"running {description}")
    higgs_rbf_kernel(cfg)


def make_simulated_cfg(label, *, n, num_transform, alpha, testparam, num_sim=500):
    return DABConfig(
        n=n,
        d=2,
        num_transform=num_transform,
        compose_transform_names=SIMULATED_COMPOSE,
        average_transform_names=SIMULATED_AVERAGE,
        num_average_transforms=5,
        label=label,
        seed=0,
        num_sim=num_sim,
        batch_size=10,
        data_batch_size=10,
        alpha=alpha,
        testparam=testparam,
    )


def make_mnist_cfg(*, n, num_transform, alpha, testparam):
    MNIST_DOWNSIZE = 14
    return DABConfig(
        n=n,
        d=MNIST_DOWNSIZE * MNIST_DOWNSIZE,
        num_transform=num_transform,
        compose_transform_names=IMAGE_COMPOSE,
        average_transform_names=IMAGE_AVERAGE,
        num_average_transforms=1,
        label="WBmnist",
        seed=0,
        num_sim=1000,
        batch_size=10,
        data_batch_size=10,
        alpha=alpha,
        testparam=testparam,
    )


def make_cifar_cfg(*, n, num_transform, alpha, testparam):
    CIFAR_DOWNSIZE = 16
    return DABConfig(
        n=n,
        d=CIFAR_DOWNSIZE * CIFAR_DOWNSIZE * 3,
        num_transform=num_transform,
        compose_transform_names=IMAGE_COMPOSE,
        average_transform_names=IMAGE_AVERAGE,
        num_average_transforms=1,
        label="WBcifar",
        seed=0,
        num_sim=1000,
        batch_size=5,
        data_batch_size=5,
        alpha=alpha,
        testparam=testparam,
    )


def make_higgs_cfg(*, n, num_transform, alpha, testparam):
    HIGGS_D = 21
    return DABConfig(
        n=n,
        d=HIGGS_D,
        num_transform=num_transform,
        compose_transform_names=HIGGS_COMPOSE,
        average_transform_names=HIGGS_AVERAGE,
        num_average_transforms=1,
        label=f"WB180higgs",
        seed=0,
        num_sim=1000,
        batch_size=10,
        data_batch_size=10,
        alpha=alpha,
        testparam=testparam,
    )



def run_all_simulated():
    for spec in SIMULATED_SPECS:
        run_simulated(
            make_simulated_cfg(
                spec["alpha_label"],
                n=[5, 30],
                num_transform=[500],
                alpha=ALPHA_GRID,
                testparam=[0],
            ),
            spec["setting"],
            f'{spec["alpha_label"]} alpha sweep',
        )
        
        for shift in np.arange(0.0, 10.1, 0.5):
            run_simulated(
                make_simulated_cfg(
                    spec["shift_label"],
                    n=[5],
                    num_transform=[500],
                    alpha=[0.05],
                    testparam=[float(shift)],
                ),
                spec["setting"],
                f'{spec["shift_label"]} n=5 shift={shift:g}',
            )

        for shift in np.arange(0.0, 1.1, 0.05):
            run_simulated(
                make_simulated_cfg(
                    spec["shift_label"],
                    n=[30],
                    num_transform=[500],
                    alpha=[0.05],
                    testparam=[float(shift)],
                ),
                spec["setting"],
                f'{spec["shift_label"]} n=30 shift={shift:g}',
            )


def run_all_mnist():
    for noise in np.linspace(0.0, 0.3, 40):
        run_mnist(
            make_mnist_cfg(
                n=[50],
                num_transform=[100],
                alpha=[0.05],
                testparam=[['all_noise', noise]],
            ),
            f"WBmnist noise {noise}",
        )
    
    noise = 0.12
    run_mnist(
        make_mnist_cfg(
            n=list(range(10, 121, 5)),
            num_transform=[100],
            alpha=[0.05],
            testparam=[['all_noise', noise]],
        ),
        f"WBmnist noise {noise}",
    )
    
    noise = 0.12
    run_mnist(
        make_mnist_cfg(
            n=[50],
            num_transform=list(range(5, 101, 5)),
            alpha=[0.05],
            testparam=[['all_noise', noise]],
        ),
        f"WBmnist noise {noise}",
    )

    run_mnist(
        make_mnist_cfg(
            n=list(range(10, 121, 5)),
            num_transform=[100],
            alpha=[0.05],
            testparam=['all'],
        ),
        "WBmnist vary n",
    )

    run_mnist(
        make_mnist_cfg(
            n=[50],
            num_transform=list(range(5, 101, 5)),
            alpha=[0.05],
            testparam=['all'],
        ),
        "WBmnist vary k",
    )

def run_all_cifar():
    run_cifar(
        make_cifar_cfg(
            n=list(range(10, 301, 10)),
            num_transform=[100],
            alpha=[0.05],
            testparam=['oddeven'],
        ),
        "WBcifar vary n",
    )

    run_cifar(
        make_cifar_cfg(
            n=[30],
            num_transform=list(range(5, 101, 5)),
            alpha=[0.05],
            testparam=['oddeven'],
        ),
        "WBcifar vary k",
    )


    run_cifar(
        make_cifar_cfg(
            n=list(range(10, 121, 5)),
            num_transform=[100],
            alpha=[0.05],
            testparam=['odd'],
        ),
        "WBcifar odd vary n",
    )

    run_cifar(
        make_cifar_cfg(
            n=[30],
            num_transform=list(range(5, 101, 5)),
            alpha=[0.05],
            testparam=['odd'],
        ),
        "WBcifar odd vary k",
    )



def run_all_higgs():

    run_higgs(
        make_higgs_cfg(
            n=list(range(50, 2005, 50)),
            num_transform=[50],
            alpha=[0.05],
            testparam=['SIGNAL-BACKGROUND'],
        ),
        f"HIGGS SIGNAL-BACKGROUND vary n",
    )


    run_higgs(
        make_higgs_cfg(
            n=[2000],
            num_transform=list(range(5, 101, 2)),
            alpha=[0.05],
            testparam=['SIGNAL-BACKGROUND'],
        ),
        f"HIGGS SIGNAL-BACKGROUND vary k",
    )

    run_higgs(
        make_higgs_cfg(
            n=list(range(50, 2005, 50)),
            num_transform=[50],
            alpha=[0.05],
            testparam=['BACKGROUND'],
        ),
        f"HIGGS BACKGROUND vary n",
    )

    run_higgs(
        make_higgs_cfg(
            n=[2000],
            num_transform=list(range(5, 101, 2)),
            alpha=[0.05],
            testparam=['BACKGROUND'],
        ),
        f"HIGGS BACKGROUND vary n",
    )


def main():
    run_all_simulated()
    run_all_mnist()
    run_all_cifar()
    run_all_higgs()


if __name__ == "__main__":
    main()
