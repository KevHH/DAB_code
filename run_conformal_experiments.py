from utils.dabstats import DABConfig
from settings.conformal import simulate_amptorch_conformal, simulate_linear_regression, simulate_mmlu_conformal
import numpy as np

# example code for linear regression

calib_ratio=0.2

for n in range(20, 261, 20):
    simulate_linear_regression(DABConfig(
        n=[n],
        d=2,
        num_transform=[1],
        compose_transform_names=['id', 'random_ortho', 'permute_coord'],
        average_transform_names=['random_ortho', 'permute_coord'],
        num_average_transforms=1,
        label=f'CP2dGaussian_calib{calib_ratio}',
        seed=0,
        num_sim=500,
        alpha=[0.05],
        testparam=[],
        one_sided='upper',
        batch_size=100,
    ), 
        setting='Gaussian',
        calib_ratio=calib_ratio,
    )

for n in range(20, 261, 20):
    simulate_linear_regression(DABConfig(
        n=[n],
        d=2,
        num_transform=[1],
        compose_transform_names=['id', 'random_ortho', 'permute_coord'],
        average_transform_names=['random_ortho', 'permute_coord'],
        num_average_transforms=1,
        label=f'CP2dCentredGamma_calib{calib_ratio}',
        seed=0,
        num_sim=500,
        alpha=[0.05],
        testparam=[],
        one_sided='upper',
        batch_size=100,
    ), 
        setting='CentredGamma',
        calib_ratio=calib_ratio,
    )


for n in range(20, 261, 20):
    simulate_linear_regression(DABConfig(
        n=[n],
        d=2,
        num_transform=[1],
        compose_transform_names=['id', 'random_ortho', 'permute_coord'],
        average_transform_names=['random_ortho', 'permute_coord'],
        num_average_transforms=1,
        label=f'CP2dRademacher_calib{calib_ratio}',
        seed=0,
        num_sim=500,
        alpha=[0.05],
        testparam=[],
        one_sided='upper',
        batch_size=100,
    ), 
        setting='Rademacher',
        calib_ratio=calib_ratio,
    )

    
# example code for AmpTorch

n_train = None
for neighbors in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
    for n in [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 390, 420, 450, 480, 520, 550]:
        simulate_amptorch_conformal(
            DABConfig(
                n=[n],
                d=64,
                num_transform=[1],
                compose_transform_names=["id"],
                label=f"CPAmpTorch_unsort_neighbors{neighbors}", 
                seed=0,
                num_sim=500,
                alpha=[0.05],
                testparam=[],
                one_sided="upper",
                batch_size=2,
            ),
            k_neighbors=neighbors,
            n_train=n_train,
            random_unsort=True,
        )

    simulate_amptorch_conformal(
        DABConfig(
            n=[180],
            d=64,
            num_transform=[5, 10, 15, 20, 25, 30],
            compose_transform_names=["id"],
            label=f"CPAmpTorch_unsort_neighbors{neighbors}", #f"CPAmpTorch_ntrain{n_train}",
            seed=0,
            num_sim=500,
            alpha=[0.05],
            testparam=[],
            one_sided="upper",
            batch_size=2,
        ),
        k_neighbors=neighbors,
        n_train=n_train,
        random_unsort=True,
    )


# example code for MMLU

for n in range(2, 201, 5):
    simulate_mmlu_conformal(
        DABConfig(
            n=[n],
            d=4,
            compose_transform_names=["id", "random_permute"],
            num_transform=[10],
            label=f"CP_MMLU_FIRST6",
            seed=0,
            num_sim=100,
            alpha=[0.05],
            testparam=[],
            one_sided="upper",
            batch_size=10,
        ),
        subjects=["college_computer_science", "formal_logic", "high_school_computer_science", "computer_security", "machine_learning", "clinical_knowledge"],
        scores_dir="data/mmlu/local_scores_ALL",
    )


simulate_mmlu_conformal(
    DABConfig(
        n=[10],
        d=4,
        compose_transform_names=["id", "random_permute"],
        num_transform=[2,4,6,8,10,12,14,16,18,20],
        label=f"CP_MMLU_FIRST6",
        seed=0,
        num_sim=100,
        alpha=[0.05],
        testparam=[],
        one_sided="upper",
        batch_size=10,
    ),
    subjects=["college_computer_science", "formal_logic", "high_school_computer_science", "computer_security", "machine_learning", "clinical_knowledge"],
    scores_dir="data/mmlu/local_scores_ALL",
)
