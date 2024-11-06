import numpy as np
from typing import List, Optional, Union
from assembled_ensembles.wrapper.abstract_weighted_ensemble import AbstractWeightedEnsemble # Abstract class for ensemble selection methods
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state

# Pymoo Imports (for running NSGA-II)
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize

# ART Imports for adversarial robustness evaluation
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier

# TODO: Add requirements (with specific version) in dockerfile

class MOOEnsembleSelection(AbstractWeightedEnsemble):
    """
    Multi-Objective Ensemble Selection using NSGA-II and Adversarial Robustness Toolbox.

    Parameters
    ----------
    base_models: List[Callable]
        The pool of fitted base models.
    n_generations: int
        Number of generations for NSGA-II algorithm.
    population_size: int
        Population size for the NSGA-II algorithm.
    score_metric: AbstractMetric
        Metric for accuracy evaluation.
    random_state: Optional[Union[int, np.random.RandomState]], default=None
        Random state for reproducibility.
    n_jobs: int, default=-1
        Cores to use for parallelization. If -1, use all available cores.
        Please be aware that multi-processing introduces a time overhead.
    """
    # Initialize parameters for MOO Ensemble Selection
    def __init__(self, base_models: List[Callable], n_generations: int, population_size: int,
                 score_metric: AbstractMetric, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_jobs: int = -1) -> None:
        super().__init__(base_models, "predict_proba")

        self.n_generations = n_generations
        self.population_size = population_size
        self.score_metric = score_metric
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs


