from assembled_ensembles.wrapper.abstract_ensemble import AbstractEnsemble
import numpy as np
from typing import List, Optional, Callable, Union
from assembled_ensembles.wrapper.abstract_weighted_ensemble import AbstractWeightedEnsemble # Abstract class for ensemble selection methods
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state

# Pymoo Imports (for running NSGA-II)
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize

# Import moo problem definition for Pymoo 
from moo_ensemble_problem import MOOEnsembleProblem

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

    TODO: Add parameter for final ensemble selection (value between 0 and 1 to prioritize either accuracy or robustness).
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

    def ensemble_fit(self, base_models_predictions: List, labels: np.ndarray) -> AbstractEnsemble:
        # Defines optimization problem by creating an instance of MOOEnsembleProblem. 
        # Fits the ensemble by finding optimal weights using NSGA-II.  
        # Chooses optimal ensemble/weight vector based on evaluation performance. 
        # TODO: Include trade-off parameter to balance objectives in selection step of final ensemble
        return super().ensemble_fit(base_models_predictions, labels)
    
    def predict_proba(self, X):
        # Predicts probabilities for new data using the fitted final ensemble.
        # Calls ensemble_predict method
        # Info: Is not used for the ensemble selection process itself
        return super().predict_proba(X)

    def ensemble_predict(self, predictions: Any | List) -> np.ndarray:
        # Aggregate ensemble prediction based on base model predictions and ensemble weights.
        return super().ensemble_predict(predictions)