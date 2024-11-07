import numpy as np
from pymoo.core.problem import Problem
from assembled_ensembles.util.metrics import AbstractMetric
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier
from sklearn.utils import check_random_state
from typing import List, Optional, Union

class MOOEnsembleProblem(Problem):
    """
    Defines the multi-objective optimization problem for Pymoo / ensemble selection via NSGA-II.

    Parameters
    ----------
    n_base_models: int
        Number of base models in the ensemble.
    predictions: List[np.ndarray]
        List of predictions from base models.
    labels: np.ndarray
        True labels.
    score_metric: AbstractMetric
        Metric for accuracy evaluation.
    base_models: List[Callable]
        List of base model instances.
    random_state: Optional[Union[int, np.random.RandomState]]
        Random state for reproducibility.
    n_jobs: int
        Number of jobs for parallel processing.

    TODO: Add parameter for final ensemble selection (value between 0 and 1 to prioritize either accuracy or robustness).
    """

    def __init__(self, n_base_models: int, predictions: List[np.ndarray], labels: np.ndarray,
                 score_metric: AbstractMetric, base_models: List[Callable],
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_jobs: int = -1):
        # Initialize the superclass (Pymoo problem)
        super().__init__(
            n_var=n_base_models,     # Number of decision variables (weights for base models)
            n_obj=2,                 # Number of objectives (accuracy and robustness)
            n_constr=0,              # Number of constraints
            xl=0.0,                  # Lower bounds for variables (weights >= 0)
            xu=1.0,                  # Upper bounds for variables (weights <= 1)
            type_var=np.double       # Data type of variables
        )

        self.predictions = predictions
        self.labels = labels
        self.score_metric = score_metric
        self.base_models = base_models
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs