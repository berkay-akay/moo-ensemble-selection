import numpy as np
from pymoo.core.problem import Problem
from assembled_ensembles.util.metrics import AbstractMetric
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier
from sklearn.utils import check_random_state
from typing import List, Optional, Callable, Union

class MOOEnsembleProblem(Problem):
    """
    Defines the multi-objective optimization problem for Pymoo / ensemble selection via NSGA-II.

    Parameters (required for Pymoo Problem)
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

    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate accuracy/robustness for a batch of ensembles/weight vectors.

        Parameters
        ----------
        x: np.ndarray
            Population/batch of weight vectors.
        out: dict
            Dictionary to store objective evaluation values.
        """
        # Initialize lists to store accuracy/robustness values
        f1 = []  # List for accuracy 
        f2 = []  # List for robustness

        # Iterate over weight vectors in population/batch
        for weights in x:
            # Normalize weights (to sum to 1)
            weights = weights / np.sum(weights)

            # Aggregate ensemble predictions using weights
            ensemble_pred = self._ensemble_predict(self.predictions, weights)

            # Evaluate accuracy
            accuracy = self.score_metric(self.labels, ensemble_pred, to_loss=False, checks=False)
            f1.append(-accuracy)  # Pymoo minimizes objective, so we need to negate (minimize negative accuracy = maximize accuracy)

            # Evaluate robustness (using adversarial attacks)
            robustness = self._evaluate_robustness(weights)
            f2.append(-robustness)  # Pymoo minimizes objective, so we need to negate (minimize negative robustness = maximize robustness) 

        # Store objective values in output dictionary
        out["F"] = np.column_stack([f1, f2])

    
    def _ensemble_predict(self, predictions: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        # TODO: Compute predictions by aggregating predictions of base models (using weights)
        return 

    
    def _evaluate_robustness(self, weights: np.ndarray) -> float:
        """
        Evaluate robustness of ensemble using ART black-box attack

        Parameters
        ----------
        weights: np.ndarray
            Ensemble weight vector.

        Returns
        -------
        float
            Robustness metric (adversarial accuracy).
        """
        # Create ensemble model that is compatible with ART (with internal class)
        ensemble_model = self._EnsembleModel(self.base_models, weights)

        # Wrap model with ART's SklearnClassifier
        classifier = SklearnClassifier(model=ensemble_model, clip_values=(0, 1))

        # Instantiate adversarial attack (f.e. Boundary Attack by Brendel et al. [2018])
        # TODO: Not final, maybe use other black-box attacks from ART 
        attack = BoundaryAttack(estimator=classifier)

        # Generate adversarial examples using the attack
        # TODO: Pass the required input data (self.X) correctly -> Data needs to be extracted from the metatask
        #       and passed to the MOOEnsembleSelection instance, after that it needs to be passed to the MOOEnsembleProblem instance so
        #       that it can be used for adversarial attack generation.
        x_test_adv = attack.generate(x=self.X) # self.X is not defined yet !!! 

        # Get ensemble predictions on adversarial examples
        adv_preds = classifier.predict(x_test_adv)

        # Evaluate robustness (adversarial accuracy)
        adv_accuracy = self.score_metric(self.labels, adv_preds, to_loss=False, checks=False)

        return adv_accuracy

    
    class _EnsembleModel:
        """
        Inner class to represent ensemble as model that is compatible with ART.

        Parameters
        ----------
        base_models: List[Callable]
            List of base models.
        weights: np.ndarray
            Ensemble weight vector.
        """
        # Initialize parameters for ensemble model compatible with ART
        def __init__(self, base_models: List[Callable], weights: np.ndarray):
            self.base_models = base_models # Store base models
            self.weights = weights / np.sum(weights) # Store normalized weights

        
        def predict(self, X):
            # TODO: Get predicted class labels for data instance X
            return

        
        def predict_proba(self, X):
            # TODO: Get ensemble model prediction probabilities for data instance X
            return