import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from pymoo.core.problem import Problem
from assembled_ensembles.util.metrics import AbstractMetric
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from sklearn.utils import check_random_state
from typing import List, Optional, Callable, Union
import logging

# Class for logging
class PredictLogger:
    def __init__(self, predict_fn, max_eval=None, name="HSJ"):
        self.predict_fn = predict_fn
        self.max_eval = max_eval
        self.name = name
        self.n_calls = 0
    def __call__(self, X):
        self.n_calls += getattr(X, "shape", (0,))[0] or 0
        if self.n_calls % 100 == 0:
            if self.max_eval:
                pct = min(100.0, 100.0 * self.n_calls / self.max_eval)
                print(f"[{self.name}] queries={self.n_calls}/{self.max_eval} ({pct:.1f}%)", flush=True)
            else:
                print(f"[{self.name}] queries={self.n_calls}", flush=True)
        return self.predict_fn(X)


class EnsembleSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible wrapper around a weighted ensemble of sklearn base models.

    Parameters
    ----------
    base_models : list
        Unpickled sklearn pipelines with 'predict_proba'.
    weights : np.ndarray
        Ensemble weights (will be normalized if not summing to 1).
    classes_ : np.ndarray
        Class labels array (as used by the base models).
    feature_names : Optional[List[str]]
        Column names used during training. If provided, X is wrapped into a DataFrame with these columns.
    """
    def __init__(self, base_models, weights, classes_, feature_names=None):
        self.base_models = base_models
        self.weights = np.asarray(weights, dtype=float)
        # sklearn expects a classes_ attribute on fitted classifiers
        self.classes_ = np.asarray(classes_) if classes_ is not None else None
        self.feature_names = list(feature_names) if feature_names is not None else None

    # Implementation according to sklearn estimator API
    def fit(self, X=None, y=None):
        """fit to satisfy sklearn’s estimator API.
        The wrapper does not train the base models, it only combines them.
        """
        # Ensure classes_ is available and valid
        if self.classes_ is None:
            raise ValueError("EnsembleSklearnWrapper requires 'classes_' to be provided.")
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self

    # Helper function
    def _ensure_X(self, X):
        if self.feature_names is not None:
            try:
                return pd.DataFrame(X, columns=self.feature_names)
            except Exception:
                return X
        return X

    def predict_proba(self, X):
        if self.classes_ is None:
            raise ValueError("'classes_' is not set; call fit() or provide classes_ in __init__.")
        X_in = self._ensure_X(X)
        probs = [bm.predict_proba(X_in) for bm in self.base_models]
        probs = np.asarray(probs)  # (n_models, n_samples, n_classes)
        w = self.weights
        if not np.isclose(w.sum(), 1.0):
            w = w / (w.sum() + 1e-12)
        avg = np.tensordot(w, probs, axes=(0, 0))  # (n_samples, n_classes)
        # keep it a valid probability simplex
        avg = np.clip(avg, 0.0, 1.0)
        s = avg.sum(axis=1, keepdims=True)
        s[s == 0.0] = 1.0
        return avg / s

    def predict(self, X):
        proba = self.predict_proba(X)
        idx = np.argmax(proba, axis=1)
        return self.classes_[idx]


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
    X: np.ndarray
        Data instances (with input features) from validation set.
    random_state: Optional[Union[int, np.random.RandomState]]
        Random state for reproducibility.
    n_jobs: int
        Number of jobs for parallel processing.
    skip_attack : bool, default=False
        Flag to skip the adversarial attack and return dummy data instead. Only for testing purposes.

    TODO: Add parameter for deciding which adversarial attack to use for robustness evaluation.
          Add parameter for final ensemble selection (value between 0 and 1 to prioritize either accuracy or robustness).
          For the first approach return the candidate in the final generation with the highest accuracy.
    """

    def __init__(self, n_base_models: int, predictions: List[np.ndarray], labels: np.ndarray,
                 score_metric: AbstractMetric, base_models: List[Callable], X: np.ndarray,
                 random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_jobs: int = -1,
                 classes_: Optional[np.ndarray] = None,
                 base_models_metadata: Optional[List[dict]] = None,
                 skip_attack: bool = False):
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
        self.X = X
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs
        self.classes_ = classes_
        self.base_models_metadata = base_models_metadata
        self.skip_attack = bool(skip_attack)



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
            # Normalize weights
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
        """
        Compute ensemble predictions by aggregating predictions of base models (using weights)

        Parameters
        ----------
        predictions: List[np.ndarray]
            Predictions from base models.
        weights: np.ndarray
            Weights for combining base models.

        Returns
        -------
        np.ndarray
            Weighted ensemble predictions.
        """
        # Compute weighted sum of base model predictions
        weighted_preds = np.tensordot(weights, predictions, axes=([0], [0]))

        return weighted_preds

    def _evaluate_robustness(self, weights: np.ndarray) -> float:
        """
        Evaluate robustness of ensemble using ART black-box attack
        Returns adversarial accuracy.
        """
        # If skip_attack flag is set to true skip attacks and return dummy data
        if getattr(self, "skip_attack", False):
            dummy_adv_accuracy = 0.5
            print("[RobustnessEval] SKIPPED adversarial attack; returning dummy adv_accuracy=0.5", flush=True)
            return float(dummy_adv_accuracy)

        # logging for ART
        import logging
        root = logging.getLogger()
        if not root.handlers:
            h = logging.StreamHandler()
            h.setLevel(logging.INFO)
            h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
            root.addHandler(h)
            root.setLevel(logging.INFO)
        for name in ("art", "art.attacks", "art.attacks.evasion"):
            logging.getLogger(name).setLevel(logging.INFO)

        # Try to recover original feature names from metadata (from fake base models)
        feature_names = None
        try:
            if self.base_models_metadata and len(self.base_models_metadata) > 0:
                meta0 = self.base_models_metadata[0]
                if isinstance(meta0, dict):
                    feature_names = (
                            meta0.get('feature_names')
                            or meta0.get('columns')
                            or meta0.get('feature_names_in_')
                    )
        except Exception:
            feature_names = None

        # Build sklearn-compatible ensemble wrapper around unpickled base models
        ensemble_model = EnsembleSklearnWrapper(
            base_models=self.base_models,
            weights=weights,
            classes_=self.classes_,
            feature_names=feature_names,
        )
        assert isinstance(ensemble_model, BaseEstimator)
        assert isinstance(ensemble_model, ClassifierMixin)

        ensemble_model.fit()

        # Enable logging
        logging.getLogger("art").setLevel(logging.INFO)
        logging.getLogger("art.attacks").setLevel(logging.INFO)
        logging.getLogger("art.attacks.evasion").setLevel(logging.INFO)

        print("[RobustnessEval] entering _evaluate_robustness")
        # Wrap with ART and run a black-box attack
        pred_with_logging = PredictLogger(ensemble_model.predict_proba, max_eval=10000, name="HSJ")
        x_min = float(np.min(self.X))
        x_max = float(np.max(self.X))
        classifier = BlackBoxClassifier(
            pred_with_logging,
            (self.X.shape[1],),
            len(self.classes_),
            clip_values=(x_min, x_max),
        )
        print(f"[RobustnessEval] clip_values set to: {classifier.clip_values}", flush=True)
        print("[RobustnessEval] built / wrapped classifier")
        print("[RobustnessEval] starting attack.generate | X shape:", getattr(self.X, 'shape', None), flush=True)
        # Instantiate adversarial attack
        attack = HopSkipJump(   # values for testing purposes
            classifier=classifier,
            targeted=False,  # untargeted, so no need to pass y
            max_iter=1,  # tune for speed/quality
            max_eval=10,
            init_eval=10,
            init_size=5,
            verbose=True, # enable built-in logging
        )
        # Generate adversarial examples and evaluate
        x_test_adv = attack.generate(x=self.X)
        print("[RobustnessEval] attack.generate DONE", flush=True)
        adv_preds = classifier.predict(x_test_adv)

        # Convert to labels if score_metric expects labels
        y_idx = np.argmax(adv_preds, axis=1)
        y_pred = self.classes_[y_idx]
        adv_accuracy = self.score_metric(self.labels, y_pred, to_loss=False, checks=False)
        return adv_accuracy
