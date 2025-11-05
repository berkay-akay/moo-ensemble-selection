import os
import sys
from assembled_ensembles.wrapper.abstract_ensemble import AbstractEnsemble
import numpy as np
# Needs to be done for bugfixing (because we had to downgrade NumPy)
import warnings as _pywarnings

if not hasattr(np, "warnings"):
    np.warnings = _pywarnings
from typing import List, Optional, Callable, Union
from assembled_ensembles.wrapper.abstract_weighted_ensemble import \
    AbstractWeightedEnsemble  # Abstract class for ensemble selection methods
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state
import pickle

# Pymoo Imports (for running NSGA-II)
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize

# Import moo problem definition for Pymoo
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from moo_ensemble_problem import MOOEnsembleProblem

# ART Imports for adversarial robustness evaluation
from art.attacks.evasion import BoundaryAttack
from art.estimators.classification import SklearnClassifier
from assembled_ask.util.metatask_base import get_metatask
import numpy as np


def _get_val_data_from_metatask(mt,
                                fold_idx):  # Currently not needed because we can just set passthrough = true to get access to the validation data
    """
    Helper to extract X_val, y_val from a MetaTask for the given fold.
    Works if MetaTask provides validation_indices or folds_indicator.
    """
    # Determine indices
    if hasattr(mt, "validation_indices") and mt.validation_indices is not None:
        idx = mt.validation_indices[fold_idx]
        idx = np.array(idx, dtype=int)
    elif hasattr(mt, "folds_indicator") and mt.folds_indicator is not None:
        idx = np.where(np.asarray(mt.folds_indicator) == fold_idx)[0]
    else:
        raise ValueError("MetaTask has neither validation_indices nor folds_indicator")

    X_all = getattr(mt, "X", None)
    y_all = getattr(mt, "ground_truth", None)
    if X_all is None or y_all is None:
        raise ValueError("MetaTask does not expose X/ground_truth. Ensure full dataset was loaded.")

    # Slice robustly for pandas or numpy
    if hasattr(X_all, "iloc"):
        X_val = X_all.iloc[idx]
    else:
        X_val = X_all[idx]

    if hasattr(y_all, "iloc"):
        y_val = y_all.iloc[idx]
    else:
        y_val = y_all[idx]

    return X_val, y_val


class MOOEnsembleSelection(AbstractWeightedEnsemble):
    supports_passthrough = True  # Declare passthrough support so AbstractEnsemble.fit can route validation data
    """
    Multi-Objective Ensemble Selection using NSGA-II and Adversarial Robustness Toolbox.

    Parameters
    ----------
    base_models: List[Callable]
        "Fake" base models, that only encapsulate the base model predictions.
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
                 n_jobs: int = -1, passthrough: bool = True) -> None:
        super().__init__(
            base_models,
            "predict_proba",
            "predict_proba",
            passthrough)
        self.n_generations = n_generations
        self.population_size = population_size
        self.score_metric = score_metric
        self.random_state = check_random_state(random_state)
        self.n_jobs = n_jobs

    def ensemble_passthrough_predict(self, X, base_model_predictions):
        """
        Passthrough prediction on raw features X. Compute base-model predictions on X
        and aggregate using learned weights.
        """
        # Ensure ensemble was fitted
        if not hasattr(self, "weights_"):
            raise RuntimeError("Ensemble not fitted; call fit before predict.")

        # Recover feature names if available
        feature_names = None
        try:
            if getattr(self, 'base_models_metadata_exists', False) and hasattr(self, 'base_models_metadata_'):
                meta0 = self.base_models_metadata_[0] if len(self.base_models_metadata_) > 0 else None
                if isinstance(meta0, dict):
                    feature_names = meta0.get('feature_names') or meta0.get('columns') or meta0.get('feature_names_in_')
        except Exception:
            feature_names = None

        X_for_models = X
        if feature_names is not None:
            try:
                import pandas as pd
                X_for_models = pd.DataFrame(X, columns=list(feature_names))
            except Exception:
                X_for_models = X

        # Per-base-model probabilities
        base_models_predictions = [bm.predict_proba(X_for_models) for bm in self.base_models]
        return self.ensemble_predict(base_models_predictions)

    def ensemble_passthrough_fit(self, X, base_model_predictions,
                                 labels) -> 'MOOEnsembleSelection':  # base_model_predictions can be ignored because we can recompute using the actual models
        """
        Loads actual base models.
        Defines optimization problem by creating an instance of MOOEnsembleProblem.
        Fits the ensemble by finding optimal weights using NSGA-II.
        Chooses optimal ensemble/weight vector based on evaluation performance.

        TODO: Include trade-off parameter to balance objectives in selection step of final ensemble

        Parameters
        ----------

        X: np.ndarray
            Training data.
        y: np.ndarray
            True labels of training data.

        Returns
        -------
        self
        """
        # Number of base models
        n_base_models = len(self.base_models)

        # # ONLY FOR DEBUGGING !!!
        # # Inspect the attributes of each base model in self.base_models
        # for idx, predictor in enumerate(self.base_models):
        #     print(f"Base model {idx + 1} attributes:")
        #     # Print all attributes of the fake base model
        #     print(dir(predictor))

        #     # If the predictor has a __dict__ attribute (custom attributes), print its contents
        #     if hasattr(predictor, '__dict__'):
        #         print("Custom attributes in __dict__:")
        #         for key, value in predictor.__dict__.items():
        #             print(f"  {key}: {value}")

        #     # If description is present, print its contents
        #     if hasattr(predictor, 'description'):
        #         print("Description attribute:")
        #         print(predictor.description)

        #     print("\n" + "="*40 + "\n")  # Separator between base models

        # Load actual base models using paths from predictor metadata
        actual_base_models = []
        for predictor in self.base_models:
            # Access model metadata from fake base model
            model_metadata = getattr(predictor, "model_metadata", None)
            if model_metadata is None:
                raise ValueError("Fake base model does not contain model_metadata")

            # Retrieve base_model_path from the metadata
            base_model_path = model_metadata.get("base_model_path")
            print("Base Model Path: ", base_model_path)  # Debugging
            if base_model_path is None:
                raise ValueError("Fake base model does not contain base_model_path")

            # Load the actual base model from the stored path
            with open(base_model_path, "rb") as f:
                base_model = pickle.load(f)
            actual_base_models.append(base_model)
            print(f"Appended Model: {type(base_model)}")  # Debugging (Model Metadata: {model_metadata})

        # Save the loaded actual base models
        self.base_models = actual_base_models

        # Wrap X with original feature names from fake model metadata
        try:
            feature_names = None
            if getattr(self, 'base_models_metadata_exists', False) and hasattr(self, 'base_models_metadata_'):
                meta0 = self.base_models_metadata_[0] if len(self.base_models_metadata_) > 0 else None
                if isinstance(meta0, dict):
                    feature_names = meta0.get('feature_names') or meta0.get('columns') or meta0.get('feature_names_in_')
        except Exception:
            feature_names = None

        X_for_models = X
        if feature_names is not None:
            try:
                import pandas as pd
                X_for_models = pd.DataFrame(X, columns=list(feature_names))
            except Exception:
                # If pandas is unavailable or columns mismatch, fall back to ndarray
                X_for_models = X

        # Use actual base models to predict on training data
        base_models_predictions = []
        for actual_base_model in actual_base_models:
            base_models_predictions.append(actual_base_model.predict_proba(X_for_models))

        # Create instance of MOOEnsembleProblem
        problem = MOOEnsembleProblem(
            n_base_models=n_base_models,
            predictions=base_models_predictions,
            labels=labels,
            score_metric=self.score_metric,
            base_models=actual_base_models,  # Use loaded base models
            X=X,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            classes_=self.classes_,
            base_models_metadata=(
                self.base_models_metadata_ if getattr(self, 'base_models_metadata_exists', False) else None),
            skip_attack=True,  # !!! Only for testing purposes
        )

        # Configure population size of NSGA-II algorithm
        algorithm = get_algorithm("nsga2", pop_size=self.population_size)

        # Run optimization with Pymoo
        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            seed=self.random_state.randint(1, 100000),
            verbose=True
        )

        # Extract Pareto front solutions
        # For now, select the solution with the lowest negative accuracy (highest accuracy)
        # Since we minimized negative accuracy, we find the index with the lowest 'F' value in the first column
        best_index = np.argmin(
            res.F[:, 0])  # Minimize negative accuracy (column 0 for accuracy, column 1 for robustness)
        best_weights = res.X[best_index]  # Get corresponding weights

        # Store normalized weights
        self.weights_ = best_weights / np.sum(best_weights)

        # Store validation loss
        self.validation_loss_ = -res.F[best_index, 0]  # Convert back to positive accuracy

        # Number of solutions evaluated per iteration (use NSGA‑II population size)
        self.iteration_batch_size_ = int(self.population_size)

        # Validation score(s) from optimization (convert −accuracy back to accuracy)
        self.val_loss_over_iterations_ = [float(-f[0]) for f in np.atleast_2d(res.F)]

        # # Store validation robustness
        # self.validation_robustness_ = res.F[best_index, 1]

        return self

    def predict_proba(self, X):
        """
        Return class probabilities for samples X using the learned ensemble weights.
        This must return probabilities (not labels).

        Parameters
        ----------
        X: np.ndarray
            Input feature matrix for which to predict probabilities.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples, n_classes) with class probabilities.
        """
        # Ensure that ensemble has been fitted and weights are available
        if not hasattr(self, "weights_"):
            raise Exception("The ensemble has not been fitted yet.")

        import pandas as pd

        # Wrap X with original feature names from fake model metadata (bugfixing)
        feature_names = None
        try:
            if getattr(self, 'base_models_metadata_exists', False) and hasattr(self, 'base_models_metadata_'):
                meta0 = self.base_models_metadata_[0] if len(self.base_models_metadata_) > 0 else None
                if isinstance(meta0, dict):
                    feature_names = meta0.get('feature_names') or meta0.get('columns') or meta0.get('feature_names_in_')
        except Exception:
            feature_names = None

        X_for_models = pd.DataFrame(X, columns=list(feature_names)) if feature_names is not None else X
        # Gather base model probability outputs on X by using the unpickled models
        base_models_predictions = [bm.predict_proba(X_for_models) for bm in self.base_models]
        # Combine base-model probabilities using ensemble weights and return probabilities
        return self.ensemble_predict_proba(base_models_predictions)

    def ensemble_predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """
        Return label predictions given base model probability predictions.
        Overrides subclass method to make sure to return labels and not probabilities.
        """
        from assembled_ensembles.wrapper.abstract_weighted_ensemble import AbstractWeightedEnsemble
        return AbstractWeightedEnsemble.ensemble_predict(self, predictions)
