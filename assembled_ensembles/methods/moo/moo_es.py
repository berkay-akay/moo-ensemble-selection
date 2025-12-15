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
    AbstractWeightedEnsemble
from assembled_ensembles.methods.moo.moo_ensemble_problem import EnsembleSklearnWrapper, PredictLogger
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
from art.estimators.classification import BlackBoxClassifier
from art.attacks.evasion import HopSkipJump


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

        # Recover feature names
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
            Validation data.
        labels: np.ndarray
            True labels of validation data.

        Returns
        -------
        self
        """
        # Number of base models
        n_base_models = len(self.base_models)

        # Load actual base models using paths from predictor metadata
        print("[MOO-ES] Start loading base models from metadata", flush=True)
        actual_base_models = []
        for predictor in self.base_models:
            # Access model metadata from fake base model
            model_metadata = getattr(predictor, "model_metadata", None)
            if model_metadata is None:
                raise ValueError("Fake base model does not contain model_metadata")

            # Retrieve base_model_path from the metadata
            base_model_path = model_metadata.get("base_model_path")
            # print("Base Model Path: ", base_model_path)  # Debugging
            if base_model_path is None:
                raise ValueError("Fake base model does not contain base_model_path")

            # Load the actual base model from the stored path
            with open(base_model_path, "rb") as f:
                base_model = pickle.load(f)
            actual_base_models.append(base_model)
            # print(f"Appended Model: {type(base_model)}")  # Debugging (Model Metadata: {model_metadata})

        print("[MOO-ES] Finished loading base models from metadata", flush=True)
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
                X_for_models = X

        # Use actual base models to predict on validation data
        base_models_predictions = []
        for actual_base_model in actual_base_models:
            base_models_predictions.append(actual_base_model.predict_proba(X_for_models))

        # Precompute adversarial union pool and caches
        print("[MOO-ES] Starting precomputation of adversarial union pool (per base model attacks)", flush=True)

        # Get feature names for consistent input schema
        feature_names_for_attack = None
        try:
            if getattr(self, 'base_models_metadata_exists', False) and hasattr(self, 'base_models_metadata_'):
                meta0 = self.base_models_metadata_[0] if len(self.base_models_metadata_) > 0 else None
                if isinstance(meta0, dict):
                    feature_names_for_attack = meta0.get('feature_names') or meta0.get('columns') or meta0.get('feature_names_in_')
        except Exception:
            feature_names_for_attack = None

        # Build per-model perturbed validation set using HopSkipJump
        X_attack_in = X
        if feature_names_for_attack is not None:
            try:
                import pandas as pd
                X_attack_in = pd.DataFrame(X, columns=list(feature_names_for_attack))
            except Exception:
                X_attack_in = X

        perturbed_sets = []  # list to store perturbed validation sets
        origin_idx = []      # origin model index (needed to use ensemble weightening during optimization)

        # Use per-feature bounds / clip-values
        X_np = np.asarray(X, dtype=np.float32)
        feat_min = X_np.min(axis=0)
        feat_max = X_np.max(axis=0)
        # Avoid zero-range features
        same = feat_max <= feat_min
        feat_max[same] = feat_min[same] + 1e-8
        print(f"[MOO-ES] Using per-feature clip_values with shapes: {feat_min.shape}, {feat_max.shape}", flush=True)

        # Define max-eval parameter for HSJ
        HSJ_MAX_EVAL=20

        for m_idx, bm in enumerate(actual_base_models):
            print(f"[MOO-ES] Generating adversarial set for base model {m_idx}", flush=True)
            wrapper = EnsembleSklearnWrapper(
                base_models=[bm],
                weights=np.array([1.0], dtype=float),
                classes_=self.classes_,
                feature_names=feature_names_for_attack,
            ).fit()
            pred_with_logging = PredictLogger(wrapper.predict_proba, max_eval=X_np.shape[0] * HSJ_MAX_EVAL, name=f"HSJ_bm{m_idx}")
            clf = BlackBoxClassifier(
                pred_with_logging,
                input_shape=(X_np.shape[1],),
                nb_classes=len(self.classes_),
                clip_values=(feat_min, feat_max),
            )
            attack = HopSkipJump(
                classifier=clf,
                targeted=False,
                max_iter=5,
                max_eval=HSJ_MAX_EVAL,
                init_eval=10,
                init_size=5,
                verbose=True
            )
            x_adv = attack.generate(x=X_np)
            perturbed_sets.append(x_adv)
            origin_idx.append(np.full(x_adv.shape[0], m_idx, dtype=int))
            print(f"[MOO-ES] Done generating adversarial set for model {m_idx}", flush=True)

        # Build union pool U and labels
        U = np.vstack(perturbed_sets) if len(perturbed_sets) > 0 else np.empty_like(X)
        # Set true labels (repeat labels once for each base model / perturbed validation set)
        U_labels = np.hstack([labels for _ in range(len(perturbed_sets))]) if len(perturbed_sets) > 0 else labels.copy()
        # Store base model index (could be used for origin-aware weighting later on)
        U_origin = np.hstack(origin_idx) if len(origin_idx) > 0 else np.zeros(len(labels), dtype=int)
        print(f"[MOO-ES] Built union pool U with shape {getattr(U, 'shape', None)}", flush=True)

        # Cache base model predictions on U
        adv_union_predictions = []  # will become (n_models, n_union, n_classes)

        # Ensure schema (feature names)
        U_for_models = U
        if feature_names_for_attack is not None:
            try:
                import pandas as pd
                U_for_models = pd.DataFrame(U, columns=list(feature_names_for_attack))
            except Exception:
                U_for_models = U
        for j, bm in enumerate(actual_base_models):
            print(f"[MOO-ES] Caching predict_proba on U for base model {j}", flush=True)
            adv_union_predictions.append(bm.predict_proba(U_for_models))
        adv_union_predictions = np.asarray(adv_union_predictions)
        print(f"[MOO-ES] Cached adv_union_predictions with shape {adv_union_predictions.shape}", flush=True)

        # Create instance of MOOEnsembleProblem (use surrogate robustness via adv caches)
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
            adv_union_predictions=adv_union_predictions,
            adv_union_labels=U_labels,
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

        # Extract Pareto front weights and metrics
        pareto_weights = np.atleast_2d(res.X)
        pareto_obj = np.atleast_2d(res.F)
        pareto_acc = (-pareto_obj[:, 0]).astype(float)
        pareto_robust_surr = (-pareto_obj[:, 1]).astype(float)

        print(f"[MOO-ES] Number of Pareto-optimal ensembles: {len(pareto_weights)}", flush=True)
        for i, (a, r) in enumerate(zip(pareto_acc, pareto_robust_surr)):
            print(f"[MOO-ES] Pareto-optimal ensemble (surrogate) idx={i} | accuracy={a:.6f} | adv_acc_surr={r:.6f}", flush=True)

        # Re-attack Pareto-optimal ensembles to obtain true robustness
        print("[MOO-ES] Re-attacking Pareto-optimal ensembles to compute true robustness", flush=True)
        true_robust_list = []
        for i, w in enumerate(pareto_weights):
            w = np.asarray(w, dtype=float)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()
            print(f"[MOO-ES] Re-attack ensemble {i} ...", flush=True)
            try:
                adv_acc_true = float(problem._evaluate_robustness(w))
            except Exception as e:
                print(f"[MOO-ES] WARNING: re-attack failed for candidate {i}: {e}", flush=True)
                adv_acc_true = float('nan')
            true_robust_list.append(adv_acc_true)
            print(f"[MOO-ES] true_adv_accuracy={adv_acc_true}", flush=True)
        pareto_robust_true = np.asarray(true_robust_list, dtype=float)

        # Select final solution by highest accuracy
        best_index = int(np.argmax(pareto_acc))
        best_weights = pareto_weights[best_index]

        # Store normalized weights
        self.weights_ = best_weights / np.sum(best_weights)

        # Store validation loss (here it is actually the selected accuracy)
        self.validation_loss_ = float(pareto_acc[best_index])

        # Number of solutions evaluated per iteration (use NSGA‑II population size)
        self.iteration_batch_size_ = int(self.population_size)

        # Validation score(s) from optimization (convert −accuracy back to accuracy)
        self.val_loss_over_iterations_ = [float(a) for a in pareto_acc]

        # Store (true) robustness for the selected solution
        self.validation_robustness_ = float(pareto_robust_true[best_index])

        # Store metadata (is saved to disk under benchmark/output)
        self.model_specific_metadata_.update(dict(
            selection_rule="by_accuracy",
            surrogate_pool_size=int(len(pareto_robust_surr) and getattr(adv_union_predictions, 'shape', [0,0,0])[1] or 0),
            pareto_accuracy=[float(v) for v in pareto_acc.tolist()],
            pareto_surrogate_robustness=[float(v) for v in pareto_robust_surr.tolist()],
            pareto_true_robustness=[None if np.isnan(v) else float(v) for v in pareto_robust_true.tolist()],
            selected_index=int(best_index),
            selected_accuracy=float(pareto_acc[best_index]),
            selected_robustness_true=None if np.isnan(pareto_robust_true[best_index]) else float(pareto_robust_true[best_index]),
        ))
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
