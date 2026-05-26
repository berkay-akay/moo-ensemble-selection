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
from assembled_ensembles.methods.moo.moo_ensemble_problem import (
    MOOEnsembleProblem,
    EnsembleSklearnWrapper,
    run_permute_attack_single_instance,
)
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state
import pickle

# Pymoo Imports (for running NSGA-II)
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import time

try:
    from pymoo.performance_indicator.hv import Hypervolume as _HV
except Exception:  # older/newer pymoo or missing module
    _HV = None

# ART Imports for adversarial robustness evaluation
from assembled_ask.util.metatask_base import get_metatask

from assembled.metatask import MetaTask
from pathlib import Path

from assembled.metatask import MetaTask
from pathlib import Path
import json
import re

def _get_train_data_from_metatask_and_fold(mt, fold_idx):
    """
    Extract X_train, y_train from a fully loaded MetaTask using fold_idx.
    Uses validation_indices if available; otherwise falls back to folds_indicator.
    """
    X_all = getattr(mt, "dataset", None)
    y_all = getattr(mt, "ground_truth", None)

    if X_all is None or y_all is None:
        raise ValueError(
            f"MetaTask does not expose usable feature data / labels. "
            f"type(dataset)={type(getattr(mt, 'dataset', None))}, "
            f"type(ground_truth)={type(getattr(mt, 'ground_truth', None))}"
        )

    # Keep only the true input feature columns, never the target column
    if hasattr(X_all, "loc"):
        cols = None

        # Prefer non-categorical feature names because you only run numeric datasets
        if hasattr(mt, "non_cat_feature_names") and mt.non_cat_feature_names is not None:
            cols = list(mt.non_cat_feature_names)

        # Fallback: use feature_names, but explicitly remove target_name if present
        elif hasattr(mt, "feature_names") and mt.feature_names is not None:
            cols = list(mt.feature_names)

        if cols is not None:
            target_name = getattr(mt, "target_name", None)
            if target_name in cols:
                cols.remove(target_name)

            missing_cols = [c for c in cols if c not in X_all.columns]
            if missing_cols:
                raise ValueError(
                    f"Some feature columns are missing in mt.dataset: {missing_cols}. "
                    f"Available columns: {list(X_all.columns)}"
                )

            X_all = X_all.loc[:, cols]

    print(f"[MOO-ES] X_all columns used for attack: {list(X_all.columns)}", flush=True)
    print(f"[MOO-ES] X_all shape after feature subsetting: {X_all.shape}", flush=True)

    if hasattr(mt, "validation_indices") and mt.validation_indices is not None and len(mt.validation_indices) > 0:
        vi = mt.validation_indices

        if isinstance(vi, dict):
            if fold_idx in vi:
                val_idx = np.array(vi[fold_idx], dtype=int)
            elif str(fold_idx) in vi:
                val_idx = np.array(vi[str(fold_idx)], dtype=int)
            else:
                raise KeyError(
                    f"fold_idx={fold_idx} not found in mt.validation_indices. "
                    f"Available keys: {list(vi.keys())}"
                )
        else:
            val_idx = np.array(vi[fold_idx], dtype=int)

        n_samples = len(X_all) if hasattr(X_all, "__len__") else X_all.shape[0]
        all_idx = np.arange(n_samples, dtype=int)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)

    elif hasattr(mt, "folds_indicator") and mt.folds_indicator is not None:
        train_idx = np.where(np.asarray(mt.folds_indicator) != fold_idx)[0]

    else:
        raise ValueError("MetaTask has neither usable validation_indices nor folds_indicator")

    if hasattr(X_all, "iloc"):
        X_train = X_all.iloc[train_idx]
    else:
        X_train = X_all[train_idx]

    if hasattr(y_all, "iloc"):
        y_train = y_all.iloc[train_idx]
    else:
        y_train = y_all[train_idx]

    return X_train, y_train

def _load_full_metatask_from_benchmark_input(openml_task_id, benchmark_name, pruner="SiloTopN", delayed_evaluation_load=False):
    """
    Load the full MetaTask from benchmark/input exactly like run_evaluate_ensemble_on_metatask.py.
    """
    file_path = Path(os.path.dirname(os.path.abspath(__file__)))
    repo_root = file_path.parents[3]  # .../moo-ensemble-selection
    tmp_input_dir = repo_root / "moo-ensemble-selection" / "benchmark" / "input" / benchmark_name / pruner

    print(f"[MOO-ES] Loading full MetaTask from: {tmp_input_dir}", flush=True)

    mt = MetaTask()
    mt.read_metatask_from_files(tmp_input_dir, str(openml_task_id), delayed_evaluation_load=delayed_evaluation_load)

    #print(f"[MOO-ES] Loaded MetaTask. Has X: {getattr(mt, 'X', None) is not None}, "
    #      f"has ground_truth: {getattr(mt, 'ground_truth', None) is not None}", flush=True)

    return mt

def _get_train_data_from_passed_validation(mt, X_val_passed):
    """
    Reconstruct X_train, y_train from the full MetaTask data and the validation
    split that is already passed into ensemble_passthrough_fit via passthrough.

    Preferred behavior:
    1) If indices are preserved, use index complement.
    2) Otherwise, fall back to row-wise matching by feature values.
    """
    X_all = getattr(mt, "X", None)
    y_all = getattr(mt, "ground_truth", None)
    if X_all is None or y_all is None:
        raise ValueError("MetaTask does not expose X/ground_truth. Ensure full dataset was loaded.")

    # ------------------------------------------------------------
    # Case 1: index-preserving pandas objects
    # ------------------------------------------------------------
    if hasattr(X_all, "index") and hasattr(X_val_passed, "index"):
        try:
            val_index = X_val_passed.index
            train_mask = ~X_all.index.isin(val_index)

            X_train = X_all.loc[train_mask]
            y_train = y_all.loc[train_mask] if hasattr(y_all, "loc") else y_all[train_mask]
            return X_train, y_train
        except Exception:
            pass

    # ------------------------------------------------------------
    # Case 2: fallback to row-wise matching by values
    # ------------------------------------------------------------
    X_all_np = X_all.to_numpy() if hasattr(X_all, "to_numpy") else np.asarray(X_all)
    X_val_np = X_val_passed.to_numpy() if hasattr(X_val_passed, "to_numpy") else np.asarray(X_val_passed)

    if X_all_np.ndim != 2 or X_val_np.ndim != 2:
        raise ValueError("Expected 2D feature matrices for X_all and X_val_passed.")

    n_all = X_all_np.shape[0]
    train_mask = np.ones(n_all, dtype=bool)

    # Mark the first matching occurrence in X_all for every validation row as validation
    # so that duplicates are handled conservatively.
    used = np.zeros(n_all, dtype=bool)

    for row in X_val_np:
        row_matches = np.where(np.all(np.isclose(X_all_np, row, rtol=1e-8, atol=1e-10, equal_nan=True), axis=1))[0]
        row_matches = row_matches[~used[row_matches]]

        if len(row_matches) == 0:
            raise ValueError("Could not match a passed validation row back to mt.X while reconstructing training data.")

        match_idx = row_matches[0]
        used[match_idx] = True
        train_mask[match_idx] = False

    if hasattr(X_all, "iloc"):
        X_train = X_all.iloc[train_mask]
    else:
        X_train = X_all[train_mask]

    if hasattr(y_all, "iloc"):
        y_train = y_all.iloc[train_mask]
    else:
        y_train = y_all[train_mask]

    return X_train, y_train

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


def _get_train_data_from_metatask(mt, fold_idx):
    """
    Helper to extract X_train, y_train from a MetaTask for the given fold.
    It computes the training indices as the complement of the validation/test indices for that fold.

    This works if MetaTask provides either:
      - validation_indices: a list/array with validation indices per fold, or
      - folds_indicator: an array marking which fold each sample belongs to (as the validation/test fold).
    """
    # Determine training indices
    if hasattr(mt, "validation_indices") and mt.validation_indices is not None:
        vi = mt.validation_indices

        if isinstance(vi, dict):
            if fold_idx in vi:
                val_idx = np.array(vi[fold_idx], dtype=int)
            elif str(fold_idx) in vi:
                val_idx = np.array(vi[str(fold_idx)], dtype=int)
            else:
                raise KeyError(
                    f"fold_idx={fold_idx} not found in mt.validation_indices. "
                    f"Available keys: {list(vi.keys())}"
                )
        else:
            val_idx = np.array(vi[fold_idx], dtype=int)
        # Build complement over all samples
        X_all = getattr(mt, "X", None)
        if X_all is None:
            raise ValueError("MetaTask does not expose X. Ensure full dataset was loaded before calling this.")
        n_samples = len(X_all) if hasattr(X_all, "__len__") else X_all.shape[0]
        all_idx = np.arange(n_samples, dtype=int)
        train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)
    elif hasattr(mt, "folds_indicator") and mt.folds_indicator is not None:
        # All samples that are NOT assigned to the current fold are considered training
        train_idx = np.where(np.asarray(mt.folds_indicator) != fold_idx)[0]
    else:
        raise ValueError("MetaTask has neither validation_indices nor folds_indicator")

    X_all = getattr(mt, "X", None)
    y_all = getattr(mt, "ground_truth", None)
    if X_all is None or y_all is None:
        raise ValueError("MetaTask does not expose X/ground_truth. Ensure full dataset was loaded.")

    # Slice robustly for pandas or numpy
    if hasattr(X_all, "iloc"):
        X_train = X_all.iloc[train_idx]
    else:
        X_train = X_all[train_idx]

    if hasattr(y_all, "iloc"):
        y_train = y_all.iloc[train_idx]
    else:
        y_train = y_all[train_idx]

    return X_train, y_train


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
    reattack_top3_only: bool, default=True
        When set to True, only the top-3 ensembles (based on clean accuracy) in the pareto front will be re-attacked.
        When set to False, all pareto-optimal ensembles will be re-attacked.
    TODO: Add parameter for final ensemble selection (value between 0 and 1 to prioritize either accuracy or robustness).
    """

    # Initialize parameters for MOO Ensemble Selection
    def __init__(self, base_models: List[Callable], n_generations: int, population_size: int,
                 score_metric: AbstractMetric, random_state: Optional[Union[int, np.random.RandomState]] = None,
                 n_jobs: int = -1, passthrough: bool = True, reattack_top3_only: bool = True,
                 permute_attack_kwargs: Optional[dict] = None) -> None:
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
        self.reattack_top3_only = reattack_top3_only

        default_permute_attack_kwargs = dict(
            sol_per_pop=35,
            num_parents_mating=15,
            num_generations=100,
            n_runs=1,
            beta=0.96,
            black_list=None,
            verbose=False,
            target=None,
        )
        if permute_attack_kwargs is not None:
            default_permute_attack_kwargs.update(permute_attack_kwargs)
        self.permute_attack_kwargs = default_permute_attack_kwargs

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
        _t0_total = time.perf_counter()

        # --- EARLY EXIT IF DATASET HAS CATEGORICAL FEATURES (via metatask_schema.json) ---
        import json

        md0 = getattr(self.base_models[0], "model_metadata", None)

        if not isinstance(md0, dict):
            raise ValueError("Base model[0] lacks 'model_metadata' dict; cannot locate metatask_schema.json")



        # Prefer values directly from metadata (if present)
        dataset_name = md0.get("dataset_name")
        openml_task_id = md0.get("openml_task_id")

        # Fallback: parse the metatask JSON
        if dataset_name is None or openml_task_id is None:
            mt_path = md0.get("metatask_json_path") or md0.get("metatask_path")
            if not mt_path:
                raise ValueError("Missing dataset_name/openml_task_id in metadata and no metatask path to infer them")
            with open(mt_path, "r", encoding="utf-8") as f:
                mt_json = json.load(f)
            dataset_name = dataset_name or mt_json.get("dataset_name")
            openml_task_id = openml_task_id or mt_json.get("openml_task_id")
            if dataset_name is None or openml_task_id is None:
                raise ValueError("Could not infer dataset_name/openml_task_id from metatask JSON")

        # Build the canonical path to the schema file in benchmark/output
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        schema_path = os.path.join(
            repo_root,
            "benchmark", "output", str(dataset_name), f"task_{int(openml_task_id)}",
            "metatask_schema.json",
        )
        if not os.path.isfile(schema_path):
            raise FileNotFoundError(f"metatask_schema.json not found at: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)

        cat_feature_names = schema.get("cat_feature_names") or []
        if len(cat_feature_names) > 0:
            self.model_specific_metadata_.setdefault('aborted_reason', 'categorical_features_unsupported')
            self.model_specific_metadata_['cat_feature_names'] = list(cat_feature_names)
            # Raise to abort the fold cleanly -> Skip scoring
            raise RuntimeError("Datasets with categorical input features are not supported yet for MOO-ES")
        # -------------------------------------------------------------------------------
        base_model_path = md0.get("base_model_path")
        if not isinstance(base_model_path, str):
            raise ValueError("Cannot infer fold_idx: model_metadata['base_model_path'] is missing or not a string.")

        m = re.search(r"/fold_(\d+)(?:/|$)", base_model_path)
        if m is None:
            raise ValueError(f"Cannot infer fold_idx from base_model_path: {base_model_path!r}")

        fold_idx = int(m.group(1))
        print(f"[MOO-ES] Inferred fold_idx={fold_idx} from base_model_path.", flush=True)

        mt = _load_full_metatask_from_benchmark_input(
            openml_task_id=openml_task_id,
            benchmark_name=dataset_name,
            pruner="SiloTopN",
            delayed_evaluation_load=False,
        )

        # DEBUG PRINTS
        #print(f"[MOO-ES] type(mt.X) = {type(getattr(mt, 'X', None))}", flush=True)
        #print(f"[MOO-ES] type(mt.ground_truth) = {type(getattr(mt, 'ground_truth', None))}", flush=True)
        #print(f"[MOO-ES] type(mt.dataset_frame) = {type(getattr(mt, 'dataset_frame', None))}", flush=True)
        #print(f"[MOO-ES] type(mt.data_frame) = {type(getattr(mt, 'data_frame', None))}", flush=True)
        #print(f"[MOO-ES] mt attributes sample = {[a for a in dir(mt) if not a.startswith('_')][:80]}", flush=True)

        #print(f"[MOO-ES] type(mt.dataset) = {type(getattr(mt, 'dataset', None))}", flush=True)
        #print(f"[MOO-ES] type(mt.meta_dataset) = {type(getattr(mt, 'meta_dataset', None))}", flush=True)

        ds = getattr(mt, "dataset", None)
        mds = getattr(mt, "meta_dataset", None)

        if ds is not None:
            try:
                print(f"[MOO-ES] mt.dataset shape = {ds.shape}", flush=True)
            except Exception:
                print("[MOO-ES] mt.dataset has no shape attribute", flush=True)

        if mds is not None:
            try:
                print(f"[MOO-ES] mt.meta_dataset shape = {mds.shape}", flush=True)
            except Exception:
                print("[MOO-ES] mt.meta_dataset has no shape attribute", flush=True)

        # DEBUG PRINTS END

        #print(f"[MOO-ES] validation_indices type: {type(mt.validation_indices)}", flush=True)
        #if isinstance(mt.validation_indices, dict):
        #    print(f"[MOO-ES] validation_indices keys: {list(mt.validation_indices.keys())}", flush=True)

        X_train_attack, y_train_attack = _get_train_data_from_metatask_and_fold(mt, fold_idx)

        if hasattr(X_train_attack, "to_numpy"):
            X_train_attack = X_train_attack.to_numpy()
        X_train_attack = np.asarray(X_train_attack, dtype=np.float32)

        self.model_specific_metadata_["fold_idx"] = int(fold_idx)
        self.model_specific_metadata_["train_shape"] = tuple(X_train_attack.shape)
        # ------------------------------------------------------------


        # Number of base models
        n_base_models = len(self.base_models)

        # Load actual base models using paths from predictor metadata
        print("[MOO-ES] Start loading base models from metadata", flush=True)
        actual_base_models = []
        for predictor in self.base_models:
            model_metadata = getattr(predictor, "model_metadata", None)
            if model_metadata is None:
                raise ValueError("Fake base model does not contain model_metadata")
            base_model_path = model_metadata.get("base_model_path")
            if base_model_path is None:
                raise ValueError("Fake base model does not contain base_model_path")
            with open(base_model_path, "rb") as f:
                base_model = pickle.load(f)
            actual_base_models.append(base_model)
        print("[MOO-ES] Finished loading base models from metadata", flush=True)
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

        # --------------------- Precompute adversarial union pool (timed) ---------------------
        _t0_pool = time.perf_counter()
        print("[MOO-ES] Starting precomputation of adversarial union pool (per base model attacks)", flush=True)

        feature_names_for_attack = None
        try:
            if getattr(self, 'base_models_metadata_exists', False) and hasattr(self, 'base_models_metadata_'):
                meta0 = self.base_models_metadata_[0] if len(self.base_models_metadata_) > 0 else None
                if isinstance(meta0, dict):
                    feature_names_for_attack = meta0.get('feature_names') or meta0.get('columns') or meta0.get('feature_names_in_')
        except Exception:
            feature_names_for_attack = None

        X_attack_in = X
        if feature_names_for_attack is not None:
            try:
                import pandas as pd
                X_attack_in = pd.DataFrame(X, columns=list(feature_names_for_attack))
            except Exception:
                X_attack_in = X

        perturbed_sets = []  # list to store perturbed validation sets
        origin_idx = []      # origin model index (needed to use ensemble weightening during optimization)
        adv_success_masks = []  # true only where a real adversarial copy was found
        base_model_attack_stats = [] # storing success / failure stats for each base model

        X_np = np.asarray(X, dtype=np.float32)

        for m_idx, bm in enumerate(actual_base_models):
            print(f"[MOO-ES] Generating adversarial set for base model {m_idx}", flush=True)

            model_type = type(bm).__name__

            try:
                if hasattr(bm, "steps") and len(bm.steps) > 0:
                    last_step = bm.steps[-1][1]

                    if hasattr(last_step, "choice") and last_step.choice is not None:
                        model_type = type(last_step.choice).__name__
                    else:
                        model_type = type(last_step).__name__

            except Exception:
                pass

            print(
                f"[MOO-ES] Base model {m_idx}: extracted model_type={model_type}",
                flush=True,
            )

            attack_classes = np.arange(len(self.classes_))

            wrapper = EnsembleSklearnWrapper(
                base_models=[bm],
                weights=np.array([1.0], dtype=float),
                classes_=attack_classes,
                feature_names=feature_names_for_attack,
            ).fit()

            # ------------------------------------------------------------
            # Step 1: determine which validation instances are clean-correct for this base model
            # ------------------------------------------------------------
            clean_proba = wrapper.predict_proba(X_attack_in)  # shape (N, C)
            clean_pred = np.argmax(clean_proba, axis=1)
            clean_mask = (clean_pred == np.asarray(labels))
            n_clean_correct = int(np.sum(clean_mask))
            n_total = int(X_np.shape[0])

            print(
                f"[MOO-ES] Base model {m_idx}: clean-correct = {n_clean_correct}/{n_total}",
                flush=True,
            )

            # Start from the clean validation set as fallback for all instances
            x_adv_full = X_np.copy()

            # If there are no clean-correct instances, keep the clean copy block
            if n_clean_correct == 0:
                print(
                    f"[MOO-ES] Base model {m_idx}: no clean-correct instances. "
                    f"Using clean validation copy as adversarial fallback block.",
                    flush=True,
                )

                success_mask_full = np.zeros(x_adv_full.shape[0], dtype=bool)

                perturbed_sets.append(x_adv_full)
                origin_idx.append(np.full(x_adv_full.shape[0], m_idx, dtype=int))
                adv_success_masks.append(success_mask_full)

                print(f"[MOO-ES] Done generating adversarial set for model {m_idx}", flush=True)
                continue

            # Attack only the clean-correct subset using PermuteAttack
            X_sub = X_np[clean_mask]

            print(
                f"[MOO-ES] Base model {m_idx}: attacking only clean-correct subset "
                f"with shape={X_sub.shape} using PermuteAttack",
                flush=True,
            )

            clean_indices = np.where(clean_mask)[0]
            labels_np = np.asarray(labels)
            n_failed = 0
            n_success = 0
            success_mask_full = np.zeros(x_adv_full.shape[0], dtype=bool)

            # DEBUG PRINTS
            #print(
            #    f"[MOO-ES] Wrapper classes_ dtype={np.asarray(wrapper.classes_).dtype}, "
            #    f"classes_={wrapper.classes_}",
            #    flush=True,
            #)

            for local_counter, orig_idx in enumerate(clean_indices, start=1):

                if local_counter % 50 == 0 or local_counter == len(clean_indices):
                    print(
                        f"[MOO-ES] Base model {m_idx}: PermuteAttack progress "
                        f"{local_counter}/{len(clean_indices)} instances",
                        flush=True,
                    )

                x_i = X_np[orig_idx]

                x_adv_i, fail_reason = run_permute_attack_single_instance(
                    estimator=wrapper,
                    x_i=x_i,
                    x_train=X_train_attack,
                    true_label=labels_np[orig_idx],
                    feature_names=feature_names_for_attack,
                    **self.permute_attack_kwargs,
                )

                if x_adv_i is None:
                    n_failed += 1
                    #print(
                    #    f"[MOO-ES] Base model {m_idx}: PermuteAttack failed on instance {orig_idx} "
                    #    f"with reason={fail_reason}; keeping clean fallback.",
                    #    flush=True,
                    #)
                    continue

                x_adv_full[orig_idx] = x_adv_i[0]
                success_mask_full[orig_idx] = True
                n_success += 1

            print(
                f"[MOO-ES] PermuteAttack finished for base model {m_idx} | "
                f"success={n_success}, failed={n_failed}",
                flush=True,
            )

            success_rate = float(n_success / max(1, n_clean_correct))

            # Store success / failure stats for this base model
            base_model_attack_stats.append({
                "base_model_id": int(m_idx),
                "base_model_type": str(model_type),
                "n_clean_correct": int(n_clean_correct),
                "n_success": int(n_success),
                "n_failed": int(n_failed),
                "success_rate": success_rate,
            })

            perturbed_sets.append(x_adv_full)
            origin_idx.append(np.full(x_adv_full.shape[0], m_idx, dtype=int))
            adv_success_masks.append(success_mask_full)

            print(
                f"[MOO-ES] Done generating adversarial set for model {m_idx} "
                f"(full block shape={x_adv_full.shape})",
                flush=True,
            )

        if len(perturbed_sets) > 0:
            U = np.vstack(perturbed_sets)
            U_labels = np.hstack([labels for _ in range(len(perturbed_sets))])
            adv_success_mask = np.hstack(adv_success_masks)
        else:
            X_np = np.asarray(X, dtype=np.float32)
            U = np.empty((0, X_np.shape[1]), dtype=X_np.dtype)
            U_labels = np.empty((0,), dtype=np.asarray(labels).dtype)
            adv_success_mask = np.empty((0,), dtype=bool)

        print(f"[MOO-ES] Built union pool U with shape {getattr(U, 'shape', None)}", flush=True)

        # Build cached predictions for adversarial union pool
        if U.shape[0] == 0:
            adv_union_predictions = np.zeros((len(actual_base_models), 0, len(self.classes_)), dtype=float)
        else:
            adv_union_predictions = []
            U_for_models = U
            if feature_names_for_attack is not None:
                try:
                    import pandas as pd
                    U_for_models = pd.DataFrame(U, columns=list(feature_names_for_attack))
                except Exception:
                    U_for_models = U

            print(f"[MOO-ES] Caching predict_proba on U for base models")
            for j, bm in enumerate(actual_base_models):
                adv_union_predictions.append(bm.predict_proba(U_for_models))
            adv_union_predictions = np.asarray(adv_union_predictions)
        print(f"[MOO-ES] Cached adv_union_predictions with shape {getattr(adv_union_predictions, 'shape', None)}", flush=True)
        _t1_pool = time.perf_counter()

        # Build problem
        problem = MOOEnsembleProblem(
            n_base_models=n_base_models,
            predictions=base_models_predictions,
            labels=labels,
            score_metric=self.score_metric,
            base_models=actual_base_models,
            X=X,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            classes_=self.classes_,
            base_models_metadata=(
                self.base_models_metadata_ if getattr(self, 'base_models_metadata_exists', False) else None),
            adv_union_predictions=adv_union_predictions,
            adv_union_labels=U_labels,
            X_train_attack=X_train_attack,
            permute_attack_kwargs=self.permute_attack_kwargs,
        )

        algorithm = get_algorithm("nsga2", pop_size=self.population_size, eliminate_duplicates=True)

        # --------------------- Optimization with per-generation callback (timed) ---------------------
        self.model_specific_metadata_.setdefault('per_generation_stats', [])
        _run_start = time.perf_counter()
        _t0_opt = time.perf_counter()

        nds = NonDominatedSorting(method="fast_non_dominated_sort")

        _hv_ref = None

        def _unique_ratio(X, tol: float = 1e-12):
            try:
                X = np.asarray(X, dtype=float)
                if X.ndim != 2 or X.shape[0] == 0:
                    return None
                # Use rounding relative to scale to merge near-duplicates
                max_abs = float(np.nanmax(np.abs(X))) if np.isfinite(X).any() else 1.0
                scale = max(1.0, max_abs)
                Q = np.round(X / (scale * tol))
                if Q.size == 0:
                    return None
                # bytes view for row-unique
                try:
                    uniq = np.unique(Q.view([('', Q.dtype)] * Q.shape[1]))
                    return float(len(uniq)) / float(X.shape[0])
                except Exception:
                    # Fallback: slower path
                    rows = {tuple(row) for row in Q}
                    return float(len(rows)) / float(X.shape[0])
            except Exception:
                return None

        def _per_gen_callback(algorithm): # Helper method to capture/store per-generation stats
            nonlocal _hv_ref
            try:
                gen = int(getattr(algorithm, 'n_gen', 0))
                F = algorithm.pop.get("F")
                if F is None or len(F) == 0:
                    return
                F = np.asarray(F, dtype=float)
                acc = -F[:, 0]
                rob_surr = -F[:, 1]

                # Nondominated count for current population
                nd_idx = nds.do(F, only_non_dominated_front=True)
                n_nondom = int(len(nd_idx))

                # Crowding statistics (if available)
                crowding_mean = None
                crowding_max = None
                try:
                    crowd = algorithm.pop.get('crowding')
                    if crowd is None:
                        crowd = algorithm.pop.get('CV')  # try alternative attribute
                    if crowd is not None:
                        crowd = np.asarray(crowd, dtype=float)
                        if crowd.size:
                            crowding_mean = float(np.nanmean(crowd))
                            crowding_max = float(np.nanmax(crowd))
                except Exception:
                    pass

                # Objective correlation
                try:
                    if np.std(acc) > 0 and np.std(rob_surr) > 0:
                        obj_corr = float(np.corrcoef(acc, rob_surr)[0, 1])
                    else:
                        obj_corr = None
                except Exception:
                    obj_corr = None

                # Unique weights ratio
                try:
                    Xpop = algorithm.pop.get('X')
                    unique_weights_ratio = _unique_ratio(Xpop)
                except Exception:
                    unique_weights_ratio = None

                # Hypervolume using a fixed reference point (lazily set to dominate current pop)
                hv_val = None
                try:
                    if _HV is not None and np.isfinite(F).all():
                        if _hv_ref is None:
                            ref = np.nanmax(F, axis=0)
                            _hv_ref = ref + 1e-6
                        hv = _HV(ref_point=_hv_ref)
                        hv_val = float(hv.do(F))
                except Exception:
                    hv_val = None

                # Placeholders for unavailable or instrumented metrics
                offspring_from_crossover = None
                offspring_from_mutation = None
                offspring_from_elite = None
                selection_pressure = None
                igd = None
                gd = None

                stats = dict(
                    gen=gen,
                    pop_size=int(F.shape[0]),
                    acc_min=float(np.nanmin(acc)), acc_mean=float(np.nanmean(acc)), acc_max=float(np.nanmax(acc)),
                    rob_min=float(np.nanmin(rob_surr)), rob_mean=float(np.nanmean(rob_surr)), rob_max=float(np.nanmax(rob_surr)),
                    n_nondominated=n_nondom,
                    t_sec=float(time.perf_counter() - _run_start),
                    # New guarded fields
                    crowding_mean=crowding_mean,
                    crowding_max=crowding_max,
                    obj_corr=obj_corr,
                    unique_weights_ratio=unique_weights_ratio,
                    hypervolume=hv_val,
                    offspring_from_crossover=offspring_from_crossover,
                    offspring_from_mutation=offspring_from_mutation,
                    offspring_from_elite=offspring_from_elite,
                    selection_pressure=selection_pressure,
                    igd=igd,
                    gd=gd,
                )
                self.model_specific_metadata_['per_generation_stats'].append(stats)
            except Exception as _e:
                print(f"[MOO-ES] WARNING: per-generation callback failed: {_e}", flush=True)
                return

        res = minimize(
            problem,
            algorithm,
            ('n_gen', self.n_generations),
            seed=self.random_state.randint(1, 100000),
            verbose=True,
            callback=_per_gen_callback,
        )
        _t1_opt = time.perf_counter()

        # --------------------- Extract Pareto and re-attack (timed) ---------------------
        pareto_weights = np.atleast_2d(res.X)
        pareto_obj = np.atleast_2d(res.F)
        pareto_acc = (-pareto_obj[:, 0]).astype(float)
        pareto_robust_surr = (-pareto_obj[:, 1]).astype(float)

        print(f"[MOO-ES] Number of Pareto-optimal ensembles: {len(pareto_weights)}", flush=True)
        for i, (a, r) in enumerate(zip(pareto_acc, pareto_robust_surr)):
            print(f"[MOO-ES] Pareto-optimal ensemble (surrogate) idx={i} | accuracy={a:.6f} | adv_acc_surr={r:.6f}", flush=True)

        if self.reattack_top3_only:
            top_k = min(3, len(pareto_weights))
            reattack_idx = np.argsort(-pareto_acc)[:top_k]
            print(
                f"[MOO-ES] Re-attacking only top {top_k} Pareto-optimal ensembles by accuracy: "
                f"{reattack_idx.tolist()}",
                flush=True,
            )
        else:
            reattack_idx = np.arange(len(pareto_weights))
            print("[MOO-ES] Re-attacking all Pareto-optimal ensembles to compute true robustness", flush=True)

        _t0_reattack = time.perf_counter()

        pareto_robust_true = np.full(len(pareto_weights), np.nan, dtype=float)

        pareto_attack_stats = []

        for i in reattack_idx:
            w = np.asarray(pareto_weights[i], dtype=float)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / w.sum()

            print(f"[MOO-ES] Re-attack ensemble {i} ...", flush=True)
            try:
                attack_result = problem._evaluate_robustness(w)
                adv_acc_true = float(attack_result["adv_accuracy"])
            except Exception as e:
                print(f"[MOO-ES] WARNING: re-attack failed for candidate {i}: {e}", flush=True)
                attack_result = {
                    "adv_accuracy": float("nan"),
                    "n_clean_correct": None,
                    "n_success": None,
                    "n_failed": None,
                    "success_rate": None,
                }
                adv_acc_true = float("nan")

            pareto_robust_true[i] = adv_acc_true

            pareto_attack_stats.append({
                "pareto_index": int(i),
                "adv_accuracy": None if np.isnan(adv_acc_true) else float(adv_acc_true),
                "n_clean_correct": attack_result["n_clean_correct"],
                "n_success": attack_result["n_success"],
                "n_failed": attack_result["n_failed"],
                "success_rate": attack_result["success_rate"],
            })

            print(f"[MOO-ES] true_adv_accuracy={adv_acc_true}", flush=True)

        _t1_reattack = time.perf_counter()

        # ------------------------------------------------------------
        # Select the best re-attacked Pareto ensemble by true robustness
        # ------------------------------------------------------------
        valid_true_robustness_mask = np.isfinite(pareto_robust_true)

        if np.any(valid_true_robustness_mask):
            valid_indices = np.where(valid_true_robustness_mask)[0]

            # Choose the re-attacked Pareto solution with highest true adversarial robustness.
            # Tie-breaker: higher clean accuracy.
            robust_rank_order = sorted(
                valid_indices,
                key=lambda idx: (pareto_robust_true[idx], pareto_acc[idx]),
                reverse=True,
            )

            best_true_robust_index = int(robust_rank_order[0])
            best_true_robust_accuracy = float(pareto_acc[best_true_robust_index])
            best_true_robust_robustness = float(pareto_robust_true[best_true_robust_index])
        else:
            best_true_robust_index = None
            best_true_robust_accuracy = None
            best_true_robust_robustness = None

        # Select final solution by highest accuracy
        best_index = int(np.argmax(pareto_acc))
        best_weights = pareto_weights[best_index]

        # Store normalized weights
        self.weights_ = best_weights / np.sum(best_weights)

        # Store validation loss (here: selected accuracy)
        self.validation_loss_ = float(pareto_acc[best_index])

        # Number of solutions evaluated per iteration (use NSGA‑II population size)
        self.iteration_batch_size_ = int(self.population_size)

        # Store accuracies from the Pareto set (end-of-run)
        self.val_loss_over_iterations_ = [float(a) for a in pareto_acc]

        # Store (true) robustness for the selected solution
        self.validation_robustness_ = float(pareto_robust_true[best_index])

        # Timing aggregation
        _t1_total = time.perf_counter()

        # End-of-run optimization diagnostics
        final_pareto_front_size = None
        final_hypervolume = None
        try:
            pareto_F = np.atleast_2d(res.F)
            final_pareto_front_size = int(len(NonDominatedSorting(method="fast_non_dominated_sort").do(pareto_F, only_non_dominated_front=True)))
        except Exception:
            pass
        try:
            pareto_F = np.atleast_2d(res.F)
            if _HV is not None and np.isfinite(pareto_F).all():
                # use the same ref point as during per-gen if available
                ref_point = None
                try:
                    ref_point = _hv_ref if ('_hv_ref' in locals() or '_hv_ref' in globals()) else None
                except Exception:
                    ref_point = None
                if ref_point is None:
                    ref_point = np.nanmax(pareto_F, axis=0) + 1e-6
                hv = _HV(ref_point=ref_point)
                final_hypervolume = float(hv.do(pareto_F))
        except Exception:
            pass

        # Store metadata (is saved to disk under benchmark/output)
        self.model_specific_metadata_.update(dict(
            selection_rule="by_accuracy",
            surrogate_pool_size=int(getattr(adv_union_predictions, 'shape', (0, 0, 0))[1]),
            pareto_accuracy=[float(v) for v in pareto_acc.tolist()],
            pareto_surrogate_robustness=[float(v) for v in pareto_robust_surr.tolist()],
            pareto_true_robustness=[None if np.isnan(v) else float(v) for v in pareto_robust_true.tolist()],
            best_acc_index=int(best_index),
            best_acc_accuracy=float(pareto_acc[best_index]),
            best_acc_robustness_true=None if np.isnan(pareto_robust_true[best_index]) else float(pareto_robust_true[best_index]),
            best_rob_index=best_true_robust_index,
            best_rob_accuracy=best_true_robust_accuracy,
            best_rob_robustness_true=best_true_robust_robustness,
            best_rob_weights=(
                None if best_true_robust_index is None else
                [float(w) for w in np.asarray(pareto_weights[best_true_robust_index], dtype=float).tolist()]
            ),
            # End-of-run diagnostics
            final_pareto_front_size=final_pareto_front_size,
            final_hypervolume=final_hypervolume,
            # Stats on attack success per base model
            base_model_attack_stats=base_model_attack_stats,
            # Stats on attack success per ensemble
            pareto_attack_stats=pareto_attack_stats,
            # Wall-clock times
            wallclock_total_sec=float(_t1_total - _t0_total),
            wallclock_precompute_adv_pool_sec=float(_t1_pool - _t0_pool),
            wallclock_optimization_sec=float(_t1_opt - _t0_opt),
            wallclock_reattack_sec=float(_t1_reattack - _t0_reattack),
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
