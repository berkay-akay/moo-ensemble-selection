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
from assembled_ensembles.methods.moo.moo_ensemble_problem import MOOEnsembleProblem, EnsembleSklearnWrapper, PredictLogger
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state
import pickle

# Pymoo Imports (for running NSGA-II)
from pymoo.factory import get_algorithm
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import time
# Optional: Hypervolume performance indicator (guarded by try/except for compatibility)
try:
    from pymoo.performance_indicator.hv import Hypervolume as _HV
except Exception:  # older/newer pymoo or missing module
    _HV = None

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

        # Prefer values directly from metadata if present
        dataset_name = md0.get("dataset_name")
        openml_task_id = md0.get("openml_task_id")

        # Fallback: parse the metatask JSON (input-side) to extract them – do NOT call get_metatask here
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
            # Raise to abort the fold cleanly; the caller should catch and skip scoring
            raise RuntimeError("Datasets with categorical input features are not supported yet for MOO-ES")
        # -------------------------------------------------------------------------------




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

        # Use per-feature bounds / clip-values
        X_np = np.asarray(X, dtype=np.float32)
        feat_min = X_np.min(axis=0)
        feat_max = X_np.max(axis=0)
        same = feat_max <= feat_min
        feat_max[same] = feat_min[same] + 1e-8
        print(f"[MOO-ES] Using per-feature clip_values with shapes: {feat_min.shape}, {feat_max.shape}", flush=True)

        HSJ_MAX_EVAL = 25

        for m_idx, bm in enumerate(actual_base_models):
            print(f"[MOO-ES] Generating adversarial set for base model {m_idx}", flush=True)

            wrapper = EnsembleSklearnWrapper(
                base_models=[bm],
                weights=np.array([1.0], dtype=float),
                classes_=self.classes_,
                feature_names=feature_names_for_attack,
            ).fit()

            # ------------------------------------------------------------
            # Step 1: determine which validation instances are clean-correct
            # for THIS base model
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
                perturbed_sets.append(x_adv_full)
                origin_idx.append(np.full(x_adv_full.shape[0], m_idx, dtype=int))
                print(f"[MOO-ES] Done generating adversarial set for model {m_idx}", flush=True)
                continue

            # Attack only the clean-correct subset
            X_sub = X_np[clean_mask]

            pred_with_logging = PredictLogger(
                wrapper.predict_proba,
                max_eval=X_sub.shape[0] * HSJ_MAX_EVAL,
                name=f"HSJ_bm{m_idx}"
            )

            clf = BlackBoxClassifier(
                pred_with_logging,
                input_shape=(X_np.shape[1],),
                nb_classes=len(self.classes_),
                clip_values=(feat_min, feat_max),
            )

            attack = HopSkipJump(
                classifier=clf,
                targeted=False,
                max_iter=3,
                max_eval=HSJ_MAX_EVAL,
                init_eval=20,
                init_size=3,
                verbose=True
            )

            print(
                f"[MOO-ES] Base model {m_idx}: attacking only clean-correct subset "
                f"with shape={X_sub.shape}",
                flush=True,
            )

            x_adv_sub = attack.generate(x=X_sub)

            # Insert adversarial examples only for clean-correct rows.
            # Clean-wrong rows remain the original clean instances.
            x_adv_full[clean_mask] = x_adv_sub

            perturbed_sets.append(x_adv_full)
            origin_idx.append(np.full(x_adv_full.shape[0], m_idx, dtype=int))

            print(
                f"[MOO-ES] Done generating adversarial set for model {m_idx} "
                f"(full block shape={x_adv_full.shape})",
                flush=True,
            )

        if len(perturbed_sets) > 0:
            U = np.vstack(perturbed_sets)
            U_labels = np.hstack([labels for _ in range(len(perturbed_sets))])
            U_origin = np.hstack(origin_idx)
        else:
            X_np = np.asarray(X, dtype=np.float32)
            U = np.empty((0, X_np.shape[1]), dtype=X_np.dtype)
            U_labels = np.empty((0,), dtype=np.asarray(labels).dtype)
            U_origin = np.empty((0,), dtype=int)
        print(f"[MOO-ES] Built union pool U with shape {getattr(U, 'shape', None)}", flush=True)

        # Build cached predictions for adversarial union pool
        if U.shape[0] == 0:
            adv_union_predictions = np.zeros((len(actual_base_models), 0, len(self.classes_)), dtype=float)
        else:
            adv_union_predictions = []  # will become (n_models, n_union, n_classes)
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
        print(f"[MOO-ES] Cached adv_union_predictions with shape {getattr(adv_union_predictions, 'shape', None)}", flush=True)
        _t1_pool = time.perf_counter()

        # --------------------- Build problem ---------------------
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
            base_models_metadata=(self.base_models_metadata_ if getattr(self, 'base_models_metadata_exists', False) else None),
            adv_union_predictions=adv_union_predictions,
            adv_union_labels=U_labels,
        )

        # Configure algorithm
        algorithm = get_algorithm("nsga2", pop_size=self.population_size)

        # --------------------- Optimization with per-generation callback (timed) ---------------------
        self.model_specific_metadata_.setdefault('per_generation_stats', [])
        _run_start = time.perf_counter()
        _t0_opt = time.perf_counter()

        nds = NonDominatedSorting(method="fast_non_dominated_sort")

        # Telemetry helpers for optimization dynamics
        _hv_ref = None  # lazily determined reference point for hypervolume

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

        def _per_gen_callback(algorithm):
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

        print("[MOO-ES] Re-attacking Pareto-optimal ensembles to compute true robustness", flush=True)
        _t0_reattack = time.perf_counter()
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
        _t1_reattack = time.perf_counter()

        pareto_robust_true = np.asarray(true_robust_list, dtype=float)

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

        # End-of-run optimization diagnostics (guarded)
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
            selected_index=int(best_index),
            selected_accuracy=float(pareto_acc[best_index]),
            selected_robustness_true=None if np.isnan(pareto_robust_true[best_index]) else float(pareto_robust_true[best_index]),
            # End-of-run diagnostics
            final_pareto_front_size=final_pareto_front_size,
            final_hypervolume=final_hypervolume,
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
