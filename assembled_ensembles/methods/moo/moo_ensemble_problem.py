import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from pymoo.core.problem import Problem
from assembled_ensembles.util.metrics import AbstractMetric
from sklearn.utils import check_random_state
from typing import List, Optional, Callable, Union

from assembled_ensembles.methods.moo.permute_attack.ga_attack import GAdvExample

def run_permute_attack_single_instance(
    estimator,
    x_i,
    x_train,
    true_label,
    feature_names=None,
    sol_per_pop=35,
    num_parents_mating=15,
    num_generations=100,
    n_runs=1,
    beta=0.96,
    black_list=None,
    verbose=False,
    target=None,
):
    """
    Runs PermuteAttack on a single instance.
    Returns (x_adv_i, reason), where:
      - x_adv_i is shape (1, d) if attack succeeded
      - x_adv_i is None if no valid adversarial was found
    """
    x_i = np.asarray(x_i, dtype=np.float32).reshape(-1)
    x_train = np.asarray(x_train, dtype=np.float32)

    attack = GAdvExample(
        cat_vars_ohe=None,  # you already abort on categorical-feature datasets
        feature_names=None,
        sol_per_pop=sol_per_pop,
        num_parents_mating=num_parents_mating,
        num_generations=num_generations,
        n_runs=n_runs,
        black_list=[] if black_list is None else list(black_list),
        beta=beta,
        verbose=verbose,
        target=target,
    )

    try:
        # DEBUG PRINTS
        #print(
        #    f"[PermuteAttackDebug] x_i dtype={x_i.dtype}, shape={x_i.shape} | "
        #    f"x_train dtype={x_train.dtype}, shape={x_train.shape} | "
        #    f"true_label={true_label} ({type(true_label)}) | "
        #    f"feature_names_type={type(feature_names)} | "
        #    f"feature_names_sample={None if feature_names is None else feature_names[:4]}",
        #    flush=True,
        #)

        _, _, x_success = attack.attack(
            estimator=estimator,
            x=x_i,
            x_train=x_train,
        )
    except Exception as e:
        return None, (
            f"exception:{e} | "
            f"x_i_dtype={getattr(x_i, 'dtype', None)} | "
            f"x_train_dtype={getattr(x_train, 'dtype', None)} | "
            f"true_label_type={type(true_label)} | "
            f"feature_names_type={type(feature_names)}"
        )

    if x_success is None or len(x_success) == 0:
        return None, "no_success"

    x_success = np.asarray(x_success, dtype=np.float32)
    if x_success.ndim == 1:
        x_success = x_success.reshape(1, -1)

    # Pick the sparsest successful candidate, tie-break by smallest L1 distance
    x_orig = x_i.reshape(1, -1)
    l0 = np.count_nonzero(x_success != x_orig, axis=1)
    l1 = np.sum(np.abs(x_success - x_orig), axis=1)
    best_idx = np.lexsort((l1, l0))[0]
    x_adv_i = x_success[best_idx:best_idx + 1]

    # Re-check that it is actually adversarial
    pred_adv = np.asarray(estimator.predict(x_adv_i)).reshape(-1)
    if pred_adv.size == 0 or pred_adv[0] == true_label:
        return None, "not_adversarial"

    return x_adv_i, None

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
        List of predictions from base models on the clean validation set (probabilities).
    labels: np.ndarray
        True labels of the validation set (encoded to match classes_).
    score_metric: AbstractMetric
        Metric for accuracy evaluation (called with to_loss=False).
    base_models: List[Callable]
        List of base model instances (unpickled sklearn pipelines).
    X: np.ndarray
        Data instances (with input features) from validation set.
    random_state: Optional[Union[int, np.random.RandomState]]
        Random state for reproducibility.
    n_jobs: int
        Number of jobs for parallel processing.
    classes_: Optional[np.ndarray]
        Class labels as used by the models.
    base_models_metadata: Optional[List[dict]]
        Optional metadata from the fake predictors to help with feature names.
    adv_union_predictions: np.ndarray
        Cached adversarial predictions with shape (n_models, n_union, n_classes).
    adv_union_labels: np.ndarray
        Labels aligned with the union set (length n_union). Used to score the surrogate robustness.

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
                 adv_union_predictions: np.ndarray = None,
                 adv_union_labels: np.ndarray = None,
                 X_train_attack: Optional[np.ndarray] = None,
                 permute_attack_kwargs: Optional[dict] = None):

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
        self.adv_union_predictions = adv_union_predictions
        self.adv_union_labels = adv_union_labels
        self.X_train_attack = None if X_train_attack is None else np.asarray(X_train_attack, dtype=np.float32)
        if self.adv_union_predictions is not None:
            print(f"[MOOProblem] Using surrogate robustness on union set with shape: "
                  f"{getattr(self.adv_union_predictions, 'shape', None)}", flush=True)

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


    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate accuracy and robustness (through a surrogate) for a batch/generation of ensembles/weight vectors.

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

        # Iterate over weight vectors in NSGA-II population
        for weights in x:
            # Normalize weights
            w_sum = np.sum(weights)
            if w_sum <= 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights = weights / w_sum

            # Aggregate ensemble predictions using weights (predictions on the clean validation set)
            ensemble_pred = self._ensemble_predict(self.predictions, weights)

            # Evaluate clean accuracy
            accuracy = self.score_metric(self.labels, ensemble_pred, to_loss=False, checks=False)
            f1.append(-accuracy)  # Pymoo minimizes objective, so we negate to maximize accuracy

            # --- ADVERSARIAL ROBUSTNESS SURROGATE ---
            # Identify clean-correct instances in the ensemble
            ensemble_pred = self._ensemble_predict(self.predictions, weights)  # (N, C)
            y_true = np.asarray(self.labels)
            y_pred_clean = np.argmax(ensemble_pred, axis=1)
            correct_mask = (y_pred_clean == y_true)
            num_correct = int(correct_mask.sum())

            # Compute ensemble predictions on the adversarial union
            #   - Adversarial Union = union of all adversarial copies for each instance based on each base model
            ensemble_adv_pred = np.tensordot(weights, self.adv_union_predictions, axes=(0, 0))


            N = int(len(y_true)) # Number of original validation instances
            U = np.asarray(ensemble_adv_pred) # Candidate ensemble's predictions on the adversarial union set


            # Check if U is in the correct shape
            if U.ndim != 2 or U.shape[0] == 0:
                print(
                    f"[Surrogate] Degenerate union tensor (U.ndim={U.ndim}, U.shape={getattr(U, 'shape', None)}). "
                    f"Setting robust_rate=0.0",
                    flush=True,
                )
                robust_rate = 0.0
            else:
                n_union = int(U.shape[0]) # Compute total number of adversarial copies in the union
                if N == 0 or (n_union % max(1, N)) != 0: # Check correct structure of U (n_union = Number of base models * Number of validation instances)
                    # If structure incorrect, fallback to using the adversarial accuracy over all union points at once (alternative surrogate)
                    print(
                        f"[Surrogate] Degenerate U shape (N={N}, n_union={n_union}) "
                        f"[Surrogate] Fallback to scoring over all adversarial union points at once: N={N}, n_union={n_union}, "
                        f"n_union % N = {None if N == 0 else (n_union % N)}",
                        flush=True,
                    )
                    legacy = self.score_metric(self.adv_union_labels, U, to_loss=False, checks=False)
                    robust_rate = float(legacy)
                    print(
                        f"[Surrogate] Adv-accuracy over union (all points) = {robust_rate:.6f}",
                        flush=True,
                    )

                else:
                    M = n_union // N  # number of attacker blocks
                    U_reshaped = U.reshape(M, N, -1)  # (M, N, C)
                    adv_pred_labels = np.argmax(U_reshaped, axis=2)  # (M, N), convert probabilities to class labels

                    # Code for "new" surrogate that uses survival values and weighted averaging
                    # Survival indicator per attacker block and instance:
                    # kept_true[m, i] = 1 if attacked copy from block m keeps the true label for instance i
                    y_true_b = np.broadcast_to(y_true, (M, N)) # Copy true labels to all attacker blocks
                    kept_true = (adv_pred_labels == y_true_b).astype(float)

                    if num_correct == 0: # If ensemble gets no clean instance correct
                        robust_rate = 0.0
                        print("[Surrogate] No clean-correct instances; robust_rate=0.0", flush=True)
                    else:
                        # Weight attacker blocks by the current candidate ensemble weights (block 1 gets weight of bm 1 etc.)
                        attacker_weights = np.asarray(weights, dtype=float)
                        if attacker_weights.sum() <= 0:
                            attacker_weights = np.ones_like(attacker_weights) / len(attacker_weights)
                        else:
                            attacker_weights = attacker_weights / attacker_weights.sum()

                        # For each instance, compute weighted average "survival" across blocks / all adversarial copies
                        per_instance_survival = np.average(
                            kept_true,
                            axis=0,
                            weights=attacker_weights
                        )

                        # New pruning rule:
                        # if survival drops below 0.5, set it to 0.0
                        per_instance_survival_pruned = per_instance_survival.copy()
                        per_instance_survival_pruned[per_instance_survival_pruned < 0.5] = 0.0

                        # Final surrogate robustness = average pruned per-instance survival
                        # over the ensemble-clean-correct instances only
                        robust_rate = float(np.mean(per_instance_survival_pruned[correct_mask]))


            # Pymoo minimizes -> negate to maximize robustness
            f2.append(-robust_rate)

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
        Evaluate robustness of ensemble using PermuteAttack.
        Returns adversarial accuracy on the clean-correct subset.
        """

        if self.X_train_attack is None:
            raise ValueError("X_train_attack is required for PermuteAttack-based robustness evaluation.")

        # Recover original feature names from metadata
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

        # Build sklearn-compatible ensemble wrapper
        attack_classes = np.arange(len(self.classes_))

        ensemble_model = EnsembleSklearnWrapper(
            base_models=self.base_models,
            weights=weights,
            classes_=attack_classes,
            feature_names=feature_names,
        )
        assert isinstance(ensemble_model, BaseEstimator)
        assert isinstance(ensemble_model, ClassifierMixin)

        ensemble_model.fit()

        print("[RobustnessEval] entering _evaluate_robustness with PermuteAttack", flush=True)

        X_all = np.asarray(self.X, dtype=np.float32)
        y_true = np.asarray(self.labels)

        # Clean-correct mask
        proba_clean = ensemble_model.predict_proba(X_all)
        y_pred_clean = np.argmax(proba_clean, axis=1)
        mask = (y_pred_clean == y_true)
        num_correct = int(mask.sum())

        print(
            f"[RobustnessEval] clean-correct mask computed: num_clean_correct={num_correct} / N={len(y_true)}",
            flush=True,
        )

        if num_correct == 0:
            print("[RobustnessEval] No clean-correct instances; returning 0.0", flush=True)
            return {
                "adv_accuracy": 0.0,
                "n_clean_correct": 0,
                "n_success": 0,
                "n_failed": 0,
                "success_rate": 0.0,
            }

        X_sub_adv = X_all[mask].copy()
        y_sub = y_true[mask]
        orig_correct_idx = np.where(mask)[0]

        n_success = 0
        n_failed = 0

        for local_idx, orig_idx in enumerate(orig_correct_idx, start=1):
            if local_idx % 50 == 0 or local_idx == len(orig_correct_idx):
                print(
                    f"[RobustnessEval] PermuteAttack progress "
                    f"{local_idx}/{len(orig_correct_idx)} instances",
                    flush=True,
                )

            x_i = X_all[orig_idx]

            x_adv_i, fail_reason = run_permute_attack_single_instance(
                estimator=ensemble_model,
                x_i=x_i,
                x_train=self.X_train_attack,
                true_label=y_true[orig_idx],
                feature_names=feature_names,
                **self.permute_attack_kwargs,
            )

            if x_adv_i is None:
                n_failed += 1
                continue

            X_sub_adv[local_idx - 1] = x_adv_i[0]
            n_success += 1

        print(
            f"[RobustnessEval] PermuteAttack finished on clean-correct subset | "
            f"success={n_success}, failed={n_failed}",
            flush=True,
        )

        adv_preds_sub = ensemble_model.predict_proba(X_sub_adv)

        adv_accuracy = self.score_metric(y_sub, adv_preds_sub, to_loss=False, checks=False)
        attack_success_rate = float(n_success / max(1, num_correct))

        print(
            f"[RobustnessEval] adv_accuracy on clean-correct subset = {adv_accuracy:.6f}",
            flush=True,
        )

        return {
            "adv_accuracy": float(adv_accuracy),
            "n_clean_correct": int(num_correct),
            "n_success": int(n_success),
            "n_failed": int(n_failed),
            "success_rate": attack_success_rate,
        }
