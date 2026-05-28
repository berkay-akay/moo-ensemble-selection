# Assembled-Ensembles

This directory contains the ensemble methods and the code needed to run them on metatasks. It includes the original post-hoc ensembling methods from the Q(D)O-ES repository and the proposed multi-objective method **MOEns**.

The main entry point is:

```text
run_evaluate_ensemble_on_metatask.py
```

This script loads a metatask, instantiates the chosen ensemble method, fits the ensemble on validation data, evaluates it, and stores metadata and results.

## Directory Overview

* `methods/ensemble_selection`: Greedy Ensemble Selection.
* `methods/qdo`: QO-ES and QDO-ES.
* `methods/moo`: MOEns, the NSGA-II problem definition, ensemble wrapper, and PermuteAttack implementation.
* `default_configurations`: Factory functions for constructing ensemble methods from configuration strings.
* `configspaces`: Supported configurations and parameter grids.
* `util`: Metric handling, preprocessing helpers, and diversity utilities.
* `wrapper`: Standardised wrappers around ensemble methods.

## Installation

To run this part of the code, use the environment files provided in the `environment` directory. A Linux environment is recommended.

The MOEns implementation uses, among others:

* `pymoo` for NSGA-II,
* `scikit-learn` for model interfaces and metrics,
* `NumPy` and `pandas` for data handling,
* the included PermuteAttack implementation for attack-based robustness evaluation.

## MOEns Overview

MOEns is a post-hoc multi-objective ensembling method. It optimises continuous ensemble weight vectors over a pruned Auto-Sklearn base model pool.

The method uses NSGA-II with two objectives:

1. clean predictive performance on validation data, and
2. surrogate adversarial robustness.

The surrogate robustness objective is computed using an adversarial union pool. This pool is created by attacking each base model on its clean-correct validation instances with PermuteAttack. During NSGA-II, base-model predictions on this union pool are cached and reused to estimate the robustness of candidate ensembles efficiently.

After optimisation, MOEns extracts the Pareto-optimal weight vectors. By default, it re-attacks the top Pareto-optimal ensembles by clean predictive performance and computes their true attack-based robustness using PermuteAttack.

The final stored ensemble weights correspond to the Pareto-optimal ensemble with the highest clean predictive performance. The metadata also stores information about the ensemble with the highest true adversarial robustness among the re-attacked Pareto-optimal ensembles.

## Main MOEns Parameters

The main user-facing MOEns parameters in the configuration grid are:

* `n_generations`: number of NSGA-II generations.
* `population_size`: number of candidate weight vectors per NSGA-II generation.

The implementation also exposes the following internal parameters:

* `score_metric`: metric used for clean predictive performance and final true robustness evaluation.
* `n_jobs`: number of CPU cores used by the method.
* `reattack_top5_only`: if enabled, only the top Pareto-optimal ensembles by clean predictive performance are re-attacked.
* `permute_attack_kwargs`: optional dictionary for changing the PermuteAttack configuration.

The default PermuteAttack configuration is:

```python
sol_per_pop = 35
num_parents_mating = 15
num_generations = 100
n_runs = 1
beta = 0.96
black_list = None
verbose = False
target = None
```

For the thesis experiments, MOEns was run with:

```text
n_generations = 50
population_size = 50
```

This results in 2500 evaluated candidate weight vectors, matching the QO-ES optimisation budget used in the comparison.

## Minimal Example

The repository contains a toy metatask under `benchmark/input/minimal_example_ens`. The following commands run several ensemble methods on this minimal example.

### SingleBest

```shell
python run_evaluate_ensemble_on_metatask.py -1 SiloTopN "SingleBest" balanced_accuracy minimal_example_ens ensemble_evaluations_qdo no no -1 QDO conf_singlebest
```

### Greedy Ensemble Selection

```shell
python run_evaluate_ensemble_on_metatask.py -1 SiloTopN "EnsembleSelection|use_best" balanced_accuracy minimal_example_ens ensemble_evaluations_qdo no no -1 QDO conf_ges
```

### QO-ES

```shell
python run_evaluate_ensemble_on_metatask.py -1 SiloTopN "QDOEnsembleSelection|archive_type:quality|batch_size:20|crossover:two_point_crossover|crossover_probability:0.5|crossover_probability_dynamic|elite_selection_method:deterministic|emitter_initialization_method:AllL1|max_elites:16|mutation_probability_after_crossover:0.5|mutation_probability_after_crossover_dynamic|starting_step_size:1" balanced_accuracy minimal_example_ens ensemble_evaluations_qdo no no -1 QDO conf_qo_es
```

### QDO-ES

```shell
python run_evaluate_ensemble_on_metatask.py -1 SiloTopN "QDOEnsembleSelection|archive_type:sliding|batch_size:20|behavior_space:bs_configspace_similarity_and_loss_correlation|buffer_ratio:1.0|crossover:two_point_crossover|crossover_probability:0.5|crossover_probability_dynamic|elite_selection_method:deterministic|emitter_initialization_method:AllL1|max_elites:16|mutation_probability_after_crossover:0.5|mutation_probability_after_crossover_dynamic|starting_step_size:1" balanced_accuracy minimal_example_ens ensemble_evaluations_qdo no no -1 QDO conf_qdo_es
```

### MOEns

```shell
python run_evaluate_ensemble_on_metatask.py -1 SiloTopN "MOOEnsembleSelection|n_generations:10|population_size:50" balanced_accuracy minimal_example_ens ensemble_evaluations_qdo no no -1 QDO conf_moens
```

The results are stored under:

```text
benchmark/output/<benchmark_name>/task_<task_id>/<evaluation_name>/<pruner>
```

## Detail Usage Documentation

To evaluate an ensemble on a metatask, execute:

```shell
python run_evaluate_ensemble_on_metatask.py task_id pruner ensemble_method_name metric_name benchmark_name evaluation_name isolate_execution load_method folds_to_run_on config_space_name ens_save_name
```

Arguments:

* `task_id`: OpenML task ID. For testing, use `-1`.
* `pruner`: pruning strategy used to load the metatask, usually `SiloTopN`.
* `ensemble_method_name`: full method configuration string.
* `metric_name`: metric optimised by the ensemble method, for example `balanced_accuracy`.
* `benchmark_name`: name of the benchmark input folder.
* `evaluation_name`: name of the evaluation output folder.
* `isolate_execution`: use `yes` to isolate execution and reduce memory leakage; otherwise use `no`.
* `load_method`: use `delayed` for delayed loading or `no` for loading the metatask at once.
* `folds_to_run_on`: use `-1` for all folds or a single fold index such as `0`.
* `config_space_name`: currently `QDO`, also used for MOEns configurations.
* `ens_save_name`: name used to identify the configuration in the output files.

## Stored Metadata for MOEns

MOEns stores detailed metadata for later analysis. Important fields include:

* clean predictive performance values for the Pareto front,
* surrogate robustness values for the Pareto front,
* true robustness values for re-attacked Pareto-optimal ensembles,
* weights of the selected ensemble,
* index and scores of the highest-clean-performance ensemble,
* index and scores of the highest-true-robustness ensemble,
* attack statistics for individual base models,
* attack statistics for re-attacked Pareto-optimal ensembles,
* per-generation NSGA-II diagnostics,
* wall-clock times for adversarial-pool construction, optimisation, re-attack, and total runtime.

## Notes and Limitations

* MOEns currently expects numeric input features. Datasets with categorical input features are aborted.
* MOEns is computationally expensive because it applies PermuteAttack repeatedly.
* The surrogate robustness objective is an approximation and may overestimate or underestimate true attack-based robustness.
* The final true robustness evaluation uses the same evaluation metric as clean predictive performance. In the thesis experiments, this was balanced accuracy.