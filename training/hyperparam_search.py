import logging

import numpy as np
import optuna
import torch

from data.loaders import generate_synthetic_data, load_swiss_radar_data
from models.linda_pinn import LINDAPINNModel, device

optuna.logging.set_verbosity(optuna.logging.WARNING)


def optimize_pinn_hyperparams(
    rainrate_sequence,
    metadata,
    n_trials=50,
    epochs_per_trial=20,
    progress_callback=None,
):
    """Bayesian hyperparameter search for LINDA-PINN using Optuna.

    Searches over: learning rate, hidden size, number of layers,
    physics weight, and weight decay.

    Args:
        rainrate_sequence: numpy array (T, ny, nx) of rain rates
        metadata: dict with dataset metadata
        n_trials: number of Optuna trials
        epochs_per_trial: training epochs used in each trial objective
        progress_callback: optional callable(trial_number, n_trials, best_value)

    Returns:
        best_params (dict), study (optuna.Study)
    """
    from training.trainers import LINDAPINNTrainer

    completed_trials = [0]

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        num_layers = trial.suggest_int("num_layers", 3, 8)
        physics_weight = trial.suggest_float("physics_weight", 0.01, 1.0, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        layers = [4] + [hidden_size] * num_layers + [1]

        trainer = LINDAPINNTrainer(physics_weight=physics_weight)
        trainer.model = LINDAPINNModel(layers=layers)
        trainer.optimizer = torch.optim.Adam(
            trainer.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            trainer.optimizer, patience=max(5, epochs_per_trial // 5)
        )

        try:
            losses, _ = trainer.train_on_radar_sequence(
                rainrate_sequence, metadata, epochs=epochs_per_trial, verbose=False
            )
            if len(losses) == 0 or not np.isfinite(losses).any():
                return float("inf")
            val_loss = float(np.mean(losses[-max(1, len(losses) // 4):]))
        except Exception as e:
            logging.warning(f"Trial {trial.number} failed: {e}")
            return float("inf")
        finally:
            completed_trials[0] += 1
            if progress_callback is not None:
                best = trial.study.best_value if trial.study.best_trial is not None else float("inf")
                progress_callback(completed_trials[0], n_trials, best)

        return val_loss

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    return best_params, study


def run_hpo(use_synthetic_data=True, n_trials=20, epochs_per_trial=10, progress_callback=None):
    """Load data and run PINN hyperparameter optimization.

    Returns:
        best_params (dict), study (optuna.Study), results_text (str)
    """
    if use_synthetic_data:
        rainrate_sequence, metadata = generate_synthetic_data()
    else:
        try:
            rainrate_sequence, metadata = load_swiss_radar_data()
        except Exception:
            print("Failed to load real data, using synthetic instead")
            rainrate_sequence, metadata = generate_synthetic_data()

    print(f"Starting HPO: {n_trials} trials, {epochs_per_trial} epochs each")
    best_params, study = optimize_pinn_hyperparams(
        rainrate_sequence,
        metadata,
        n_trials=n_trials,
        epochs_per_trial=epochs_per_trial,
        progress_callback=progress_callback,
    )

    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    results_text = f"""
## HPO Results — PINN Hyperparameter Search

**Trials completed:** {n_complete} / {n_trials}
**Failed trials:** {n_failed}
**Best validation loss:** {study.best_value:.6f}

### Best Hyperparameters

| Parameter | Value |
|---|---|
| Learning Rate | `{best_params['lr']:.2e}` |
| Hidden Size | `{best_params['hidden_size']}` |
| Num Layers | `{best_params['num_layers']}` |
| Physics Weight | `{best_params['physics_weight']:.4f}` |
| Weight Decay | `{best_params['weight_decay']:.2e}` |

### Top 5 Trials

| Rank | Loss | lr | hidden | layers | phys_w | wd |
|---|---|---|---|---|---|---|
"""

    completed = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )
    for rank, t in enumerate(completed[:5], 1):
        p = t.params
        results_text += (
            f"| {rank} | {t.value:.6f} "
            f"| {p['lr']:.2e} | {p['hidden_size']} | {p['num_layers']} "
            f"| {p['physics_weight']:.3f} | {p['weight_decay']:.2e} |\n"
        )

    return best_params, study, results_text
