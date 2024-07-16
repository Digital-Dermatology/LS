from fastai.vision.all import *
from fastai.callback.fp16 import *
import optuna
import os
import numpy as np

# Define paths
data_path = "./LS_kfold/"
model_path = "./MODEL"

# Set CUDA device
torch.cuda.set_device(1)

# Define metrics to be averaged
metrics_names = ['val_loss', 'recall', 'precision', 'roc_auc']

def objective(trial):
    # Define hyperparameters to tune
    #######################################################################
    # Grid search configuration
    #######################################################################

    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    epochs = trial.suggest_int('epochs', 5, 15)
    freeze_epochs = trial.suggest_int('freeze_epochs', 0, 5)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])  
    # Initialize lists to store metrics for each fold
    metrics_per_fold = {metric: [] for metric in metrics_names}

    # Iterate over folds
    for fold in range(4): 
        print(f"Training and evaluating on fold {fold}...")

        # Data loaders setup for current fold
        dls = ImageDataLoaders.from_folder(
            os.path.join(data_path, f"fold_{fold}"),
            train="train", valid="valid", seed=42, bs=batch_size,  
            item_tfms=Resize(480), batch_tfms=aug_transforms(size=224)
        )

        # Create learner
        learn = cnn_learner(dls, resnet34, metrics=[accuracy, Precision(), Recall(), RocAucBinary()], path=model_path)

        # Convert learner to FP16 (mixed precision training)
        learn.to_fp16()

        # Fine-tune the model
        learn.fine_tune(epochs, lr, freeze_epochs=freeze_epochs)

        # Evaluate the model on the validation set of the current fold
        val_metrics = learn.validate()

        # Accumulate metrics for the current fold
        for i, metric_name in enumerate(metrics_names):
            metrics_per_fold[metric_name].append(val_metrics[i])

        # Print validation metrics for the current fold
        print(f"Validation metrics for fold {fold}:")
        for i, metric_name in enumerate(metrics_names):
            print(f"{metric_name}: {val_metrics[i]:.4f}")
        print("=" * 40)

    # Calculate average metrics across folds
    average_metrics = {metric_name: np.mean(metrics_per_fold[metric_name]) for metric_name in metrics_names}

    # Return the average value of the metric on which to optimize
    return average_metrics['val_loss'] 

# Optuna study
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)

# Best hyperparameters
best_params = study.best_params
best_lr = best_params['lr']
best_epochs = best_params['epochs']
best_freeze_epochs = best_params['freeze_epochs']
best_batch_size = best_params['batch_size']

print(f"Best LR: {best_lr}, Best Epochs: {best_epochs}, Best Freeze Epochs: {best_freeze_epochs}, Best Batch Size: {best_batch_size}")
