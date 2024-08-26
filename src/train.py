import pytorch_lightning as pl
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
import mlflow
import time
import os
from sklearn.model_selection import train_test_split
import torch

from dataset import PascalPartDataset
from pose_estimation_sam_module import PascalPartModel

# Set random seeds for reproducibility
seed = 142
pl.seed_everything(seed=seed, workers=True)

def train():
    """
    Prepares data loaders, initializes the model, and trains the model using PyTorch Lightning.

    This function handles the following steps:
    - Reads training and validation image paths.
    - Sets up data augmentation and normalization.
    - Initializes datasets and data loaders.
    - Configures the model, trainer, and callbacks.
    - Integrates MLflow for experiment tracking.
    - Starts the training process.
    """
    # Define the base path to the dataset
    path_to_raw = '/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/Pascal-part'
    
    # Initialize empty DataFrames to store image and mask paths
    df_train_paths = pd.DataFrame({'PATH_TO_IMAGE': [], 'PATH_TO_MASK': []})
    df_val_paths = pd.DataFrame({'PATH_TO_IMAGE': [], 'PATH_TO_MASK': []})

    # # Define the train and validation splits
    # splits = ["train_id", "val_id"]
    # splits_samples = {}

    # # Read image IDs from split files
    # for split in splits:
    #     with open(f'/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/hierarchical_segmentation/data/{split}.txt') as f:
    #         splits_samples[split] = f.read().splitlines()
    
    # # Create DataFrames with full paths to images and masks for training and validation sets
    # df_train_paths['PATH_TO_IMAGE'] = [f'{path_to_raw}/Pascal-part/JPEGImages/{id_img}.jpg' for id_img in splits_samples['train_id']]
    # df_train_paths['PATH_TO_MASK'] = [f'{path_to_raw}/Pascal-part/gt_masks/{id_mask}.npy' for id_mask in splits_samples['train_id']]
    # df_val_paths['PATH_TO_IMAGE'] = [f'{path_to_raw}/Pascal-part/JPEGImages/{id_img}.jpg' for id_img in splits_samples['val_id']]
    # df_val_paths['PATH_TO_MASK'] = [f'{path_to_raw}/Pascal-part/gt_masks/{id_mask}.npy' for id_mask in splits_samples['val_id']]

    names_images = os.listdir('/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/Pascal-part/big_mask_samples/JPEGImages/')
    names_images = sorted(names_images)

    names_masks = os.listdir('/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/Pascal-part/big_mask_samples/gt_masks/')
    names_masks = sorted(names_masks)

    train_idx, val_idx = train_test_split(list(range(len(names_images))), train_size=0.8)

    # Create DataFrames with full paths to images and masks for training and validation sets
    df_train_paths['PATH_TO_IMAGE'] = [f'{path_to_raw}/big_mask_samples/JPEGImages/{names_images[idx]}' for idx in train_idx]
    df_train_paths['PATH_TO_MASK'] = [f'{path_to_raw}/big_mask_samples/gt_masks/{names_masks[idx]}' for idx in train_idx]
    df_val_paths['PATH_TO_IMAGE'] =  [f'{path_to_raw}/big_mask_samples/JPEGImages/{names_images[idx]}' for idx in val_idx]
    df_val_paths['PATH_TO_MASK'] = [f'{path_to_raw}/big_mask_samples/gt_masks/{names_masks[idx]}' for idx in val_idx]


    # Data augmentation and normalization transformations
    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Create datasets for training and validation
    train_dataset = PascalPartDataset(df_train_paths, transform=transform)
    val_dataset = PascalPartDataset(df_val_paths, transform=transform)

    # Define the batch size for training
    batch_size = 16

    # Create DataLoaders for training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=63)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=63)

    # Initialize the model with specific parameters
    pl_model = PascalPartModel(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        learning_rate=1e-4,
        transform=transform
    )

    # Upload trained model for body fine-tuning
    checkpoint = torch.load('/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/hierarchical_segmentation/lightning_logs/1/b84a2a48ca984bf085103a03100dc09e/artifacts/checkpoints/latest_checkpoint.pth')
    pl_model.load_state_dict(checkpoint['state_dict'])
    
    # Define callbacks for early stopping and learning rate monitoring
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, min_delta=1e-3),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Configure the PyTorch Lightning trainer
    trainer = pl.Trainer(
        max_epochs=7,
        accelerator='gpu',
        devices=1    
    )

    # command to starting mlflow server
    #  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./lightning_logs --host 0.0.0.0 --port 3033

    # Configure MLflow for logging experiments
    port = '3033'
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_name = f'DeepLabV3_{timestamp}'

    mlflow.set_tracking_uri(f'http://localhost:{port}')
    log_freq = 10

    # Enable MLflow automatic logging
    mlflow.pytorch.autolog(
        log_every_n_step=log_freq,
        log_models=True,
        registered_model_name=model_name,
        checkpoint=True,
        checkpoint_monitor='val_loss',
        checkpoint_mode='min',
        checkpoint_save_best_only=True,
        checkpoint_save_freq='epoch',
    )

    # Set the experiment name in MLflow
    project = 'PascalPart'
    mlflow.set_experiment(project)
    experiment = mlflow.get_experiment_by_name(project)

    # Start MLflow run and train the model
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=model_name,
    ):
        trainer.fit(pl_model, train_dataloader, val_dataloader)

if __name__ == '__main__':
    train()
