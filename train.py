import pytorch_lightning as pl
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import PascalPartDataset
from lightning_module import PascalPartModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import mlflow
import time

# set random seeds
seed = 42
pl.seed_everything(seed=seed, workers=True)



def train():
    df_train_paths = pd.DataFrame({'PATH_TO_IMAGE': [], 'PATH_TO_MASK': []})
    df_val_paths = pd.DataFrame({'PATH_TO_IMAGE': [], 'PATH_TO_MASK': []})

    # extract names of images and masks for train and val splts
    splits = ["train_id", "val_id"]
    splits_samples = {}

    for split in splits:
        with open(f'/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/hierarchical_segmentation/data/{split}.txt') as f:
            splits_samples[split] = f.read().splitlines()

    path_to_raw = '/storage/AIDA_PROJECTS/egor.koptelov/MIL_test_task/Pascal-part/Pascal-part'

    # create dataframes with full paths to images and masks
    df_train_paths['PATH_TO_IMAGE'] = [f'{path_to_raw}/JPEGImages/{id_img}.jpg' for id_img in splits_samples['train_id']]
    df_train_paths['PATH_TO_MASK'] = [f'{path_to_raw}/gt_masks/{id_mask}.npy' for id_mask in splits_samples['train_id']]
    df_val_paths['PATH_TO_IMAGE'] = [f'{path_to_raw}/JPEGImages/{id_img}.jpg' for id_img in splits_samples['val_id']]
    df_val_paths['PATH_TO_MASK'] = [f'{path_to_raw}/gt_masks/{id_mask}.npy' for id_mask in splits_samples['val_id']]

    # specify transforms 
    # transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # ])

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            # max_pixel_value=255.0,
        ),
        ToTensorV2()
    ])

    # create datasets
    train_dataset = PascalPartDataset(df_train_paths, transform=transform)
    val_dataset = PascalPartDataset(df_val_paths, transform=transform)

    batch_size = 64
    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=63)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=63)

    # create a model
    pl_model = PascalPartModel(
        encoder_name='resnet34',
        encoder_weights='imagenet', 
        learning_rate=5e-4, 
        transform=transform, 
        num_classes=7
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss'),
        LearningRateMonitor(logging_interval='epoch'),
    ]


    trainer = pl.Trainer(
        max_epochs=25,
        accelerator='gpu',
        devices=1    
    )

    #  mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./lightning_logs --host 0.0.0.0 --port 3033
    
    # specify server port
    port = '3033'

    # speify model name
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_name = f'DeepLabV3_{timestamp}'

    # Set tracking uri for mlflow server
    mlflow.set_tracking_uri(f'http://localhost:{port}')
    # specify mlflow logging frequency
    log_freq = 30

    # Auto log all MLflow entities
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

    project = 'PascalPart'
    # set up experiment name
    mlflow.set_experiment(project)
    experiment = mlflow.get_experiment_by_name(project)

    # start mlflow logging
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=model_name,
    ):
        # strart training the model
        trainer.fit(pl_model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    train()