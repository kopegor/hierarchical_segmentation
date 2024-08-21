import pytorch_lightning as pl
import pandas as pd
import albumentations as alb
from dataset import PascalPartDataset
from lightning_module import PascalPartModel
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision import transforms

# set random seeds
seed = 42
# pl.seed_everything(seed=seed, workers=True)


# TODO build df with paths to images and masks for train/val splits
# TODO create train/val dataloaders
# TODO create transforms



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
    transform = transforms.Compose([
        transforms.Resize((256, 256))
    ])

    # create datasets
    train_dataset = PascalPartDataset(df_train_paths, transform=transform)
    val_dataset = PascalPartDataset(df_val_paths, transform=transform)

    batch_size = 32
    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=63)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=63)

    # create a model
    pl_model = PascalPartModel()
    
    trainer = pl.Trainer(
        max_epochs=2,
        accelerator='gpu',
        devices=1    
    )

    trainer.fit(pl_model, train_dataloader, val_dataloader)


if __name__ == '__main__':
    train()