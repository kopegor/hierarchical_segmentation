from typing import Tuple

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PascalPartDataset(Dataset):
    """Class to create PascalPartDataset dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose = None) -> None:
        """Initialize dataset parameters.

        Parameters
        ----------
        df : pd.DataFrame
            paths to images and their labels
        transform : transforms.Compose
            A function/transform to apply to the images (default is None)
        """
        self.img_paths = df['PATH_TO_IMAGE'].to_list()
        self.mask_paths = df['PATH_TO_MASK'].to_list()
        self.transform = transform

    def __len__(self) -> int:
        """Method to return the length of dataset.

        Returns
        -------
        int
            Length of the dataset
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Method to get item of dataset.

        Parameters
        ----------
        idx : int
            index of item

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image and its corresponding label as tensors
        """
        # Get path to image
        path_to_img = self.img_paths[idx]
        path_to_mask = self.mask_paths[idx]

        # # Get image label
        # label = torch.tensor(self.img_labels[idx], dtype=torch.long)

        # Load the mask
        mask = np.load(path_to_mask)

        # Load the image
        image = cv2.imread(path_to_img)
        # Convert image to RGB tesnsor
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32)

        # Transpose image shape
        image = image.permute(2, 0, 1)

        # Apply transformation
        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask