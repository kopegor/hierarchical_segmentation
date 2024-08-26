from typing import Tuple
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class PascalPartDataset(Dataset):
    """Class to create PascalPartDataset dataset."""

    def __init__(self, df: pd.DataFrame, transform: transforms.Compose = None) -> None:
        """
        Initialize dataset parameters.

        Parameters
        ----------
        df : pd.DataFrame
            Paths to images and their masks.
        transform : transforms.Compose, optional
            A function/transform to apply to the images (default is None).
        """
        self.img_paths = df['PATH_TO_IMAGE'].to_list()
        self.mask_paths = df['PATH_TO_MASK'].to_list()
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns
        -------
        int
            Length of the dataset.
        """
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            Image and its corresponding mask as tensors.
        """
        # Get paths to the image and mask
        path_to_img = self.img_paths[idx]
        path_to_mask = self.mask_paths[idx]

        # Load the mask from the numpy file
        mask = np.load(path_to_mask)

        # Load the image using OpenCV
        image = cv2.imread(path_to_img)
        # Convert the image from BGR (OpenCV default) to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transformation if specified
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            # Ensure mask is in long format (for classification) and add a channel dimension
            mask = mask.long().unsqueeze(0)

        return image, mask
