import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss

import numpy as np

class PascalPartModel(pl.LightningModule):
    """
    A PyTorch Lightning module for hierarchical semantic segmentation.

    Parameters
    ----------
    encoder_name : str
        The name of the encoder architecture (e.g., 'resnet34', 'mobilenet_v3').
    encoder_weights : str
        The pre-trained weights for the encoder (e.g., 'imagenet').
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7):
        super(PascalPartModel, self).__init__()
        
        # Create a shared encoder using SMP (Segmentation Models PyTorch)
        self.encoder = smp.DeepLabV3(
            encoder_name=encoder_name,        # The name of the encoder
            encoder_weights=encoder_weights,  # Pre-trained weights for the encoder
            classes=num_classes,                     # Output classes (not needed for custom decoders)
            activation=None                   # No activation function (to be applied later)
        )
        
        # First decoder: Body (body vs. background)
        self.body_decoder = self.create_decoder(encoder=self.encoder, num_classes=2)  # For BCEWithLogitsLoss
        
        # Second decoder: Upper body vs. Lower body (4 classes: 1, 2, 4, 6 for upper, 3, 5 for lower)
        self.upper_lower_body_decoder = self.create_decoder(encoder=self.encoder, num_classes=2)  # For BCEWithLogitsLoss
        
        # Third decoder: Specific body parts (6 classes: low_hand, up_hand, torso, head, low_leg, up_leg)
        self.lower_body_decoder = self.create_decoder(encoder=self.encoder, num_classes=num_classes)  # For CrossEntropyLoss
        
        # Loss functions
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.jaccard_loss = JaccardLoss(mode='multiclass', from_logits=True)

    def create_decoder(self, encoder, num_classes):
        """
        Creates a decoder using the same architecture as the main encoder.

        Parameters
        ----------
        encoder : nn.Module
            The encoder from which to base the decoder architecture.
        num_classes : int
            The number of output classes for this decoder.

        Returns
        -------
        nn.Sequential
            A sequential model representing the decoder.
        """
        return nn.Sequential(
            encoder.decoder,
            nn.Conv2d(self.encoder.decoder.out_channels, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8.0)
        )
    
    def forward(self, x):
        """
        Defines the forward pass for the hierarchical segmentation model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, channels, height, width).

        Returns
        -------
        tuple of torch.Tensor
            Output tensors for each level of the hierarchy: 
            - body_output : segmentation of body vs. background
            - upper_lower_body_output : segmentation of upper vs. lower body
            - lower_body_output : segmentation of specific body parts
        """
        # Pass the input through the encoder
        encoded_features = self.encoder.encoder(x)

        # First level: predictions for body vs. background
        body_output = self.body_decoder(encoded_features[-1])
        
        # Second level: predictions for upper body vs. lower body
        upper_lower_body_output = self.upper_lower_body_decoder(encoded_features[-1])
        
        # Third level: predictions for specific body parts
        lower_body_output = self.lower_body_decoder(encoded_features[-1])
        
        return body_output, upper_lower_body_output, lower_body_output
    
    def _get_body_mask(self, masks):
        """
        Extract the mask for body vs. background.
        
        Parameters
        ----------
        masks : torch.Tensor
            The input mask tensor of shape (batch_size, height, width).

        Returns
        -------
        torch.Tensor
            A binary mask containing 1 for body and 0 for background.
        """
        body_mask = (masks > 0).long()  
        return body_mask

    def _get_upper_lower_body_mask(self, masks):
        """
        Extract the mask for upper vs. lower body.

        Parameters
        ----------
        masks : torch.Tensor
            The input mask tensor of shape (batch_size, height, width).

        Returns
        -------
        torch.Tensor
            A binary mask containing 1 for upper body and 0 for lower body.
        """
        upper_body_mask = torch.zeros_like(masks).long()  
        upper_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
        return upper_body_mask


    # def _convert_body_parts_mask(self, masks):
    #     body_parts_mask = torch.zeros_like(masks).float()
    #     body_parts_mask = body_parts_mask[:, 0, :, :].unsqueeze(1)
    #     # body_parts_mask = torch.sum(masks, axis=1).unsqueeze(1).float()

    #     # print(body_parts_mask.shape)

    #     for idx_logit, logit in enumerate(masks):
    #         # print(logit.shape)
    #         for idx_mask, mask in enumerate(logit):
    #             # print(mask.shape)
    #             body_parts_mask[idx_logit, 0, mask > 0] = idx_mask + 1
        
        
    #     return body_parts_mask


    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.

        Parameters
        ----------
        batch : tuple
            A batch of input images and corresponding masks.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The total loss for the current training step.
        """
        images, masks = batch
        body_output, upper_lower_body_output, lower_body_output = self(images)
        
        # Extract body vs. background mask
        body_mask = self._get_body_mask(masks)
        
        # Extract upper vs. lower body mask
        upper_body_mask = self._get_upper_lower_body_mask(masks)
        
        # Convert masks for body parts
        # body_parts_mask = self._convert_body_parts_mask(lower_body_output)

        # print(body_output.shape)
        # print(upper_lower_body_output.shape)
        # print(lower_body_output.shape)
        # print(body_parts_mask.shape)

        # print(body_mask.shape)
        # print(upper_body_mask.shape)
        # print(masks.shape)
        

        # Calculate loss for body vs. background (mIoU^0)
        # loss_body = self.bce_loss(body_output, body_mask)
        loss_body = self.cross_entropy_loss(body_output, body_mask.squeeze(1))
        
        # Calculate loss for upper vs. lower body within the body mask (mIoU^1)
        # loss_upper_lower_body = self.bce_loss(upper_lower_body_output, upper_body_mask)
        loss_upper_lower_body = self.cross_entropy_loss(upper_lower_body_output, upper_body_mask.squeeze(1))
        
        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_lower_body = self.cross_entropy_loss(lower_body_output, masks.squeeze(1))  # Adjust the mask for the lower body

        # Combine the losses
        total_loss = (loss_body + loss_upper_lower_body + loss_lower_body) / 3
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(body_output, body_mask)
        jaccard_loss_upper_lower = self.jaccard_loss(upper_lower_body_output, upper_body_mask)
        jaccard_loss_lower_body = self.jaccard_loss(lower_body_output, masks)
        
        # Log losses
        self.log('train_loss', total_loss, on_step=True)
        self.log('train_jaccard_loss_mIoU^0', jaccard_loss_body, on_step=True)
        self.log('train_jaccard_loss_mIoU^1', jaccard_loss_upper_lower, on_step=True)
        self.log('train_jaccard_loss_mIoU^2', jaccard_loss_lower_body, on_step=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the model.

        Parameters
        ----------
        batch : tuple
            A batch of input images and corresponding masks.
        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            The total validation loss for the current step.
        """
        images, masks = batch
        body_output, upper_lower_body_output, lower_body_output = self(images)
        
        # Extract body vs. background mask
        body_mask = self._get_body_mask(masks)
        
        # Extract upper vs. lower body mask
        upper_body_mask = self._get_upper_lower_body_mask(masks)

        # Convert masks for body parts
        # body_parts_out = self._convert_body_parts_mask(lower_body_output)
        
        # print(body_output.shape)
        # print(upper_lower_body_output.shape)
        # print(lower_body_output.shape)
        # print(body_parts_out.shape)

        # print(body_mask.shape)
        # print(upper_body_mask.shape)
        # print(masks.shape)
        # print(body_mask.squeeze(1).shape)

        # print(lower_body_output.dtype)
        # print(masks.dtype)

        # print(body_output.dtype)
        # print(body_mask.dtype)
        
        # Calculate validation loss for body vs. background (mIoU^0)
        # val_loss_body = self.bce_loss(body_output, body_mask)
        val_loss_body = self.cross_entropy_loss(body_output, body_mask.squeeze(1))
        
        # # Calculate validation loss for upper vs. lower body (mIoU^1)
        # val_loss_upper_lower_body = self.bce_loss(upper_lower_body_output, upper_body_mask)
        val_loss_upper_lower_body = self.cross_entropy_loss(upper_lower_body_output, upper_body_mask.squeeze(1))
        
        # # Calculate validation loss for specific body parts (mIoU^2)
        val_loss_lower_body = self.cross_entropy_loss(lower_body_output, masks.squeeze(1))  # Adjust the mask for the lower body
        # # val_loss_lower_body = self.cross_entropy_loss(body_parts_out, masks - 1)    

        total_val_loss = (val_loss_body + val_loss_upper_lower_body + val_loss_lower_body) / 3
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(body_output, body_mask)
        jaccard_loss_upper_lower = self.jaccard_loss(upper_lower_body_output, upper_body_mask)
        jaccard_loss_lower_body = self.jaccard_loss(lower_body_output, masks)

        total_jaccard_loss = (jaccard_loss_body + jaccard_loss_upper_lower + jaccard_loss_lower_body) / 3
        
        # Log validation losses
        self.log('val_loss', total_val_loss, on_step=True)
        self.log('val_jaccard_loss_mIoU^0', jaccard_loss_body, on_step=True)
        self.log('val_jaccard_loss_mIoU^1', jaccard_loss_upper_lower, on_step=True)
        self.log('val_jaccard_loss_mIoU^2', jaccard_loss_lower_body, on_step=True)
        
        return total_val_loss # total_jaccard_loss 
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

