import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss



class HierarchicalSegmentationModel(pl.LightningModule):
    """
    A PyTorch Lightning module for hierarchical semantic segmentation.

    Parameters
    ----------
    encoder_name : str
        The name of the encoder architecture (e.g., 'resnet34', 'mobilenet_v3').
    encoder_weights : str
        The pre-trained weights for the encoder (e.g., 'imagenet').
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet'):
        super(HierarchicalSegmentationModel, self).__init__()
        
        # Create a shared encoder using SMP (Segmentation Models PyTorch)
        self.encoder = smp.Unet(
            encoder_name=encoder_name,        # The name of the encoder
            encoder_weights=encoder_weights,  # Pre-trained weights for the encoder
            classes=None,                     # Output classes (not needed for custom decoders)
            activation=None                   # No activation function (to be applied later)
        )
        
        # First decoder: Body (body vs. background)
        self.body_decoder = self.create_decoder(encoder=self.encoder, num_classes=1)  # For BCEWithLogitsLoss
        
        # Second decoder: Upper body vs. Lower body (4 classes: 1, 2, 4, 6 for upper, 3, 5 for lower)
        self.upper_lower_body_decoder = self.create_decoder(encoder=self.encoder, num_classes=1)  # For BCEWithLogitsLoss
        
        # Third decoder: Specific body parts (6 classes: low_hand, up_hand, torso, head, low_leg, up_leg)
        self.lower_body_decoder = self.create_decoder(encoder=self.encoder, num_classes=6)  # For CrossEntropyLoss
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.jaccard_loss = JaccardLoss(mode='multiclass')

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
            nn.Conv2d(encoder.decoder[-1].out_channels, num_classes, kernel_size=1)
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
        body_mask = (masks > 0).float()  # BCEWithLogitsLoss expects float masks
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
        upper_body_mask = torch.zeros_like(masks).float()  # For BCEWithLogitsLoss
        upper_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
        return upper_body_mask

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
        
        # Calculate loss for body vs. background (mIoU^0)
        loss_body = self.bce_loss(body_output, body_mask)
        
        # Calculate loss for upper vs. lower body within the body mask (mIoU^1)
        loss_upper_lower_body = self.bce_loss(upper_lower_body_output, upper_body_mask)
        
        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_lower_body = self.cross_entropy_loss(lower_body_output, masks - 1)  # Adjust the mask for the lower body
        
        # Combine the losses
        total_loss = (loss_body + loss_upper_lower_body + loss_lower_body) / 3
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(body_output, body_mask)
        jaccard_loss_upper_lower = self.jaccard_loss(upper_lower_body_output, upper_body_mask)
        jaccard_loss_lower_body = self.jaccard_loss(lower_body_output, masks - 1)
        
        # Log losses
        self.log('train_loss', total_loss)
        self.log('train_jaccard_loss_mIoU^0', jaccard_loss_body)
        self.log('train_jaccard_loss_mIoU^1', jaccard_loss_upper_lower)
        self.log('train_jaccard_loss_mIoU^2', jaccard_loss_lower_body)
        
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
        
        # Calculate validation loss for body vs. background (mIoU^0)
        val_loss_body = self.bce_loss(body_output, body_mask)
        
        # Calculate validation loss for upper vs. lower body (mIoU^1)
        val_loss_upper_lower_body = self.bce_loss(upper_lower_body_output, upper_body_mask)
        
        # Calculate validation loss for specific body parts (mIoU^2)
        val_loss_lower_body = self.cross_entropy_loss(lower_body_output, masks - 1)  # Adjust the mask for the lower body
        
        total_val_loss = (val_loss_body + val_loss_upper_lower_body + val_loss_lower_body) / 3
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(body_output, body_mask)
        jaccard_loss_upper_lower = self.jaccard_loss(upper_lower_body_output, upper_body_mask)
        jaccard_loss_lower_body = self.jaccard_loss(lower_body_output, masks - 1)
        
        # Log validation losses
        self.log('val_loss', total_val_loss)
        self.log('val_jaccard_loss_mIoU^0', jaccard_loss_body)
        self.log('val_jaccard_loss_mIoU^1', jaccard_loss_upper_lower)
        self.log('val_jaccard_loss_mIoU^2', jaccard_loss_lower_body)
        
        return total_val_loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns
        -------
        torch.optim.Optimizer
            The Adam optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-3)

