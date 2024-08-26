import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torch.optim import lr_scheduler

class PascalPartModel(pl.LightningModule):
    """
    A PyTorch Lightning module for hierarchical semantic segmentation.

    Parameters
    ----------
    encoder_name : str, optional
        The name of the encoder architecture (e.g., 'resnet34', 'mobilenet_v3').
    encoder_weights : str, optional
        The pre-trained weights for the encoder (e.g., 'imagenet').
    num_classes : int, optional
        The number of output classes (default is 7).
    learning_rate : float, optional
        The learning rate for the optimizer (default is 1e-3).
    transform : transforms.Compose, optional
        A function/transform to apply to the images (default is None).
    """
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7, learning_rate=1e-3, transform=None):
        super(PascalPartModel, self).__init__()
        
        # Create a shared encoder using Segmentation Models PyTorch (SMP)
        self.encoder = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=num_classes,
            activation=None
        )
        
        # First decoder: Body (body vs. background)
        self.body_decoder = self.create_decoder(encoder=self.encoder, num_classes=2)
        
        # Loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.jaccard_loss = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0)

        self.learning_rate = learning_rate
        self.transform = transform

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
        torch.Tensor
            Output tensor for the segmentation of body vs. background.
        """
        # Pass the input through the encoder
        encoded_features = self.encoder.encoder(x)

        # First level: predictions for body vs. background
        body_output = self.body_decoder(encoded_features[-1])

        return body_output
    
    def _get_body_mask(self, masks):
        """
        Extracts the mask for body vs. background.

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
        Extracts the mask for upper vs. lower body.

        Parameters
        ----------
        masks : torch.Tensor
            The input mask tensor of shape (batch_size, height, width).

        Returns
        -------
        torch.Tensor
            A binary mask containing 1 for upper body and 0 for lower body.
        """
        upper_lower_body_mask = torch.zeros_like(masks).long()
        upper_lower_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
        return upper_lower_body_mask

    def _convert_parts_to_body(self, masks):
        """
        Converts specific body part masks into a general body mask.

        Parameters
        ----------
        masks : torch.Tensor
            The input tensor containing specific body part masks.

        Returns
        -------
        torch.Tensor
            A binary mask for body vs. background.
        """
        body_pred = torch.zeros_like(masks)[:, 0, :, :].unsqueeze(1)

        for idx_logit, logit in enumerate(masks):
            for idx_mask, mask in enumerate(logit):
                body_pred[idx_logit, 0, mask >= 1] = 1 

        return body_pred

    def _convert_parts_to_up_low(self, masks):
        """
        Converts specific body part masks into upper vs. lower body masks.

        Parameters
        ----------
        masks : torch.Tensor
            The input tensor containing specific body part masks.

        Returns
        -------
        torch.Tensor
            A mask differentiating between upper and lower body.
        """
        up_low_pred = torch.zeros_like(masks)[:, 0, :, :].unsqueeze(1)

        for idx_logit, logit in enumerate(masks):
            for idx_mask, mask in enumerate(logit):
                up_low_pred[idx_logit, 0, (mask == 1) | (mask == 2) | (mask == 4) | (mask == 6)] = 1 

        return up_low_pred

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
        body_output = self(images)

        # Extract body vs. background mask
        body_mask = self._get_body_mask(masks)
        
        # Calculate cross-entropy loss
        loss_body = self.cross_entropy_loss(body_output, body_mask.squeeze(1))
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(torch.argmax(body_output, dim=1).unsqueeze(1), body_mask)
        
        # Log the losses
        self.log('train_loss', loss_body, prog_bar=True, on_step=False, on_epoch=True)
        self.log('train_jaccard_loss_mIoU_0', jaccard_loss_body, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss_body
    
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
        body_output = self(images)

        # Extract body vs. background mask
        body_mask = self._get_body_mask(masks)
        
        # Calculate cross-entropy loss
        val_loss_body = self.cross_entropy_loss(body_output, body_mask.squeeze(1))
        
        # Calculate the Jaccard loss (IoU)
        jaccard_loss_body = self.jaccard_loss(torch.argmax(body_output, dim=1).unsqueeze(1), body_mask)
        
        # Log the losses
        self.log('val_loss', val_loss_body, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_jaccard_loss_mIoU_0', jaccard_loss_body, prog_bar=True, on_step=False, on_epoch=True)
        
        return val_loss_body
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns
        -------
        tuple
            A tuple containing the optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return [optimizer], [scheduler]
