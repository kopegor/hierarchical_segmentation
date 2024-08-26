import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torch.optim import lr_scheduler
from ultralytics import YOLO, FastSAM

class PascalPartModel(pl.LightningModule):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7, learning_rate=1e-3, transform=None):
        super(PascalPartModel, self).__init__()

        # Initialize loss functions
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.jaccard_loss = JaccardLoss(mode='multiclass', from_logits=True)
        self.jaccard_index_body = MulticlassJaccardIndex(num_classes=2)
        self.jaccard_index_up_low = MulticlassJaccardIndex(num_classes=3, ignore_index=0)
        self.jaccard_index_parts = MulticlassJaccardIndex(num_classes=7, ignore_index=0)

        self.learning_rate = learning_rate
        self.transform = transform

        # Create a shared encoder using Segmentation Models PyTorch
        self.model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=2,
            activation=None
        )

        # Freeze the encoder weights initially
        self.freeze_encoder()

        # Fusion Module: Combine image features and mask features
        self.body_conv1d = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)

        # Decoders for different levels of the hierarchical segmentation
        self.decoder_body = self.create_decoder(self.model, num_classes=2)
        self.decoder_up_low = self.create_decoder(self.model, num_classes=3)
        self.decoder_parts = self.create_decoder(self.model, num_classes=7)

    def create_decoder(self, model, num_classes):
        """
        Creates a decoder using the same architecture as the main encoder.
        """
        return nn.Sequential(
            model.decoder,
            nn.Conv2d(model.decoder.out_channels, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8.0)
        )

    def freeze_encoder(self):
        """Freeze the encoder to prevent its weights from being updated during training."""
        for param in self.model.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze the encoder to allow its weights to be updated during training."""
        for param in self.model.encoder.parameters():
            param.requires_grad = True

    def get_single_features_mask(self, img):
        """
        Extracts features and masks using YOLO and FastSAM models for a single image.
        """
        device = 'cuda' if img.get_device() >= 0 else 'cpu'
        x = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Load a model for pose estimation
        model_pose = YOLO("yolov8s-pose.pt")
        res_pose = model_pose.predict(x, conf=0.001, imgsz=256, verbose=False)

        # Extract keypoints and bounding boxes
        points, labels, bboxes = [], [], []
        for r in res_pose:
            tmp_points = np.array(r.keypoints.xy.reshape((-1, 2)).cpu().tolist() + [[0, 0]], dtype=np.uint8)
            points.append(tmp_points)
            labels.append(np.arange(len(tmp_points)).astype(np.uint8))
            tmp_bboxes = np.array(r.boxes.xywh.reshape((-1, 4)).cpu().tolist() + [[0, 0, 0, 0]], dtype=np.uint8)
            bboxes.append(tmp_bboxes)

        # Load and run FastSAM model for segmentation
        model_sam = FastSAM("FastSAM-s.pt")
        try:
            res_seg = model_sam.predict(
                source=x,
                half=True,
                verbose=False,
                iou=0.99,
                imgsz=256,
                points=points[0],
                labels=labels[0],
                bboxes=bboxes[0]
            )
        except Exception as err:
            print(err)
            print('added zeros feature map')
            return torch.zeros(256, 256).to(device)

        # Convert segmentation result to a final mask
        final_mask = torch.tensor(torch.sum(res_seg[0].masks.data.cpu(), dim=0).numpy()).to(device)
        return final_mask

    def _get_body_mask(self, masks):
        """
        Extracts the mask for body vs. background from the ground truth mask.
        """
        return (masks > 0).long()

    def _get_upper_lower_body_mask(self, masks):
        """
        Extracts the mask for upper vs. lower body from the ground truth mask.
        """
        upper_lower_body_mask = torch.zeros_like(masks).long()
        upper_lower_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
        upper_lower_body_mask[(masks == 3) | (masks == 5)] = 2
        return upper_lower_body_mask

    def freeze_pipeline(self):
        """
        Freezes/unfreezes specific layers based on the current epoch for progressive training.
        """
        if self.current_epoch <= 2:
            self._set_decoder_trainability(self.decoder_body, True)
            self._set_decoder_trainability(self.decoder_up_low, False)
            self._set_decoder_trainability(self.decoder_parts, False)
        elif self.current_epoch <= 5:
            self._set_decoder_trainability(self.decoder_body, False)
            self._set_decoder_trainability(self.decoder_up_low, True)
            self._set_decoder_trainability(self.decoder_parts, False)
        elif self.current_epoch <= 10:
            self._set_decoder_trainability(self.decoder_body, False)
            self._set_decoder_trainability(self.decoder_up_low, False)
            self._set_decoder_trainability(self.decoder_parts, True)
        else:
            self._set_decoder_trainability(self.decoder_body, True)
            self._set_decoder_trainability(self.decoder_up_low, True)
            self._set_decoder_trainability(self.decoder_parts, True)

    def _set_decoder_trainability(self, decoder, is_trainable):
        """Helper function to set the trainability of decoder layers."""
        for param in decoder.parameters():
            param.requires_grad = is_trainable

    def forward(self, x):
        device = 'cuda' if x.get_device() >= 0 else 'cpu'

        # Extract SAM features for each image in the batch
        sam_features = [self.get_single_features_mask(x_idx).cpu().tolist() for x_idx in x]
        sam_features = torch.tensor(sam_features, dtype=torch.float32).to(device).unsqueeze(1)

        # Concatenate image and SAM features
        x_for_body = torch.cat([x, sam_features], dim=1)
        x_for_body = self.body_conv1d(x_for_body)

        # Pass the input through the model for body segmentation
        output_features_body = self.model.encoder(x_for_body)[-1]
        output_body = self.decoder_body(output_features_body)
        output_up_low = self.decoder_up_low(output_features_body)
        output_parts = self.decoder_parts(output_features_body)

        return output_body, output_up_low, output_parts

    def on_train_epoch_start(self):
        """Hook for freezing/unfreezing layers at the start of each training epoch."""
        self.freeze_pipeline()

    def on_validation_epoch_start(self):
        """Hook for freezing/unfreezing layers at the start of each validation epoch."""
        self.freeze_pipeline()

    def training_step(self, batch, batch_idx):
        """Defines the training step for the model."""
        images, masks = batch
        out_body, out_up_low, out_parts = self(images)

        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        # Compute losses
        loss_body = self.cross_entropy_loss(out_body, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(out_up_low, truth_up_low_mask.squeeze(1))
        loss_parts = self.cross_entropy_loss(out_parts, masks.squeeze(1))

        total_loss = 0.34 * loss_body + 0.25 * loss_up_low + 0.4 * loss_parts

        # Log losses and metrics
        self._log_metrics('train', loss_body, loss_up_low, loss_parts, out_body, out_up_low, out_parts, truth_body_mask, truth_up_low_mask, masks)

        # Return appropriate loss based on current epoch
        if self.current_epoch <= 2:
            return loss_body
        elif self.current_epoch <= 5:
            return loss_up_low
        elif self.current_epoch <= 10:
            return loss_parts
        else:
            return total_loss

    def validation_step(self, batch, batch_idx):
        """Defines the validation step for the model."""
        images, masks = batch
        out_body, out_up_low, out_parts = self(images)

        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        # Compute losses
        loss_body = self.cross_entropy_loss(out_body, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(out_up_low, truth_up_low_mask.squeeze(1))
        loss_parts = self.cross_entropy_loss(out_parts, masks.squeeze(1))

        total_loss = 0.4 * loss_body + 0.2 * loss_up_low + 0.4 * loss_parts

        # Log losses and metrics
        self._log_metrics('val', loss_body, loss_up_low, loss_parts, out_body, out_up_low, out_parts, truth_body_mask, truth_up_low_mask, masks)

        # Return appropriate loss based on current epoch
        if self.current_epoch <= 2:
            return loss_body
        elif self.current_epoch <= 5:
            return loss_up_low
        elif self.current_epoch <= 10:
            return loss_parts
        else:
            return total_loss

    def _log_metrics(self, stage, loss_body, loss_up_low, loss_parts, out_body, out_up_low, out_parts, truth_body_mask, truth_up_low_mask, masks):
        """Helper function to log metrics during training and validation."""
        total_loss = 0.34 * loss_body + 0.25 * loss_up_low + 0.4 * loss_parts
        jaccard_loss_body = self.jaccard_loss(out_body, truth_body_mask.squeeze(1))
        jaccard_loss_up_low = self.jaccard_loss(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_loss_parts = self.jaccard_loss(out_parts, masks.squeeze(1))
        jaccard_index_body = self.jaccard_index_body(out_body, truth_body_mask.squeeze(1))
        jaccard_index_up_low = self.jaccard_index_up_low(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_index_parts = self.jaccard_index_parts(out_parts, masks.squeeze(1))

        self.log(f'{stage}_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_body_loss', loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_up_low_loss', loss_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_parts_loss', loss_parts, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_loss_mIoU_0', jaccard_loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_loss_mIoU_1', jaccard_loss_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_loss_mIoU_2', jaccard_loss_parts, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_index_mIoU_0', jaccard_index_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_index_mIoU_1', jaccard_index_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f'{stage}_jaccard_index_mIoU_2', jaccard_index_parts, prog_bar=True, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
        return [optimizer], [scheduler]






