import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from ultralytics import YOLO
from ultralytics import FastSAM


class PascalPartModel(pl.LightningModule):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7, learning_rate=1e-3, transform=None):
        super(PascalPartModel, self).__init__()

        # Loss functions
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.jaccard_loss = JaccardLoss(mode='multiclass', from_logits=True)
        self.jaccard_index = MulticlassJaccardIndex(num_classes=num_classes)

        self.learning_rate = learning_rate
        self.transform = transform

        # Create a shared encoder using SMP (Segmentation Models PyTorch)
        # self.encoder = smp.DeepLabV3(
        #     encoder_name=encoder_name,        
        #     encoder_weights=encoder_weights,  
        #     classes=num_classes,                     
        #     activation=None                   
        # )
        self.encoder = smp.MAnet(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            classes=2,                     
            activation=None                   
        )

        # Fusion Module: Combine image features and mask features
        self.fusion_conv1d = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)

    
    def get_features_mask(self, x):
        
        device = 'cuda' if x.get_device() >= 0 else 'cpu'
        x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        x = [pic for pic in x]

        # Load a model for pose estimation
        model_pose = YOLO("yolov8x-pose.pt") 

        # run pose estimation model
        res_pose = model_pose.predict(
            x,
            conf=0.001, 
            imgsz=256,
            verbose=False
        )
        
        # extract keypoints фтв bounding boxes
        points = []
        labels = []
        bboxes = []

        for r in res_pose:
            tmp_points = r.keypoints.xy.reshape((-1, 2)).cpu().tolist()
            tmp_points += [[0, 0]]
            tmp_points = np.array(tmp_points, dtype=np.uint8)
            
            points.append(tmp_points)

            labels.append(np.arange(len(tmp_points)).astype(np.uint8))
            
            tmp_bboxes = r.boxes.xywh.reshape((-1, 4)).cpu().tolist()
            tmp_bboxes += [[0, 0, 0, 0]]
            tmp_bboxes = np.array(tmp_bboxes, dtype=np.uint8)
            bboxes.append(tmp_bboxes)


        # Create a FastSAM model
        model_sam = FastSAM("FastSAM-s.pt")  # "FastSAM-x.pt"

        try:
            res_seg = [model_sam.predict(
                source=x[i],
                half=True,
                verbose=False,
                iou=0.99,
                imgsz=256,
                points=points[i], 
                labels=labels[i], 
                bboxes=bboxes[i],
                # texts='human, body parts'
                # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
            )
            for i in range(len(points))
            ]
        except Exception:
            print(len(x), len(points), len(labels), len(bboxes))
            final_mask = torch.zeros(len(x), 1, 256, 256).to('cuda')
            return final_mask
        
        final_mask = [torch.sum(img[0].masks.data.cpu(), dim=0).numpy() for img in res_seg]
        final_mask = np.array(final_mask)


        final_mask = torch.tensor(final_mask)
        final_mask = final_mask.unsqueeze(1)

        final_mask = torch.tensor(final_mask).to(device)

        return final_mask

    def _get_body_mask(self, masks):
            """
            Extract the mask for body vs. background from groundthrought mask
            """
            body_mask = (masks > 0.5).long()  
            return body_mask


    def forward(self, x):
        # extract features using pose estimation and SAM
        sam_features = self.get_features_mask(x)
        
        x = torch.cat([x, sam_features], dim=1)
        # x = x + sam_features

        # TODO try torch.sum instead of torch.cat and fusion_conv1d !!??

        x = self.fusion_conv1d(x)

        # Pass the input through the encoder
        output = self.encoder(x)


        return output


    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.
        """

        images, masks = batch
        body_out = self(images)

        pred_body_out = torch.argmax(body_out, dim=1).unsqueeze(1).long()

        true_body_masks = self._get_body_mask(masks)

        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss = self.cross_entropy_loss(body_out, true_body_masks.squeeze(1))

        jaccard_loss = self.jaccard_loss(body_out, true_body_masks.squeeze(1))
        jaccard_index = self.jaccard_index(pred_body_out, true_body_masks)

        self.log('train_loss', loss,  prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_loss_mIoU_0', jaccard_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_index_mIoU_0', jaccard_index, prog_bar=True, on_step=True, on_epoch=True)

        return jaccard_loss # loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the model.
        """

        images, masks = batch
        body_out = self(images)

        pred_body_out = torch.argmax(body_out, dim=1).unsqueeze(1).long()

        true_body_masks = self._get_body_mask(masks)

        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss = self.cross_entropy_loss(body_out, true_body_masks.squeeze(1))

        jaccard_loss = self.jaccard_loss(body_out, true_body_masks.squeeze(1))
        jaccard_index = self.jaccard_index(pred_body_out, true_body_masks)

        self.log('val_loss', loss,  prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_jaccard_loss_mIoU_0', jaccard_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_jaccard_index_mIoU_0', jaccard_index, prog_bar=True, on_step=True, on_epoch=True)

        return jaccard_loss # loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        scheduler = lr_scheduler.StepLR(
            optimizer,
            step_size=5,
            gamma=0.1,
        )

        return [optimizer], [scheduler]