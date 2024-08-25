import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import JaccardLoss
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.segmentation import MeanIoU
from torch.optim import lr_scheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

from ultralytics import YOLO
from ultralytics import FastSAM
from ultralytics.engine.results import Masks, Results


class PascalPartModel(pl.LightningModule):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7, learning_rate=1e-3, transform=None):
        super(PascalPartModel, self).__init__()

        # Loss functions
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(
            # ignore_index=0
            )
        self.jaccard_loss = JaccardLoss(mode='multiclass', from_logits=True)
        self.jaccard_index_body = MulticlassJaccardIndex(
            num_classes=2, 
            # ignore_index=0
        )
        self.jaccard_index_up_low = MulticlassJaccardIndex(num_classes=3, ignore_index=0)
        self.jaccard_index_parts = MulticlassJaccardIndex(num_classes=7, ignore_index=0)
        # self.jaccard_index = MeanIoU()
        self.learning_rate = learning_rate
        self.transform = transform

        # Create a shared encoder using SMP (Segmentation Models PyTorch)
        # self.model = smp.MAnet(
        #     encoder_name=encoder_name,        
        #     encoder_weights=encoder_weights,  
        #     classes=2,                     
        #     activation=None                   
        # )
        self.model = smp.DeepLabV3(  
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            classes=2,                    
            activation=None                   
        )

        # Freeze encoder weights
        self.freeze_encoder()

        # Fusion Module: Combine image features and mask features
        # self.fusion_conv1d = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)
        self.body_conv1d = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=3, padding=1)
        # self.up_low_conv1d = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1)
        # self.parts_conv1d = nn.Conv2d(in_channels=9, out_channels=3, kernel_size=3, padding=1)

        self.decoder_body = self.create_decoder(self.model, num_classes=2)
        self.decoder_up_low = self.create_decoder(self.model, num_classes=3)
        self.decoder_parts = self.create_decoder(self.model, num_classes=7)


    def create_decoder(self, model, num_classes):
        """
        Creates a decoder using the same architecture as the main encoder.
        """
        return nn.Sequential(
            model.decoder,
            nn.Conv2d(self.model.decoder.out_channels, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8.0)
        )

    def freeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        for param in self.model.encoder.parameters():
            param.requires_grad = True


    def get_single_features_mask(self, img):
        
        device = 'cuda' if img.get_device() >= 0 else 'cpu'

        x = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        
        # Load a model for pose estimation
        model_pose = YOLO("yolov8s-pose.pt") 

        # run pose estimation model
        res_pose = model_pose.predict(
            x,
            # iou=0.999,
            conf=0.001, 
            # imgsz=640,
            imgsz=256,
            # device='cpu'
            verbose=False
        )

        
        # extract keypoints фтв bounding boxes
        points = []
        labels = []
        bboxes = []

        for r in res_pose:

            # tmp_points = r.keypoints.xy.reshape((-1, 2)).cpu().numpy().astype(np.uint8)
            tmp_points = r.keypoints.xy.reshape((-1, 2)).cpu().tolist()
            tmp_points += [[0, 0]]
            tmp_points = np.array(tmp_points, dtype=np.uint8)
            
            points.append(tmp_points)

            labels.append(np.arange(len(tmp_points)).astype(np.uint8)) # .tolist())
            
            # tmp_bboxes = r.boxes.xywh.reshape((-1, 4)).cpu().numpy().astype(np.uint8)
            tmp_bboxes = r.boxes.xywh.reshape((-1, 4)).cpu().tolist()
            tmp_bboxes += [[0, 0, 0, 0]]
            tmp_bboxes = np.array(tmp_bboxes, dtype=np.uint8)
            bboxes.append(tmp_bboxes)

        # Create a FastSAM model
        model_sam = FastSAM("FastSAM-s.pt")  # "FastSAM-x.pt"

        # x = x.astype(np.uint8)
        res_seg = None

        # for i in range(len(points)):
        try:
            res_seg = model_sam.predict(
                source=x,
                # device='gpu',
                half=True,
                verbose=False,
                iou=0.99,
                # imgsz=640,
                imgsz=256,
                points=points[0], 
                labels=labels[0], 
                bboxes=bboxes[0],
                # conf=0.01,
                # texts='human, body parts'
                # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
            )
            
            # res_seg.append(tmp_res)

        except Exception as err:
            # print('try without text promt')
            # print(err)
            # try:
            #     res_seg = model_sam.predict(
            #         source=x,
            #         # device='gpu',
            #         half=True,
            #         verbose=False,
            #         iou=0.99,
            #         # imgsz=640,
            #         imgsz=256,
            #         points=points[0], 
            #         labels=labels[0], 
            #         bboxes=bboxes[0],
            #         # conf=0.01,
            #         # texts='human, body parts'
            #         # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
            #     )

            #     # res_seg.append(tmp_res)
            
            # except Exception as err: 
            
            final_mask = torch.zeros(256, 256).to(device)
            print(err)
            print('added zeros feature map')

                # print(f'zeros final mask shape: {final_mask.shape}')

            return final_mask


        # convert to final mask
        
        final_mask = torch.sum(res_seg[0].masks.data.cpu(), dim=0).numpy()
        final_mask = np.array(final_mask)
        final_mask = torch.tensor(final_mask)

        final_mask = torch.tensor(final_mask).to(device)
        # print(f'normal final mask shape: {final_mask.shape}')

        return final_mask


    def _get_body_mask(self, masks):
        """
        Extract the mask for body vs. background from groundthrought mask
        """
        body_mask = (masks > 0).long()  
        return body_mask

    def _get_upper_lower_body_mask(self, masks):
        """
        Extract the mask for upper vs. lower body from groundthrought mask
        0 - background
        1 - upper body
        2 - lower body
        """
        upper_lower_body_mask = torch.zeros_like(masks).long()  
        upper_lower_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
        upper_lower_body_mask[(masks == 3) | (masks == 5)] = 2
        return upper_lower_body_mask
    
    def _convert_parts_to_body(self, masks):
        body_pred = torch.zeros_like(masks)
        body_pred = body_pred[:, 0, :, :].unsqueeze(1)

        for idx_logit, logit in enumerate(masks):
            for idx_mask, mask in enumerate(logit):
                body_pred[idx_logit, 0, mask >= 0.5] = 1 

        return body_pred

    def _convert_parts_to_up_low(self, masks):
        up_low_pred = torch.zeros_like(masks)
        up_low_pred = up_low_pred[:, 0, :, :].unsqueeze(1)

        for idx_logit, logit in enumerate(masks):
            for idx_mask, mask in enumerate(logit):
                up_low_pred[idx_logit, 0, (mask == 1) | (mask == 2) | (mask == 4) | (mask == 6)] = 1 
                # up_low_pred[idx_logit, 0, (0.5 < mask and mask< 2.5) or (3.5 < mask and mask < 4.5) or (5.5 < mask)] = 1 
        
        return up_low_pred

    def freeze_pipeline(self):
        if self.current_epoch <= 1:
            for param in self.decoder_parts.parameters():
                param.requires_grad = False

        elif self.current_epoch <= 3:
            for param in self.decoder_body.parameters():
                param.requires_grad = False

            for param in self.decoder_up_low.parameters():
                param.requires_grad = False

            for param in self.decoder_parts.parameters():
                param.requires_grad = True


        elif self.current_epoch <= 6:
            for param in self.decoder_body.parameters():
                param.requires_grad = True

            for param in self.decoder_up_low.parameters():
                param.requires_grad = True

            for param in self.decoder_parts.parameters():
                param.requires_grad = True

        
        else:
            for param in self.decoder_body.parameters():
                param.requires_grad = False

            for param in self.decoder_up_low.parameters():
                param.requires_grad = False

            for param in self.decoder_parts.parameters():
                param.requires_grad = True



    def forward(self, x):
        device = 'cuda' if x.get_device() >= 0 else 'cpu'
        # extract features using pose estimation and SAM
        # sam_features = self.get_features_mask(x)


        # sam_features = [self.get_single_features_mask(x_idx).cpu().tolist() for x_idx in x]

        # sam_features = torch.tensor(sam_features, dtype=torch.float32).to(device).unsqueeze(1)

        # # concat image and extracted features from SAM     
        # x_for_body = torch.cat([x, sam_features], dim=1)
        # # x = x + sam_features

        # x_for_body = self.body_conv1d(x_for_body)

        # # Pass the input through the encoder for body segmentation
        # output_body = self.model_body(x_for_body)
        # # output_body = self.decoder_body(x_for_body)

        # x_for_up_low = torch.cat([x, sam_features, output_body], dim=1)
        # x_for_up_low = self.up_low_conv1d(x_for_up_low)

        # # output_up_low = self.model_up_low(x_for_up_low)
        # output_up_low = self.model_body.encoder(x_for_up_low)[-1]
        # output_up_low = self.decoder_up_low(output_up_low)

        # x_for_parts = torch.cat([x, sam_features, output_body, output_up_low], dim=1)
        # x_for_parts = self.parts_conv1d(x_for_parts)

        # # output_parts = self.model_parts(x_for_parts)
        # output_parts = self.model_body.encoder(x_for_parts)[-1]
        # output_parts = self.decoder_parts(output_parts)

        sam_features = [self.get_single_features_mask(x_idx).cpu().tolist() for x_idx in x]

        sam_features = torch.tensor(sam_features, dtype=torch.float32).to(device).unsqueeze(1)

        # concat image and extracted features from SAM     
        x_for_body = torch.cat([x, sam_features], dim=1)
        # x = x + sam_features

        x_for_body = self.body_conv1d(x_for_body)

        # Pass the input through the encoder for body segmentation
        output_features_body = self.model.encoder(x_for_body)[-1]
        output_body = self.decoder_body(output_features_body)
        # output_body = self.decoder_body(x_for_body)

        # x_for_up_low = torch.cat([x, sam_features, output_body], dim=1)
        # x_for_up_low = self.up_low_conv1d(x_for_up_low)

        # output_up_low = self.model_up_low(x_for_up_low)
        # output_up_low = self.model.encoder(x_for_up_low)[-1]
        # output_up_low = self.decoder_up_low(output_up_low)
        output_up_low = self.decoder_up_low(output_features_body)

        # x_for_parts = torch.cat([x, sam_features, output_body, output_up_low], dim=1)
        # x_for_parts = self.parts_conv1d(x_for_parts)

        # output_parts = self.model_parts(x_for_parts)
        # output_parts = self.model.encoder(x_for_parts)[-1]
        # output_parts = self.decoder_parts(output_parts)
        output_parts = self.decoder_parts(output_features_body)

        return output_body, output_up_low, output_parts

    def on_train_epoch_start(self):
        # freeze/unfreeze specific layers for more optimizing training
        self.freeze_pipeline()

        for name, param in self.named_parameters():
            print(name, param.requires_grad)


    def on_validation_epoch_start(self):
        # freeze/unfreeze specific layers for more optimizing training
        self.freeze_pipeline()

        for name, param in self.named_parameters():
            print(name, param.requires_grad)


    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.
        """
        
        # freeze/unfreeze specific layers for more optimizing training
        # self.freeze_pipeline()

        images, masks = batch
        out_body, out_up_low, out_parts = self(images)

        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_body = self.cross_entropy_loss(out_body, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(out_up_low, truth_up_low_mask.squeeze(1))
        loss_parts = self.cross_entropy_loss(out_parts, masks.squeeze(1))

        total_loss = 0.4*loss_body + 0.2*loss_up_low + 0.4*loss_parts
        
        jaccard_loss_body = self.jaccard_loss(out_body, truth_body_mask.squeeze(1))
        jaccard_loss_up_low = self.jaccard_loss(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_loss_parts = self.jaccard_loss(out_parts, masks.squeeze(1))
        
        total_jaccard_loss = jaccard_loss_body + jaccard_loss_up_low + jaccard_loss_parts

        jaccard_index_body = self.jaccard_index_body(out_body, truth_body_mask.squeeze(1))       
        jaccard_index_up_low = self.jaccard_index_up_low(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_index_parts = self.jaccard_index_parts(out_parts, masks.squeeze(1))
        
        self.log('train_loss', total_loss,  prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_body_loss', loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_up_low_loss', loss_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_parts_loss', loss_parts, prog_bar=True, on_step=True, on_epoch=True)

        self.log('train_jaccard_loss_mIoU_0', jaccard_loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_loss_mIoU_1', jaccard_loss_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_loss_mIoU_2', jaccard_loss_parts, prog_bar=True, on_step=True, on_epoch=True)

        self.log('train_jaccard_index_mIoU_0', jaccard_index_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_index_mIoU_1', jaccard_index_up_low, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_index_mIoU_2', jaccard_index_parts, prog_bar=True, on_step=True, on_epoch=True)

        # return different losses according to freeze pipeline
        if self.current_epoch <= 1:
            return loss_body + loss_up_low

        elif self.current_epoch <= 3:
            return loss_parts
        
        elif self.current_epoch <= 6:
            return total_loss
        
        else:
            return loss_parts

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the model.
        """
        # self.freeze_pipeline()

        images, masks = batch
        out_body, out_up_low, out_parts = self(images)


        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_body = self.cross_entropy_loss(out_body, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(out_up_low, truth_up_low_mask.squeeze(1))
        loss_parts = self.cross_entropy_loss(out_parts, masks.squeeze(1))

        total_loss = 0.4*loss_body + 0.2*loss_up_low + 0.4*loss_parts
        
        jaccard_loss_body = self.jaccard_loss(out_body, truth_body_mask.squeeze(1))
        jaccard_loss_up_low = self.jaccard_loss(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_loss_parts = self.jaccard_loss(out_parts, masks.squeeze(1))
        
        total_jaccard_loss = jaccard_loss_body + jaccard_loss_up_low + jaccard_loss_parts
        
        jaccard_index_body = self.jaccard_index_body(out_body, truth_body_mask.squeeze(1))       
        jaccard_index_up_low = self.jaccard_index_up_low(out_up_low, truth_up_low_mask.squeeze(1))
        jaccard_index_parts = self.jaccard_index_parts(out_parts, masks.squeeze(1))
        
        self.log('val_loss', total_loss,  prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_body_loss', loss_body, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_up_low_loss', loss_up_low, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_parts_loss', loss_parts, prog_bar=True, on_step=False, on_epoch=True)

        self.log('val_jaccard_loss_mIoU_0', jaccard_loss_body, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_jaccard_loss_mIoU_1', jaccard_loss_up_low, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_jaccard_loss_mIoU_2', jaccard_loss_parts, prog_bar=True, on_step=False, on_epoch=True)

        self.log('val_jaccard_index_mIoU_0', jaccard_index_body, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_jaccard_index_mIoU_1', jaccard_index_up_low, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_jaccard_index_mIoU_2', jaccard_index_parts, prog_bar=True, on_step=False, on_epoch=True)


        # return different losses according to freeze pipeline
        if self.current_epoch <= 1:
            return loss_body + loss_up_low

        elif self.current_epoch <= 3:
            return loss_parts
        
        elif self.current_epoch <= 6:
            return total_loss
        
        else:
            return loss_parts


    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        # scheduler = lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=5,
        #     gamma=0.3,
        # )
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=50,
        )

        return [optimizer], [scheduler]






