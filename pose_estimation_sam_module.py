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
from ultralytics.engine.results import Masks, Results


class PascalPartModel(pl.LightningModule):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', num_classes=7, learning_rate=1e-3, transform=None):
        super(PascalPartModel, self).__init__()

        # Loss functions
        # self.bce_loss = nn.BCEWithLogitsLoss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=0)
        self.jaccard_loss = JaccardLoss(mode='multiclass', from_logits=True)
        self.jaccard_index = MulticlassJaccardIndex(num_classes=num_classes, ignore_index=0)

        self.learning_rate = learning_rate
        self.transform = transform

        # Create a shared encoder using SMP (Segmentation Models PyTorch)
        self.encoder = smp.DeepLabV3(
            encoder_name=encoder_name,        
            encoder_weights=encoder_weights,  
            classes=num_classes,                     
            activation=None                   
        )

        # Fusion Module: Combine image features and mask features
        self.fusion_conv1d = nn.Conv2d(in_channels=4, out_channels=3, kernel_size=1)

        self.decoder = self.create_decoder(self.encoder, num_classes)


    def create_decoder(self, encoder, num_classes):
        """
        Creates a decoder using the same architecture as the main encoder.
        """
        return nn.Sequential(
            encoder.decoder,
            nn.Conv2d(self.encoder.decoder.out_channels, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=8.0)
        )


    def get_features_mask(self, x):
        
        device = 'cuda' if x.get_device() >= 0 else 'cpu'
        # tmp_transform = A.Compose([
        #     A.Resize(640, 640),
        # ])

        x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # x = tmp_transform(image=x)['image']#.numpy()
        x = [pic for pic in x]
        
        # Load a model for pose estimation
        model_pose = YOLO("yolov8x-pose.pt") 

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

        # print(res_pose)
        
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
        model_sam = FastSAM("FastSAM-x.pt")  # "FastSAM-x.pt"

        # x = x.astype(np.uint8)

        # rum SAM model
        # res_seg = model_sam.predict(
        #     source=x,
        #     # device='gpu',
        #     # iou=0.99,
        #     # imgsz=640,
        #     imgsz=256,
        #     points=points, 
        #     labels=labels, 
        #     # bboxes=bboxes,
        #     # conf=0.01,
        #     texts='human, body parts'
        #     # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
        # )

        res_seg = []
        
        for i in range(len(points)):
            try:
                tmp_res = model_sam.predict(
                    source=x[i],
                    # device='gpu',
                    half=True,
                    verbose=False,
                    iou=0.99,
                    # imgsz=640,
                    imgsz=256,
                    points=points[i], 
                    labels=labels[i], 
                    bboxes=bboxes[i],
                    # conf=0.01,
                    texts='human, body parts'
                    # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
                )
                
                res_seg.append(tmp_res)

            except Exception:
                print('try without text promt')
                try:
                    tmp_res = model_sam.predict(
                        source=x[i],
                        # device='gpu',
                        half=True,
                        verbose=False,
                        iou=0.99,
                        # imgsz=640,
                        imgsz=256,
                        points=points[i], 
                        labels=labels[i], 
                        bboxes=bboxes[i],
                        # conf=0.01,
                        # texts='human, body parts'
                        # texts= 'humans. for each human. upper body. lower body. low hand, up hand, torso, head, low leg, up leg'
                    )

                    res_seg.append(tmp_res)
                
                except Exception: 
                    # tmp_masks = Masks(masks=torch.zeros( 256, 256), orig_shape=(256, 256))
                    # tmp_res = Results(orig_img=x[i], path='', names={0: '0'},  masks=Masks(masks=torch.zeros(256, 256), orig_shape=(256, 256)))
                    # print(tmp_res)
                    # res_seg.append(tmp)
                    # print('added zeros feature map')
                    
                    pass
                    # final_mask = torch.zeros(len(x), 1, 256, 256).to('cuda')
                    # return final_mask

        if len(res_seg) == 0:
            final_mask = torch.zeros(len(x), 1, 256, 256).to('cuda')
            return final_mask
        

        # convert to final mask
        # final_mask = torch.sum(res_seg[0].masks.data.cpu(), dim=0)
        # final_mask = [torch.sum(r.masks.data.cpu(), dim=0).numpy() for r in res_seg[0]]
        
        final_mask = [torch.sum(img[0].masks.data.cpu(), dim=0).numpy() for img in res_seg]
        if len(final_mask) != len(x):
            for _ in range(len(x) - len(final_mask)):
                final_mask.append(np.zeros((256, 256)))

        final_mask = np.array(final_mask)

        # transform final mask same as initial image
        # final_mask = self.transform[0](image=final_mask)

        final_mask = torch.tensor(final_mask)
        final_mask = final_mask.unsqueeze(1)

        # x = torch.tensor(x).to('cuda')
        # device = 'gpu' if x.get_device() >= 0 else 'cpu'
        final_mask = torch.tensor(final_mask).to(device)

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
        """
        upper_lower_body_mask = torch.zeros_like(masks).long()  
        upper_lower_body_mask[(masks == 1) | (masks == 2) | (masks == 4) | (masks == 6)] = 1
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


    def forward(self, x):
        device = 'cuda' if x.get_device() >= 0 else 'cpu'
        # extract features using pose estimation and SAM
        sam_features = self.get_features_mask(x)


        # x = x.to('cuda')
        # sam_features = sam_features.to('cuda')
        # concat image and extracted features from SAM
        
        print(x.shape, sam_features.shape)
        # x = torch.cat([x, sam_features], dim=1)
        x = x + sam_features
        # print(x.dtype, x.get_device(), type(x))
        x = torch.tensor(x, dtype=torch.float32).to(device)
        # print(x.dtype, x.get_device())
        # .to(device) 

        # TODO try torch.sum instead of torch.cat and fusion_conv1d !!??

        # x = self.fusion_conv1d(x)

        # Pass the input through the encoder
        output = self.encoder(x)


        return output


    def training_step(self, batch, batch_idx):
        """
        Defines the training step for the model.
        """

        images, masks = batch
        body_parts_out = self(images)


        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        pred_body_mask = self._convert_parts_to_body(body_parts_out)
        pred_up_low_mask = self._convert_parts_to_up_low(body_parts_out)


        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_parts = self.cross_entropy_loss(body_parts_out, masks.squeeze(1))

        loss_body = self.cross_entropy_loss(pred_body_mask, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(pred_up_low_mask, truth_up_low_mask.squeeze(1))

        total_loss = loss_parts + loss_body
        

        jaccard_loss_parts = self.jaccard_loss(body_parts_out, masks.squeeze(1))
        jaccard_index_parts = self.jaccard_index(body_parts_out, masks.squeeze(1))

        jaccard_index_boby = self.jaccard_index(pred_body_mask, truth_body_mask.squeeze(1))
        jaccard_index_up_low = self.jaccard_index(pred_up_low_mask, truth_up_low_mask.squeeze(1))

        self.log('train_loss', total_loss,  prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_body_loss', loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_parts_loss', loss_parts, prog_bar=True, on_step=True, on_epoch=True)

        self.log('train_jaccard_index_mIoU_0', jaccard_index_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_index_mIoU_1', jaccard_index_up_low, prog_bar=True, on_step=True, on_epoch=True)

        self.log('train_jaccard_loss_mIoU_2', jaccard_loss_parts, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_jaccard_index_mIoU_2', jaccard_index, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step for the model.
        """

        # images, masks = batch
        # body_parts_out = self(images)

        # # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        # loss = self.cross_entropy_loss(body_parts_out, masks.squeeze(1))

        # jaccard_loss = self.jaccard_loss(body_parts_out, masks.squeeze(1))
        # jaccard_index = self.jaccard_index(body_parts_out, masks.squeeze(1))

        # self.log('val_loss', loss,  prog_bar=True, on_step=True, on_epoch=True)
        # self.log('val_jaccard_loss_mIoU_2', jaccard_loss, prog_bar=True, on_step=True, on_epoch=True)
        # self.log('val_jaccard_index_mIoU_2', jaccard_index, prog_bar=True, on_step=True, on_epoch=True)

        # return loss

        images, masks = batch
        body_parts_out = self(images)


        truth_body_mask = self._get_body_mask(masks)
        truth_up_low_mask = self._get_upper_lower_body_mask(masks)

        pred_body_mask = self._convert_parts_to_body(body_parts_out)
        pred_up_low_mask = self._convert_parts_to_up_low(body_parts_out)

        # Calculate loss for specific body parts within the relevant masks (mIoU^2)
        loss_parts = self.cross_entropy_loss(body_parts_out, masks.squeeze(1))

        loss_body = self.cross_entropy_loss(pred_body_mask, truth_body_mask.squeeze(1))
        loss_up_low = self.cross_entropy_loss(pred_up_low_mask, truth_up_low_mask.squeeze(1))

        total_loss = loss_parts + loss_body
        
        jaccard_loss_parts = self.jaccard_loss(body_parts_out, masks.squeeze(1))
        jaccard_index_parts = self.jaccard_index(body_parts_out, masks.squeeze(1))

        jaccard_index_boby = self.jaccard_index(pred_body_mask, truth_body_mask.squeeze(1))
        jaccard_index_up_low = self.jaccard_index(pred_up_low_mask, truth_up_low_mask.squeeze(1))

        self.log('val_loss', total_loss,  prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_body_loss', loss_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_parts_loss', loss_parts, prog_bar=True, on_step=True, on_epoch=True)

        self.log('val_jaccard_index_mIoU_0', jaccard_index_body, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_jaccard_index_mIoU_1', jaccard_index_up_low, prog_bar=True, on_step=True, on_epoch=True)

        self.log('val_jaccard_loss_mIoU_2', jaccard_loss_parts, prog_bar=True, on_step=True, on_epoch=True)
        self.log('val_jaccard_index_mIoU_2', jaccard_index, prog_bar=True, on_step=True, on_epoch=True)


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
            gamma=0.3,
        )

        return [optimizer], [scheduler]






