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

    #     self.decoder = self.create_decoder(self.encoder, num_classes)


    # def create_decoder(self, encoder, num_classes):
    #     """
    #     Creates a decoder using the same architecture as the main encoder.
    #     """
    #     return nn.Sequential(
    #         encoder.decoder,
    #         nn.Conv2d(self.encoder.decoder.out_channels, num_classes, kernel_size=1),
    #         nn.UpsamplingBilinear2d(scale_factor=8.0)
    #     )


    def get_features_mask(self, x):
        
        device = 'cuda' if x.get_device() >= 0 else 'cpu'
        # tmp_transform = A.Compose([
        #     A.Resize(640, 640),
        # ])
        
        # x -> (N, C, W, H)
        # (N, W, H, C)
        # print(x.shape)
        # print(x.dtype)

        x = x.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)

        # x = tmp_transform(image=x)['image']#.numpy()
        x = [pic for pic in x]
        

        # print(x.shape)
        # print(x.dtype)


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
        # for r in res_pose:
        #     tmp = r.keypoints.xy.reshape((-1, 2)).cpu().tolist()
        #     points.extend(tmp)
        #     labels.extend(np.arange(len(tmp)).tolist())
        #     bboxes.extend(r.boxes.xywh.reshape((-1, 4)).cpu().tolist())

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


        # print(len(points), len(labels), len(bboxes))
        # print(points[0].shape, labels[0].shape, bboxes[0].shape)
        # print(points[1].shape, labels[1].shape, bboxes[1].shape)

        # print(np.array(points).shape, np.array(labels).shape, np.array(bboxes).shape)
        
        # print(points)
        # print(labels)
        # print(bboxes)

        # # extract keypoints
        # points = res[0].keypoints.xy.cpu().tolist()
        # # extract bounding boxes
        # bboxes = res[0].boxes.data.cpu().tolist()

        # Create a FastSAM model
        model_sam = FastSAM("FastSAM-s.pt")  # "FastSAM-x.pt"

        # x = x.astype(np.uint8)

        # print(x[0].shape)
        # print(x[0].dtype)
        # print(np.array(x).shape)

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

        try:
            res_seg = [model_sam.predict(
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
            for i in range(len(points))
            ]
        except Exception:
            print(len(x), len(points), len(labels), len(bboxes))
            final_mask = torch.zeros(len(x), 1, 256, 256).to('cuda')
            return final_mask

        # print('sam finish')

        # # print(res_seg)
        # print(len(res_seg))
        # print(res_seg[0])
        # print(res_seg[1])

        # convert to final mask
        # final_mask = torch.sum(res_seg[0].masks.data.cpu(), dim=0)
        # final_mask = [torch.sum(r.masks.data.cpu(), dim=0).numpy() for r in res_seg[0]]
        
        final_mask = [torch.sum(img[0].masks.data.cpu(), dim=0).numpy() for img in res_seg]
        final_mask = np.array(final_mask)

        # print(final_mask.shape)

        # transform final mask same as initial image
        # final_mask = self.transform[0](image=final_mask)
        # print(final_mask['image'].shape)

        final_mask = torch.tensor(final_mask)
        # print(final_mask.shape)
        final_mask = final_mask.unsqueeze(1)

        # x = torch.tensor(x).to('cuda')
        # device = 'gpu' if x.get_device() >= 0 else 'cpu'
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


        # print(x.get_device(), sam_features.get_device())
        # print(x.shape)
        # print(sam_features.shape)

        # x = x.to('cuda')
        # sam_features = sam_features.to('cuda')
        # concat image and extracted features from SAM
        
        # print(x.get_device(), sam_features.get_device())
        
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

        # print(pred_body_out.shape, true_body_masks.shape)
        # print(true_body_masks.squeeze(1).shape)
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