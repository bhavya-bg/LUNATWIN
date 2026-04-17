# train.py (Improved version)

import numpy as np
import torch
from dataset import LUNA16_Dataset, collate_fn
from transformers import ViTConfig
from model import VitDet3D
from transformers import TrainingArguments, Trainer
from sklearn.metrics import f1_score
import os

def compute_metrics(eval_pred):
    predictions, groundtruth = eval_pred
    logits = predictions[0]
    labels = groundtruth[0]
    bbox_pred = predictions[1]
    bbox_gt = groundtruth[1]

    preds = (logits > 0).astype(int)
    f1 = f1_score(labels, preds, average='binary')
    
    # IoU only on positive samples
    mask = labels.astype(bool)
    if np.sum(mask) > 0:
        iou = iou_3d(bbox_pred[mask], bbox_gt[mask])  # iou_3d from dataset.py
    else:
        iou = 0.0
        
    return {"f1": f1, "iou": iou, "accuracy": (preds == labels).mean()}

# ========================== CONFIG ==========================
data_dir = "datasets/luna16"          # apna preprocessed data yahan ho
model_dir = "luna-train/final_run"    # better folder name
log_dir = "logs/final_run"

# Better config for training
config = ViTConfig.from_pretrained("model_config.json")

# Use 9 subsets for training, 1 for validation (10-fold style)
train_split = list(range(9))
valid_split = [9]

print("Preparing datasets...")
train_dataset = LUNA16_Dataset(
    split=train_split, 
    data_dir=data_dir, 
    crop_size=config.image_size, 
    patch_size=config.patch_size, 
    samples_per_img=16   # increase if GPU allows
).train()

valid_dataset = LUNA16_Dataset(
    split=valid_split, 
    data_dir=data_dir, 
    crop_size=config.image_size, 
    patch_size=config.patch_size, 
    samples_per_img=8
).eval()

print("Loading model...")
model = VitDet3D(config)

# ========================== TRAINING ARGS ==========================
args = TrainingArguments(
    output_dir=model_dir,
    eval_strategy="steps",
    eval_steps=2000,
    logging_steps=200,
    save_steps=2000,
    save_total_limit=3,
    learning_rate=1e-5,           # lower LR for stability in 3D ViT
    per_device_train_batch_size=2,   # 3D model hai, GPU VRAM dekho (increase slowly)
    per_device_eval_batch_size=2,
    num_train_epochs=20,             # ya max_steps=100000 use karo
    weight_decay=0.01,
    fp16=True,                       # agar GPU support kare
    dataloader_num_workers=4,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="tensorboard",
    remove_unused_columns=False,
    label_names=["labels", "bbox"],
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train(resume_from_checkpoint=False)   # True if resume karna ho