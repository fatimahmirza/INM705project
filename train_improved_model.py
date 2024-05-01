
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
import time
import os
from utility_funcs import *
from loss_func import LossFunc
from data_loader import VOCDataset

from models import Resnet50TL

config = load_config("config.yaml")
print(config)
seed = config['models_config']['seed']

torch.manual_seed(seed)
# Hyperparameters etc.

# 2e-5 is too large of learning rate for transfer learning.
# 1e-5 is converging but very slowly.
# 1e-3 is the optimal learning rate for this transfer learning model. because it is converging towards solution swiftly and without any irregular jumps.
LEARNING_RATE = float(config['models_config']['improved']['learning_rate'])

# DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE = torch.device('cpu')
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
print('DEVICE - >', DEVICE)

# 64
BATCH_SIZE = int(config['models_config']['batch_size'])
WEIGHT_DECAY = int(config['models_config']['weight_decay'])
EPOCHS = int(config['models_config']['epochs'])
API_KEY = str(config['models_config']['API_KEY'])

CHECKPOINT_MODEL = float(config['models_config']['checkpoint_mAP'])
os.environ["WANDB_API_KEY"] = API_KEY

NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


# full data paths
# IMG_DIR = "data/images_labels"
# LABEL_DIR = "data/labels"

# sample data paths
IMG_DIR = "data/data_200/images_200"
LABEL_DIR = "data/data_200/labels_200"

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    mean_loss_value = sum(mean_loss) / len(mean_loss)
    print(f"Mean loss was {mean_loss_value}")
    return mean_loss_value  # Return the calculated loss value


def main():
    model = Resnet50TL(freeze_layers=10).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    loss_fn = LossFunc()

    train_dataset = VOCDataset(
        "data/data_200/train_data_100.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    # wandb.init  initialized a project named as 'train_yolo' in my profile and make create run which will log each and every training cycle metrics.
    # you will be asked to enter your API key.
    wandb.init(project='improved_model')

    for epoch in range(EPOCHS):
        print(f"epoch_num - > {epoch}")

        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")

        # see if mAP is greater than or equals to CHECKPOINT_MODEL%
        if mean_avg_prec >= CHECKPOINT_MODEL:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

           
            # model will be only saved when mAP is greater than CHECKPOINT_MODEL%
            model_save_path = r"saved_models/improved_model_" + \
                str(epoch)+"_"+str(mean_avg_prec)+".pth.tar"
            save_checkpoint(checkpoint, filename=model_save_path)
            time.sleep(10)

        mean_loss = train_fn(train_loader, model, optimizer, loss_fn)

        # metrics are logged per epoch. every point in the wandb graph represents an epoch.
        wandb.log({'mean_loss': mean_loss, 'mean_avg_prec': mean_avg_prec})

    wandb.finish()


if __name__ == "__main__":
    main()
