import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS, STEP_OUTPUT
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import os

from models.GooGleNet import GoogleNet

# Seed everything
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class CNNClassifier(pl.LightningModule):
    def __init__(self, model_name: str, model_hparams: dict, optimizer_name: str, optimizer_hparams: dict):
        super().__init__()
        self.save_hyperparameters()
        self.model = self._create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def _create_model(self, model_name, model_hparams):
        if model_name=='GoogleNet':
            return GoogleNet(**model_hparams)
        else:
            raise Exception(f'Unknown model : {model_name}')

    def configure_optimizers(self):
        if self.hparams.optimizer_name=='SGD':
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name=='Adam':
            optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise Exception(f'Unknown optimizer : {self.hparams.optimizer_name}')
        
        # # We will reduce the learning rate by 0.1 after 100 and 150 epochs
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        # return [optimizer], [scheduler]
        return optimizer

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)

        acc = (preds.argmax(dim=-1)==labels).float().mean()

        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels==preds).float().mean()

        self.log('val_acc', acc)
    
    def test_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels==preds).float().mean()
        
        self.log('test_acc', acc)


class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(self):
        # Check dataset path
        self.DATASET_PATH = './data/CIFAR10'
        if not os.path.isdir(self.DATASET_PATH):
            os.makedirs(self.DATASET_PATH, exist_ok=True)

    def setup(self) -> None:
        train_data = CIFAR10(root=self.DATASET_PATH, train=True, download=True)
        self.DATA_MEAN = (train_data.data / 255.0).mean(axis=(0, 1, 2))
        self.DATA_STD = (train_data.data / 255.0).std(axis=(0, 1, 2))

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.ToTensor(),
                transforms.Normalize(self.DATA_MEAN, self.DATA_STD)
            ])
        
        train_dataset = CIFAR10(root=self.DATASET_PATH, train=True, transform=train_transform, download=True)
        pl.seed_everything(42)
        self.train_data, _ = random_split(train_dataset, [45000, 5000])
        self.train_loader = DataLoader(self.train_data, batch_size=64, shuffle=True, pin_memory=True)
        return self.train_loader
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.DATA_MEAN, self.DATA_STD)]
        )
        val_dataset = CIFAR10(root=self.DATASET_PATH, train=True, transform=val_transform, download=True)
        pl.seed_everything(42)
        _, self.val_data = random_split(val_dataset, [45000, 5000])
        self.val_loader = DataLoader(self.val_data, batch_size=64, shuffle=False, pin_memory=True)
        return self.val_loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.DATA_MEAN, self.DATA_STD)]
        )

        self.test_data = CIFAR10(root=self.DATASET_PATH, train=False, transforms=test_transform, download=True)
        self.test_loader = DataLoader(self.test_data, batch_size=64, shuffle=False, pin_memory=True)
        return self.test_loader

def train_model(model_name, CHECKPOINT_PATH, save_name=None, **kwargs):
    """
    Inputs:
        model_name - Name of the model you want to run. Is used to look up the class in "model_dict"
        save_name (optional) - If specified, this name will be used for creating the checkpoint and logging directory.
    """
    if save_name is None:
        save_name = model_name
    
    if not os.path.isdir(os.path.join(CHECKPOINT_PATH, save_name)):
        os.makedirs(os.path.join(CHECKPOINT_PATH, save_name), exist_ok=True)
    
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, save_name),
        gpus=1 if torch.cuda.is_available() else 0,
        max_epochs=200,
        callbacks=[
            ModelCheckpoint(mode='max', monitor='val_acc'),
            LearningRateMonitor('epoch'), # log learning rate every epoch
        ],  
    )

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(CHECKPOINT_PATH, save_name + ".ckpt")

    # if os.path.isfile(pretrained_filename):
    #     print(f"Found pretrained model at {pretrained_filename}, loading...")
    #     # Automatically loads the model with the saved hyperparameters
    #     model = CNNClassifier.load_from_checkpoint(pretrained_filename)
    # else:
    #     pl.seed_everything(42)  # To be reproducable
    #     model = CNNClassifier(model_name=model_name, **kwargs)
    #     trainer.fit(model, CIFAR10DataModule)
    #     model = CNNClassifier.load_from_checkpoint(
    #         trainer.checkpoint_callback.best_model_path
    #     )  # Load best checkpoint after training
    
    # # Test best model on validation and test set
    # results = trainer.test(model, datamodule=CIFAR10DataModule, verbose=False)
    # print(results)
    # result = {"test": results[0]["test_acc"], "val": results[0]["test_acc"]}

    # return model, result