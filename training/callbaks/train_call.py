import torch
from pytorch_lightning import Callback
from torch.utils.tensorboard import SummaryWriter


class TrainResultLogger(Callback):
    def __init__(self):
        super().__init__()
        self.writer = None

    def on_train_start(self, trainer, pl_module):
        # 初始化 SummaryWriter
        self.writer = SummaryWriter(log_dir=trainer.log_dir)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        # 获取训练过程的损失
        train_loss = torch.stack([x['loss'] for x in outputs]).mean()

        # 记录训练过程的损失到 TensorBoard
        self.writer.add_scalar("train_loss", train_loss, trainer.current_epoch)

    def on_train_end(self, trainer, pl_module):
        # 关闭 SummaryWriter
        self.writer.close()
