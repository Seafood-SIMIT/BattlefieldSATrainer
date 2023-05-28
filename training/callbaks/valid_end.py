import torch
from pytorch_lightning import Callback
from torch.utils.tensorboard import SummaryWriter


class ValidationResultLogger(Callback):
    def __init__(self):
        super().__init__()
        self.writer = None
    def on_validation_start(self, trainer, pl_module):
        # 初始化 SummaryWriter
        self.writer = SummaryWriter(log_dir=trainer.log_dir)

    def on_validation_epoch_end(self, trainer, pl_module):
        # 获取验证过程的输出
        outputs = trainer.callback_metrics

        # 记录验证过程的输出到 TensorBoard
        for metric_name, metric_value in outputs.items():
            self.writer.add_scalar(
                f"validation_step/{metric_name}", metric_value, trainer.current_epoch
            )

    def on_validation_end(self, trainer, pl_module):
        # 关闭 SummaryWriter
        self.writer.close()
