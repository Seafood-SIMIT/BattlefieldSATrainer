import os
from torch import optim
import pytorch_lightning as pl
from torchmetrics import Perplexity
from transformers import GPT2LMHeadModel
LR = 1e-3

class GPT2Chinese(pl.LightningModule):
    def __init__(self,args):
        super().__init__()

        self.gpt2 = GPT2LMHeadModel.from_pretrained(args.final_path)

        self.train_acc = Perplexity(ignore_index=-100)
        self.val_acc = Perplexity(ignore_index=-100)
        self.test_acc = Perplexity(ignore_index=-100)

    def forward(self,input_ids, length):
        out = self.gpt2(input_ids)
        return out


    def training_step(self, batch, batch_idx):
        x,y,logits, loss = self._run_on_batch(batch)
        #train step define the train loop

        self.train_acc(logits,y)
        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def _run_on_batch(self, batch, with_preds=False):
        x,y = batch
        out = self.gpt2(x,labels = y)
        loss, logits = out[:2]
        return x, y, logits, loss

    def validation_step(self,batch,batch_idx):
        x,y,logits, loss = self._run_on_batch(batch)

        self.val_acc(logits, y)
        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),lr=LR)
        return optimizer

