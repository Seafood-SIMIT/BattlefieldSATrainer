"""basic Lightning modules on which other modules can be built"""

import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
from torchmetrics import Perplexity
class GPT2_BaseLitModel(pl.LightningModule):
    """
    Generic Pytorch-Lightning class that must be initialized with a Pytorch module
    """

    def __init__(self, model, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = args


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),lr=self.args.lr)
        return optimizer


    def forward(self, x):
        output = self.model(x)
        return output.logits

    def predict(self, x):
        output = self.model.generate(**x,
                                    max_length=256,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    do_sample=True,
                                    top_p=0.8,
                                    eos_token_id=50256,
                                    pad_token_id=0,
                                    num_return_sequences=5,
                                    no_repeat_ngram_size=2,
                                    temperature=0.9,
                                    )
        return output

    def training_step(self, batch, batch_idx):
        x,y,logits, loss = self._run_on_batch(batch)
        #train step define the train loop

        #self.train_acc(logits,y)
        self.log("train/loss", loss)
        #self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        return loss

    def _run_on_batch(self, batch, with_preds=False):
        inputs_ids,attention_mask, label = batch
        #output = self.model(**inputs, labels=label)
        output = self.model(inputs_ids,attention_mask=attention_mask,labels=label)
        loss, logits = output[:2]
        return inputs_ids, label, logits, loss

    def validation_step(self, batch, batch_idx):
        x,y,logits, loss = self._run_on_batch(batch)

        #self.val_acc(logits, y)
        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        #self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        #self.test_acc(logits, y)

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        #self.log("test/acc", self.test_acc, on_step=False, on_epoch=True) 
