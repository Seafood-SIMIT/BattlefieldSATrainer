"""basic Lightning modules on which other modules can be built"""

import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

import jieba
class KG_BaseLitModel(pl.LightningModule):
    """
    Generic Pytorch-Lightning class that must be initialized with a Pytorch module
    """

    def __init__(self, model,tokenizer, args: argparse.Namespace = None):
        super().__init__()
        self.model = model
        self.args = args
        self.tokenizer = tokenizer


        optimizer = self.args.optimizer

        self.loss_fn = torch.nn.CrossEntropyLoss()
        #self.train_acc = Accuracy()
        #self.val_acc = Accuracy()
        #self.test_acc = Accuracy()



    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.parameters(),lr=self.args.lr)
        return optimizer

    def forward(self, x):
        output = self.model(x)
        predicted_labels = torch.argmax(output.logits, axis=-1)
        print(predicted_labels)
        loss = self.loss_fn(predicted_labels)
        return loss, predicted_labels
    
    def predict(self, texts):
        for text in texts:
            jieba.load_userdict("files/tokenizer/jieba_dict.txt")
            words = list(jieba.cut(text))
            for word in words:
                if word not in self.tokenizer.vocab:
                    self.tokenizer.add_tokens([word])
            inputs = self.tokenizer.encode_plus(words,
                                    add_special_tokens=True,
                                        return_attention_mask=True,
                                        return_tensors="pt")
            print(inputs)
            predict_labels = self.model(**inputs).logits
            predict_labels = predict_labels.argmax(-1)
            print(predict_labels)

            labels = []
            for pred in predict_labels[0]:
                if pred == 0:
                    label='N'
                elif pred == 1:
                    label='E'
                elif pred == 2:
                    label = 'R'
                else:
                    label='S'
                labels.append[label]
            
            #assert len(batch.)

            print(labels) 

    def training_step(self, batch, batch_idx):
        predicted_labels, loss = self._run_on_batch(batch)

        self.log("train/loss", loss)

        outputs = {"predicted_labels":predicted_labels,"loss": loss}

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        inputs_ids,attention_mask, label = batch
        #output = self.model(**inputs, labels=label)
        output = self.model(inputs_ids,attention_mask=attention_mask,labels=label)
        predicted_labels = torch.argmax(output.logits, axis=-1)
        #print(predicted_labels, label)
        loss = output.loss
        return predicted_labels, loss

    def validation_step(self, batch, batch_idx):
        predicted_labels, loss = self._run_on_batch(batch)


        outputs = {"predicted_labels":predicted_labels,"loss": loss}

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)

        return outputs

    def test_step(self, batch, batch_idx):
        predict_labels = self.model(**batch).logits
        predict_labels = predict_labels.argmax(-1)

        labels = []
        for pred in predict_labels:
            if pred == 0:
                label='N'
            elif pred == 1:
                label='E'
            elif pred == 2:
                label = 'R'
            else:
                label='S'
            labels.append[label]
        
        #assert len(batch.)

        print(predict_labels)