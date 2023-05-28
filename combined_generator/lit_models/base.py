"""basic Lightning modules on which other modules can be built"""

import argparse

import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy

import jieba
LR=1e-3
class CombinedBaseModule(pl.LightningModule):
    """
    Generic Pytorch-Lightning class that must be initialized with a Pytorch module
    """

    def __init__(self, model1,model2,tokenizer):
        super().__init__()
        self.bert = model1
        self.gpt2 = model2

        self.tokenizer = tokenizer

        #self.train_acc = Accuracy()
        #self.val_acc = Accuracy()
        #self.test_acc = Accuracy()



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.trainer.model.parameters(),lr=LR)
        return optimizer

    def forward(self, text):
        triplets = self._extract_triplets(text)

        # script a word
        input_text = self._join_triplets(triplets)

        # generate description
        output_text = self._generate_text(input_text)

        return output_text


    def predict(self, x):
        logits = self.module2.forward(self.module1.forward(x))
        return torch.argmax(logits, dim=1)

    def training_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.train_acc(logits, y)

        self.log("train/loss", loss)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)

        outputs = {"loss": loss}

        return outputs

    def _run_on_batch(self, batch, with_preds=False):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        return x, y, logits, loss

    def validation_step(self, batch, batch_idx):
        x, y, logits, loss = self._run_on_batch(batch)
        self.val_acc(logits, y)

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)
        self.log("validation/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        outputs = {"loss": loss}

        return outputs

    def test_step(self, text):
        for i in text:
            output = self(i)
            print(output)
        return output
    
    def _extract_triplets(self,text):
        words = jieba.lcut(text)
        print(words)
        
        tokens=[]
        for word in words:
            sub_tokens = self.tokenizer.tokenize(word)
            if not sub_tokens:
                sub_tokens=['UNK']
            tokens.extend(sub_tokens)
        print(tokens)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask=[1]* len(input_ids)
        print(input_ids,attention_mask)
        encoding = self.tokenizer(text, padding=True)
        input_ids = torch.tensor(encoding['input_ids']).reshape(1,-1)
        #offsets = encoding['offset_mapping']

        output = self.bert(input_ids=input_ids)
        predicted_labels = torch.argmax(output.logits, axis=-1)[0]
        print(predicted_labels, input_ids[0,1])

        entities=[]
        relations=[]
        shuxing=[]
        #words = self.tokenizer.convert_ids_to_tokens(input_ids)[1:-1]
        words=[]
        new_word=None
        for token in range(len(tokens)):
            print(tokens[token])
            if len(tokens[token])==1 and predicted_labels[token]!=0:
                new_word=WORDS(input_ids[0][token], predicted_labels[token])
            elif len(tokens[token]) == 3:
                new_word.add_word(input_ids[0][token])
            else:
                if new_word:
                    words.append(new_word)
                else:
                    new_word=None
                #print(text[label],input_ids[0][label])
                #print(self.tokenizer.convert_ids_to_tokens([input_ids[0][label],]))

        print(words)
        return words

    def _join_triplets(self,triplets):

            #join the info
        input_text=" ".join([f"{relation}[{entity1},{entity2}]" for entity1, relation, entity2 in triplets] )

        input_text = input_text.join(".")

        return input_text

    def _generate_text(self,input_text):
        output = self.gpt2.generate(input_ids=input_text, max_length=200, do_sample=True)

        #Decode and return text
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return generated_text
    

class WORDS():
    def __init__(self,id,type):
        self.word = [id,]
        self.type = type
    def add_word(self,id):
        self.word.append(id)
    
