import os
from typing import Any, Optional
import torch
import argparse
import pytorch_lightning as pl
import deepspeed
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer, loggers
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import GPT2LMHeadModel

class WenzhongQALitModel(pl.LightningModule):

    def __init__(self, args,model, num_data):
        super().__init__()
        self.args = args
        self.num_data = num_data
        print('num_data:', num_data)
        self.model = model
        #self.save_hyperparameters()
        

    def setup(self, stage) -> None:
        if stage == 'fit':
            #num_gpus = self.trainer.gpus if self.trainer.gpus is not None else 0
            self.total_step = int(self.trainer.max_epochs * self.num_data
                                  / (max(1, 1) * self.trainer.accumulate_grad_batches))
            print('Total training step:', self.total_step)

    def forward(self,x):
        output = self.model(**x)
        return output.logits
        
    def predict_backup(self, x):
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
        output = self.model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('train_loss', output.loss,on_epoch=True, prog_bar=True,logger=True)
        return output.loss

    def comput_metrix(self, logits, labels):
        y_pred = torch.argmax(logits, dim=-1)
        y_pred = y_pred.view(size=(-1,))
        y_true = labels.view(size=(-1,)).float()
        corr = torch.eq(y_pred, y_true)
        acc = torch.sum(corr.float()) / labels.size()[0]
        return acc

    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        # output = self.model(input_ids=batch['input_ids'], labels=batch['labels'])
        # acc = self.comput_metrix(output.logits, batch['labels'])
        self.log('val_loss', output.loss,on_epoch=True,prog_bar=True,logger=True)
        # self.log('val_acc', acc)

    def generate(self, x):
        output = self.model(**x)
        return output

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args.learning_rate)
        #optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    
def main():
    total_parser = argparse.ArgumentParser("QA Task")
    total_parser.add_argument('--do_eval_only', action='store_true', default=False)
    total_parser.add_argument('--pretrained_model_path', default='google/mt5-small', type=str)
    total_parser.add_argument('--output_save_path', default='./predict.json', type=str)
    # * Args for data preprocessing
    total_parser = GPT2QADataModel.add_data_specific_args(total_parser)
    # * Args for training
    total_parser = Trainer.add_argparse_args(total_parser)
    total_parser = GPT2FinetuneMedicalQAModelCheckpoint.add_argparse_args(total_parser)
    total_parser = GPT2FinetuneMedicalQA.add_model_specific_args(total_parser)
    # * Args for base model
    args = total_parser.parse_args()

    data_model = GPT2QADataModel(args)
    if not args.do_eval_only:
        model = GPT2FinetuneMedicalQA(args, len(data_model.train_dataloader()))
        checkpoint_callback = GPT2FinetuneMedicalQAModelCheckpoint(args).callbacks
        logger = loggers.TensorBoardLogger(save_dir=os.path.join(
            args.default_root_dir, 'log/'), name='WenZhong')
        trainer = Trainer.from_argparse_args(args,
                                             logger=logger,
                                             callbacks=[checkpoint_callback]
                                             )
        trainer.fit(model, data_model)


if __name__ == '__main__':
    main()