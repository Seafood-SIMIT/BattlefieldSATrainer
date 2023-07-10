import torch
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
SHOW_DATA=False
class Wenzhong3BModule(pl.LightningModule):
    def __init__(self, args,model, tokenizer):
        super().__init__()
        self.args_litmodel = args
        self.model = model
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        train_loader = self.trainer.datamodule.train_dataloader()
        if self.trainer.max_epochs > 0:
            world_size = self.trainer.world_size
            tb_size = self.args_litmodel.train_batchsize * max(1, world_size)
            ab_size = self.trainer.accumulate_grad_batches
            self.total_step = (len(train_loader.dataset) *
                       self.trainer.max_epochs // tb_size) // ab_size
        else:
            self.total_step = self.trainer.max_steps
        print('Total training step:', self.total_step)


    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'layer_norm.', 'layernorm.']
        paras = list(
            filter(lambda p: p[1].requires_grad, self.named_parameters()))
        paras = [{
            'params':
            [p for n, p in paras if not any(nd in n for nd in no_decay)],
            'weight_decay': self.args_litmodel.weight_decay
        }, {
            'params': [p for n, p in paras if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }]
        optimizer = torch.optim.AdamW(paras, lr=self.args_litmodel.learning_rate)
        #optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(paras, lr=self.args.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(self.total_step * self.args_litmodel.warmup),
            self.total_step)

        return [{
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }]

    def forward(self, **batch):
        #print(batch)
        return self.model(**batch)

    def detokenize(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def comput_metrix(self, logits, labels):
        with torch.no_grad():
            y_pred = torch.argmax(logits, dim=-1)
            y_pred = y_pred.view(size=(-1,))
            y_true = labels.view(size=(-1,)).float()
            corr = torch.eq(y_pred, y_true)
            acc = torch.sum(corr.float())/labels.shape[0]
        return acc

    def training_step(self, batch, batch_idx):
        if self.trainer.global_rank == 0:
            global SHOW_DATA
            if not SHOW_DATA:
                SHOW_DATA = True
                print('source: {}'.format(batch['input_ids'][0]))
                print('target: {}'.format(batch['labels'][0]))
                print('source: {}'.format(self.detokenize(batch['input_ids'][0])))
                label_idx = batch['labels'][0] != -100
                print('target: {}'.format(self.detokenize(
                    batch['labels'][0][label_idx])))
                print('mask: {}'.format(batch['attention_mask'][0]))
                #print('position_ids: {}'.format(batch['position_ids'][0]))
        output = self(**batch)
        self.log('train_loss', output.loss, on_epoch=True, prog_bar=True,logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, on_epoch=True, prog_bar=True,logger=True)
        return output.loss
