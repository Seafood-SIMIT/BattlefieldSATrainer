import torch
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup

class Llama(pl.LightningModule):
    @staticmethod
    def add_module_specific_args(parent_parser):
        parser = parent_parser.add_argument_group('ziya_llama finetune')
        parser.add_argument('--max_seq_length', type=int, default=1024)
        parser.add_argument('--model_parallel_size', type=int, default=1)
        parser.add_argument('--tokenizer_path', default=None, type=str)
        return parent_parser

    def __init__(self, args, tokenizer):
        super().__init__()
        self.save_hyperparameters(args)
        self.tokenizer = tokenizer

    def setup(self, stage) -> None:
        self.total_step = int(self.trainer.max_epochs * self.num_data
                                  / self.trainer.accumulate_grad_batches)
        print('Total training step:', self.total_step)


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

    def forward(self, **batch):
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
                print('position_ids: {}'.format(batch['position_ids'][0]))
        output = self(**batch)
        self.log('train/loss', output.loss, sync_dist=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, sync_dist=True)
        return output.loss

    def predict_step(self, batch, batch_idx):
        # generate data
        generate_kwargs = {
        	"do_sample": True,
        	"top_p": 1.0,   
        	"top_k": 0,
        	"max_length": 256,
        	"repetition_penalty": 1.0,
        	"temperature": 0.8,
        	"pad_token_id": self.tokenizer.eos_token_id,
        	"eos_token_id": self.tokenizer.eos_token_id,
        }
        batch_input_ids = batch['input_ids'].cpu().numpy().tolist()
        print('batch_input_ids:\n', batch_input_ids)
        queries = [self.detokenize(input_ids).split('<bot>:')[0].replace('<s>', '')+'<bot>:' for input_ids in batch_input_ids]
        print('queries:\n', queries)
        # queries = ['怎样给世界一点爱？', '生命的意义是什么？']
        ans = generate(queries=queries,
                tokenizer=self.tokenizer,
                model=self.model,
                device=self.model.device,
                **generate_kwargs)
        print('ans:\n', ans)
        ## end

    def on_load_checkpoint(self, checkpoint) -> None:
        if 'global_samples' in checkpoint:
            self.consumed_samples = checkpoint['global_samples']