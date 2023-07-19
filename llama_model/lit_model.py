import torch
import pytorch_lightning as pl
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.optimization import AdamW
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
#from torch.optim import AdamW
from .llama_generate import generate
SHOW_DATA=False
class LlamaModule(pl.LightningModule):
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
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': self.args_litmodel.weight_decay},
            {'params': [p for n, p in self.named_parameters() if any(
                nd in n for nd in no_decay) and p.requires_grad], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_params, lr=self.args_litmodel.learning_rate)
        #optimizer = FusedAdam(optimizer_grouped_params, lr=self.args_litmodel.learning_rate,adam_w_mode=True,)
        #                  betas=(self.args_litmodel.adam_beta1, self.args_litmodel.adam_beta2),
        #                  eps=self.args_litmodel.adam_epsilon)

        return [optimizer]

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
                print('source: {}'.format(self.tokenizer.decode(batch['input_ids'][0])))
                label_idx = batch['labels'][0] != -100
                print('target: {}'.format(self.tokenizer.decode(
                    batch['labels'][0][label_idx])))
                print('mask: {}'.format(batch['attention_mask'][0]))
                print('position_ids: {}'.format(batch['position_ids'][0]))
        output = self(**batch)
        self.log('train_loss', output.loss, on_epoch=True, prog_bar=True,logger=True)
        return output.loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        self.log('val_loss', output.loss, on_epoch=True, prog_bar=True,logger=True)
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