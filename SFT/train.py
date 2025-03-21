import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from accelerate.utils import DistributedType, set_seed
from tqdm import tqdm
import os
from gsm8k_data import preprocess_gsm8k
class  TextDataset(Dataset):
    def __init__(self, data_dir, max_length):
        self.examples = []
        with open(data_dir, 'r') as f:
            for line in f:
                self.examples.append(line, max_length=max_length, padding='max_length', truncation=True)
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return {"text": self.examples[idx]}
    
def forward_process(input_ids, eps=1e-3,mask_id=126336):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    return noisy_batch, masked_indices, p_mask
def compute_loss(input_ids, model,prompt_length,mask_id=126336):
    noisy_batch, masked_indices, p_mask = forward_process(input_ids,mask_id=mask_id)
    # temp_tensor = torch.arange(noisy_batch.size(1), device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    prompt_mask = (token_positions < prompt_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]

# Calculate the answer length (including the padded <EOS> tokens)
    prompt_mask = prompt_mask.to(torch.int64)    
    answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

    masked_indices = (noisy_batch == mask_id)
    noisy_batch=noisy_batch.to(model.device)
    logits = model(input_ids=noisy_batch).logits
    input_ids=input_ids.to(model.device)
    p_mask=p_mask.to(model.device)
    token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    answer_lengths=answer_lengths.to(model.device)
    ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]
    # logits = model(input_ids=noisy_batch).logits
    # token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
    # loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
    return ce_loss


def prepare(args):
    accelerator = Accelerator(
                              log_with=args.log,
                              split_batches=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,trust_remote_code=True).to(accelerator.device).train()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    dataset= preprocess_gsm8k(tokenizer,args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    for param in model.parameters():
        param.requires_grad = True
        param.to(torch.float32)
    train_parameters=[param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(train_parameters, lr=args.lr,)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (args.batch_size)
    model,optimizer=accelerator.prepare(model,optimizer)
    return model, tokenizer, dataloader,accelerator,optimizer


# def save_model(model, output_dir):

# def sample():

def create_args():
    args=argparse.ArgumentParser()
    args.add_argument('--lr', type=float, default=1e-5,)
    args.add_argument('--batch_size', type=int, default=8)
    args.add_argument('--max_length', type=int, default=128)
    args.add_argument('--data_dir', type=str, default=None)
    args.add_argument('--model_dir', type=str, default=None)
    args.add_argument('--output_dir', type=str, default=None)
    args.add_argument('--num_epochs', type=int, default=10)
    args.add_argument('--save_steps', type=int, default=1000)
    args.add_argument('--mixed_precision', type=str, default="bf16")
    args.add_argument("--log",type=str,default="wandb")
    args.add_argument("--mask_id",type=int,default=128255)
    return args.parse_args()
def main():
    args=create_args()
    model, tokenizer, dataloader,accelerator,optimizer=prepare(args)
    global_step=0
    mask_id =args.mask_id
    progress_bar=tqdm(range(args.num_epochs*len(dataloader)),disable=not accelerator.is_local_main_process)
    for i in range(args.num_epochs):
        for batch in dataloader:
            # input_ids = tokenizer(batch['text']).to(accelerator.device)
            input_ids = batch['data'] # [prompt + answer + padding], length=2048
            prompt_length = batch['input_length']  # prompt length
            max_length = args.max_length
            input_ids = input_ids[:, :max_length]

            # if torch.rand(1) < 0.01:
            #     random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
            #     input_ids = input_ids[:, :random_length]
            with accelerator.accumulate(model):
                loss = compute_loss(input_ids, model,prompt_length,mask_id)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step+=1
                if accelerator.is_main_process:
                    if global_step%args.save_steps==0:
                        output_dir=os.path.join(args.output_dir,f"checkpoint_{global_step}")
                        os.makedirs(output_dir,exist_ok=True)
            if global_step%args.save_steps==0:
                output_dir=os.path.join(args.output_dir,f"checkpoint_{global_step}")
                accelerator.save_state(output_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(os.path.join(args.output_dir, "model"))

    print('training finished')
    accelerator.end_training()
if __name__=="__main__":
    main()

