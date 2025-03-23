import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from transformers import AutoModel, AutoTokenizer,AutoModelForCausalLM,LlamaConfig
import torch.nn.functional as F
import argparse
from torch.optim import AdamW
from accelerate.utils import DistributedType, set_seed
from tqdm import tqdm
import os
import shutil
from gsm8k_data import preprocess_gsm8k
from generate import generate
from models.llama import LlamaModel,LlamaForCausalLM
os.environ["WANDB_MODE"] = "offline"
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
def remove_oldest_folder(directory,checkpoint_num):
    # 获取所有文件夹的路径
    folders = [os.path.join(directory, f) for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    if len(folders)<checkpoint_num:
        return
    
    # 按创建时间排序（Windows 使用 `os.path.getctime`，Linux/macOS 使用 `os.path.getmtime`）
    oldest_folder = min(folders, key=os.path.getctime)

    # 删除最早的文件夹
    shutil.rmtree(oldest_folder)
    print(f"已删除最早的文件夹: {oldest_folder}")
    
def forward_process(input_ids, eps=1e-3,mask_id=126336):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, mask_id, input_ids)
    return noisy_batch, masked_indices, p_mask
def compute_loss(input_ids, model,prompt_length,mask_id=128255,attention_mask=None):
    noisy_batch, masked_indices, p_mask = forward_process(input_ids,mask_id=mask_id)
    # temp_tensor = torch.arange(noisy_batch.size(1), device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    # print((noisy_batch==mask_id).sum())
    token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
    prompt_mask = (token_positions < prompt_length.unsqueeze(1))
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    # print((noisy_batch==mask_id).sum())
# Calculate the answer length (including the padded <EOS> tokens)
    prompt_mask = prompt_mask.to(torch.int64)    
    answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
    answer_lengths = answer_lengths.repeat(1, noisy_batch.shape[1])    

    masked_indices = (noisy_batch == mask_id)
    noisy_batch=noisy_batch.to(model.device)
    logits = model(input_ids=noisy_batch,attention_mask=attention_mask,use_cache=False).logits
    input_ids=input_ids.to(model.device)
    p_mask=p_mask.to(model.device)
    # print((input_ids[masked_indices]==mask_id).sum())
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
    os.makedirs(args.output_dir,exist_ok=True)
    if args.mode==1:
        model = LlamaForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,trust_remote_code=True).train()
    if args.mode==0:
        model= AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16,trust_remote_code=True).train()
    # model.config._attn_implementation="eager"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    dataset= preprocess_gsm8k(tokenizer,args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)
    for param in model.parameters():
        param.requires_grad = True
        param.to(torch.float32)
    train_parameters=[param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(train_parameters, lr=args.lr,)
    config = {
    "num_iterations":args.num_epochs,
    "learning_rate": args.lr,
    "max_length":args.max_length
}

    accelerator.init_trackers(args.project_name, config=config)
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (args.batch_size)
    model,optimizer=accelerator.prepare(model,optimizer)
    if args.resume_path is not None:
        accelerator.load_state(args.resume_path)
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
    args.add_argument("--resume_path",type=str,default=None)
    args.add_argument("--project_name",type=str,default="llada")
    args.add_argument("--checkpoint_num",type=int,default=10)
    args.add_argument("--mode",type=int,default=0)
    return args.parse_args()
def eval(model,question,output_dir,tokenizer,mask_id,attention_mask=None):
    with torch.no_grad():
        out= generate(model, question, steps=1024, gen_length=1024, block_length=1024, temperature=0., cfg_scale=0., remasking='low_confidence',mask_id=mask_id,attention_mask=attention_mask)
        answer = tokenizer.batch_decode(out[:, question.shape[1]:], skip_special_tokens=False)[0]
        # print(answer)
        with open(os.path.join(output_dir,"eval.txt"),"w",encoding="utf-8") as f:
            f.write(answer)
        # print("txt complete")

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
                if args.mode==1:
                    attention_mask=torch.zeros((input_ids.shape[0],1,input_ids.shape[1],input_ids.shape[1]),dtype=model.dtype,device=model.device)
                    loss = compute_loss(input_ids, model,prompt_length,mask_id,attention_mask)
                else:
                    attention_mask=None
                    loss = compute_loss(input_ids, model,prompt_length,mask_id)
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step+=1
                accelerator.log({"loss":loss},step=global_step)
                if accelerator.is_main_process:
                    if global_step%args.save_steps==0:

                        output_dir=os.path.join(args.output_dir,f"checkpoint_{global_step}")
                        os.makedirs(output_dir,exist_ok=True)
                        remove_oldest_folder(args.output_dir,checkpoint_num=args.checkpoint_num)
                        question="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
                        prompt=tokenizer(question)["input_ids"]
                        prompt = torch.tensor(prompt).to(model.device).unsqueeze(0)
                        eval(model,prompt,output_dir,tokenizer,mask_id,attention_mask)
            accelerator.wait_for_everyone()
            if global_step%args.save_steps==0:
                output_dir=os.path.join(args.output_dir,f"checkpoint_{global_step}")
                os.makedirs(output_dir,exist_ok=True)
                accelerator.save_state(output_dir)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(os.path.join(args.output_dir, "model"))

    print('training finished')
    accelerator.end_training()
if __name__=="__main__":
    main()

