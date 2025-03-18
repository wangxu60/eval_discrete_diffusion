import torch
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
def forward_process(input_ids, eps=1e-3):
    b, l = input_ids.shape
    t = torch.rand(b, device=input_ids.device)
    p_mask = (1 - eps) * t + eps
    p_mask = p_mask[:, None].repeat(1, l)

    masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
    # 126336 is used for [MASK] token
    noisy_batch = torch.where(masked_indices, 126336, input_ids)
    return noisy_batch, masked_indices, p_mask

# The data is an integer tensor of shape (b, 4096), 
# where b represents the batch size and 4096 is the sequence length.
input_ids = batch["input_ids"]

# We set 1% of the pre-training data to a random length that is uniformly sampled from the range [1, 4096].
# The following implementation is not elegant and involves some data waste. 
# However, the data waste is minimal, so we ignore it.
if torch.rand(1) < 0.01:
    random_length = torch.randint(1, input_ids.shape[1] + 1, (1,))
    input_ids = input_ids[:, :random_length]

noisy_batch, masked_indices, p_mask = forward_process(input_ids)
logits = model(input_ids=noisy_batch).logits

token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
def prepare():
    accelerator = Accelerator()
    model = AutoModel.from_pretrained('/home/wx/data/model/model/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16,trust_remote_code=True).to(accelerator.device).train()
    tokenizer = AutoTokenizer.from_pretrained('/home/wx/data/model/model/LLaDA-8B-Instruct',trust_remote_code=True)
    dataset=
    dataLoader = DataLoader(dataset, batch_size=8, shuffle=True)
    return model, tokenizer, dataLoader
def train():
    for batch in dataLoader:
def args():
    
