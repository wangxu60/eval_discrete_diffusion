input_ids:
<BOS><start_id>user<end_id>\nWhat is the capital of France?<eot_id><start_id>assistant<end_id>\nParis.<EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS><EOS>
<BOS><start_id>user<end_id>\nWhat is the capital of Canada?<eot_id><start_id>assistant<end_id>\nThe capital of Canada is Ottawa, located in Ontario.<EOS>

prompt_lengths:
[17, 17]
input_ids, prompt_lengths = batch["input_ids"], batch["prompt_lengths"]

noisy_batch, _, p_mask = forward_process(input_ids)

# Do not add noise to the prompt
token_positions = torch.arange(noisy_batch.shape[1], device=noisy_batch.device).expand(noisy_batch.size(0), noisy_batch.size(1))
prompt_mask = (temp_tensor < prompt_length.unsqueeze(1))
noisy_batch[prompt_mask] = input_ids[prompt_mask]

# Calculate the answer length (including the padded <EOS> tokens)
prompt_mask = prompt_mask.to(torch.int64)    
answer_lengths = torch.sum((1 - prompt_mask), dim=-1, keepdim=True)
answer_lengths = answer_length.repeat(1, noisy_batch.shape[1])    

masked_indices = (noisy_batch == 126336)

logits = model(input_ids=noisy_batch).logits
    
token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none') / p_mask[masked_indices]
ce_loss = torch.sum(token_loss / answer_lengths[masked_indices]) / input_ids.shape[0]