import torch

from generate import generate,new_generate,ver_generate
from transformers import AutoTokenizer, AutoModel
import random
import numpy as np
from accelerate import load_checkpoint_and_dispatch
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def chat():
    device = 'cuda:5'
    model = AutoModel.from_pretrained('/home/wx/data/model/model/LLaDA-8B-Instruct', torch_dtype=torch.bfloat16,trust_remote_code=True).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('/home/wx/data/model/model/LLaDA-8B-Instruct',trust_remote_code=True)
#     model = load_checkpoint_and_dispatch(
#     model, checkpoint='/home/wx/data/model/model/LLaDA-8B-Instruct', device_map="balanced", no_split_module_classes=['model.transformer.blocks.7.ff_proj,',"model.transformer.blocks.7.up_proj","model.transformer.blocks.7",'model.transformer.blocks.7.dropout', 'model.transformer.blocks.7.act', 'model.transformer.blocks.7.attn_out', 'model.transformer.blocks.7.ff_out', 'model.transformer.blocks.7.rotary_emb', 'model.transformer.blocks.7.attn_norm', 'model.transformer.blocks.7.ff_norm', 'model.transformer.blocks.7.q_proj', 'model.transformer.blocks.7.k_proj', 'model.transformer.blocks.7.v_proj']
# )
    # print(model.hf_device_map)

    gen_length = 1024
    steps = 1024
    print('*' * 66)
    print(f'**  Answer Length: {gen_length}  |  Sampling Steps: {steps}  **')
    print('*' * 66)

    conversation_num = 0
    while True:
        user_input = input("Enter your question: ")

        m = [{"role": "user", "content": user_input}]
        user_input = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(user_input)['input_ids']
        input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

        if conversation_num == 0:
            prompt = input_ids
        else:
            prompt = torch.cat([prompt, input_ids[:, 1:]], dim=1)

        out,_ = new_generate(model, prompt, steps=steps, gen_length=gen_length, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence',acc_num=5)
        # print(out)
        answer = tokenizer.batch_decode(out[:, prompt.shape[1]:], skip_special_tokens=False)[0]
        print(f"Bot's reply: {answer}")

        # remove the <EOS>
        prompt = out[out != 126081].unsqueeze(0)
        conversation_num += 1
        print('-----------------------------------------------------------------------')


if __name__ == "__main__":
    set_seed(1234)
    chat()

