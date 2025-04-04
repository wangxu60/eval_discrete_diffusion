CUDA_VISIBLE_DEVICES=1,2 accelerate launch  \
  --main_process_port 29537 --num_processes 2  /home/wx/data/eval_discrete_diffusion/SFT/train.py \
  --lr 1e-5 \
  --batch_size 1 \
  --max_length 1024 \
  --model_dir /data/jiachun/ckpts/pretrained/Llama-3.2-1B \
  --output_dir /data/wx/llada_results \
  --num_epochs 10 \
  --save_steps 1000 \
  --log wandb