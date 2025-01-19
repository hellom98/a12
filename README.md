(.conda) testspace ➤ python train.py                                 
using device: mps

Model Summary:
Total Parameters: 124,439,808
Trainable Parameters: 124,439,808

Model Architecture:
transformer.wte.weight: torch.Size([50257, 768])
transformer.wpe.weight: torch.Size([1024, 768])
transformer.h.0.ln_1.weight: torch.Size([768])
transformer.h.0.ln_1.bias: torch.Size([768])
transformer.h.0.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.0.attn.c_attn.bias: torch.Size([2304])
transformer.h.0.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.0.attn.c_proj.bias: torch.Size([768])
transformer.h.0.ln_2.weight: torch.Size([768])
transformer.h.0.ln_2.bias: torch.Size([768])
transformer.h.0.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.0.mlp.c_fc.bias: torch.Size([3072])
transformer.h.0.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.0.mlp.c_proj.bias: torch.Size([768])
transformer.h.1.ln_1.weight: torch.Size([768])
transformer.h.1.ln_1.bias: torch.Size([768])
transformer.h.1.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.1.attn.c_attn.bias: torch.Size([2304])
transformer.h.1.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.1.attn.c_proj.bias: torch.Size([768])
transformer.h.1.ln_2.weight: torch.Size([768])
transformer.h.1.ln_2.bias: torch.Size([768])
transformer.h.1.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.1.mlp.c_fc.bias: torch.Size([3072])
transformer.h.1.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.1.mlp.c_proj.bias: torch.Size([768])
transformer.h.2.ln_1.weight: torch.Size([768])
transformer.h.2.ln_1.bias: torch.Size([768])
transformer.h.2.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.2.attn.c_attn.bias: torch.Size([2304])
transformer.h.2.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.2.attn.c_proj.bias: torch.Size([768])
transformer.h.2.ln_2.weight: torch.Size([768])
transformer.h.2.ln_2.bias: torch.Size([768])
transformer.h.2.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.2.mlp.c_fc.bias: torch.Size([3072])
transformer.h.2.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.2.mlp.c_proj.bias: torch.Size([768])
transformer.h.3.ln_1.weight: torch.Size([768])
transformer.h.3.ln_1.bias: torch.Size([768])
transformer.h.3.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.3.attn.c_attn.bias: torch.Size([2304])
transformer.h.3.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.3.attn.c_proj.bias: torch.Size([768])
transformer.h.3.ln_2.weight: torch.Size([768])
transformer.h.3.ln_2.bias: torch.Size([768])
transformer.h.3.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.3.mlp.c_fc.bias: torch.Size([3072])
transformer.h.3.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.3.mlp.c_proj.bias: torch.Size([768])
transformer.h.4.ln_1.weight: torch.Size([768])
transformer.h.4.ln_1.bias: torch.Size([768])
transformer.h.4.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.4.attn.c_attn.bias: torch.Size([2304])
transformer.h.4.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.4.attn.c_proj.bias: torch.Size([768])
transformer.h.4.ln_2.weight: torch.Size([768])
transformer.h.4.ln_2.bias: torch.Size([768])
transformer.h.4.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.4.mlp.c_fc.bias: torch.Size([3072])
transformer.h.4.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.4.mlp.c_proj.bias: torch.Size([768])
transformer.h.5.ln_1.weight: torch.Size([768])
transformer.h.5.ln_1.bias: torch.Size([768])
transformer.h.5.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.5.attn.c_attn.bias: torch.Size([2304])
transformer.h.5.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.5.attn.c_proj.bias: torch.Size([768])
transformer.h.5.ln_2.weight: torch.Size([768])
transformer.h.5.ln_2.bias: torch.Size([768])
transformer.h.5.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.5.mlp.c_fc.bias: torch.Size([3072])
transformer.h.5.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.5.mlp.c_proj.bias: torch.Size([768])
transformer.h.6.ln_1.weight: torch.Size([768])
transformer.h.6.ln_1.bias: torch.Size([768])
transformer.h.6.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.6.attn.c_attn.bias: torch.Size([2304])
transformer.h.6.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.6.attn.c_proj.bias: torch.Size([768])
transformer.h.6.ln_2.weight: torch.Size([768])
transformer.h.6.ln_2.bias: torch.Size([768])
transformer.h.6.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.6.mlp.c_fc.bias: torch.Size([3072])
transformer.h.6.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.6.mlp.c_proj.bias: torch.Size([768])
transformer.h.7.ln_1.weight: torch.Size([768])
transformer.h.7.ln_1.bias: torch.Size([768])
transformer.h.7.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.7.attn.c_attn.bias: torch.Size([2304])
transformer.h.7.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.7.attn.c_proj.bias: torch.Size([768])
transformer.h.7.ln_2.weight: torch.Size([768])
transformer.h.7.ln_2.bias: torch.Size([768])
transformer.h.7.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.7.mlp.c_fc.bias: torch.Size([3072])
transformer.h.7.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.7.mlp.c_proj.bias: torch.Size([768])
transformer.h.8.ln_1.weight: torch.Size([768])
transformer.h.8.ln_1.bias: torch.Size([768])
transformer.h.8.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.8.attn.c_attn.bias: torch.Size([2304])
transformer.h.8.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.8.attn.c_proj.bias: torch.Size([768])
transformer.h.8.ln_2.weight: torch.Size([768])
transformer.h.8.ln_2.bias: torch.Size([768])
transformer.h.8.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.8.mlp.c_fc.bias: torch.Size([3072])
transformer.h.8.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.8.mlp.c_proj.bias: torch.Size([768])
transformer.h.9.ln_1.weight: torch.Size([768])
transformer.h.9.ln_1.bias: torch.Size([768])
transformer.h.9.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.9.attn.c_attn.bias: torch.Size([2304])
transformer.h.9.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.9.attn.c_proj.bias: torch.Size([768])
transformer.h.9.ln_2.weight: torch.Size([768])
transformer.h.9.ln_2.bias: torch.Size([768])
transformer.h.9.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.9.mlp.c_fc.bias: torch.Size([3072])
transformer.h.9.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.9.mlp.c_proj.bias: torch.Size([768])
transformer.h.10.ln_1.weight: torch.Size([768])
transformer.h.10.ln_1.bias: torch.Size([768])
transformer.h.10.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.10.attn.c_attn.bias: torch.Size([2304])
transformer.h.10.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.10.attn.c_proj.bias: torch.Size([768])
transformer.h.10.ln_2.weight: torch.Size([768])
transformer.h.10.ln_2.bias: torch.Size([768])
transformer.h.10.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.10.mlp.c_fc.bias: torch.Size([3072])
transformer.h.10.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.10.mlp.c_proj.bias: torch.Size([768])
transformer.h.11.ln_1.weight: torch.Size([768])
transformer.h.11.ln_1.bias: torch.Size([768])
transformer.h.11.attn.c_attn.weight: torch.Size([2304, 768])
transformer.h.11.attn.c_attn.bias: torch.Size([2304])
transformer.h.11.attn.c_proj.weight: torch.Size([768, 768])
transformer.h.11.attn.c_proj.bias: torch.Size([768])
transformer.h.11.ln_2.weight: torch.Size([768])
transformer.h.11.ln_2.bias: torch.Size([768])
transformer.h.11.mlp.c_fc.weight: torch.Size([3072, 768])
transformer.h.11.mlp.c_fc.bias: torch.Size([3072])
transformer.h.11.mlp.c_proj.weight: torch.Size([768, 3072])
transformer.h.11.mlp.c_proj.bias: torch.Size([768])
transformer.ln_f.weight: torch.Size([768])
transformer.ln_f.bias: torch.Size([768])


loaded 338025 tokens
1 epoch = 2640 batches
Training for 3 epochs, 2640 steps per epoch, 7920 total steps
Initial learning rate: 8.00e-06
Epoch 1/3 | Step 0/2640 | Loss: 11.0314 | LR: 8.00e-06 | Time: 0.56s
Epoch 1/3 | Step 100/2640 | Loss: 7.6470 | LR: 3.73e-05 | Time: 12.81s
Epoch 1/3 | Step 200/2640 | Loss: 5.1919 | LR: 1.07e-04 | Time: 25.02s
Epoch 1/3 | Step 300/2640 | Loss: 6.0116 | LR: 1.74e-04 | Time: 37.22s
Epoch 1/3 | Step 400/2640 | Loss: 4.2084 | LR: 2.00e-04 | Time: 49.42s
Epoch 1/3 | Step 500/2640 | Loss: 6.0660 | LR: 2.00e-04 | Time: 61.70s
Epoch 1/3 | Step 600/2640 | Loss: 5.8017 | LR: 2.00e-04 | Time: 73.96s
Epoch 1/3 | Step 700/2640 | Loss: 4.6991 | LR: 1.99e-04 | Time: 86.17s
Epoch 1/3 | Step 800/2640 | Loss: 4.2620 | LR: 1.99e-04 | Time: 98.36s
Epoch 1/3 | Step 900/2640 | Loss: 5.8733 | LR: 1.98e-04 | Time: 110.53s
Epoch 1/3 | Step 1000/2640 | Loss: 4.6918 | LR: 1.97e-04 | Time: 122.75s
Epoch 1/3 | Step 1100/2640 | Loss: 4.2972 | LR: 1.96e-04 | Time: 134.96s
Epoch 1/3 | Step 1200/2640 | Loss: 6.1015 | LR: 1.94e-04 | Time: 147.18s
Epoch 1/3 | Step 1300/2640 | Loss: 4.2815 | LR: 1.93e-04 | Time: 159.39s
Epoch 1/3 | Step 1400/2640 | Loss: 4.8118 | LR: 1.91e-04 | Time: 171.63s
Epoch 1/3 | Step 1500/2640 | Loss: 4.9332 | LR: 1.90e-04 | Time: 184.03s
Epoch 1/3 | Step 1600/2640 | Loss: 5.0782 | LR: 1.88e-04 | Time: 196.59s
Epoch 1/3 | Step 1700/2640 | Loss: 5.4843 | LR: 1.86e-04 | Time: 208.72s
Epoch 1/3 | Step 1800/2640 | Loss: 6.8583 | LR: 1.83e-04 | Time: 220.94s
Epoch 1/3 | Step 1900/2640 | Loss: 4.0616 | LR: 1.81e-04 | Time: 233.12s
Epoch 1/3 | Step 2000/2640 | Loss: 5.8349 | LR: 1.78e-04 | Time: 245.32s
Epoch 1/3 | Step 2100/2640 | Loss: 5.3540 | LR: 1.76e-04 | Time: 257.58s
Epoch 1/3 | Step 2200/2640 | Loss: 4.4930 | LR: 1.73e-04 | Time: 269.80s
Epoch 1/3 | Step 2300/2640 | Loss: 5.3558 | LR: 1.70e-04 | Time: 282.04s
Epoch 1/3 | Step 2400/2640 | Loss: 4.6767 | LR: 1.67e-04 | Time: 294.29s
Epoch 1/3 | Step 2500/2640 | Loss: 4.1970 | LR: 1.64e-04 | Time: 306.53s
Epoch 1/3 | Step 2600/2640 | Loss: 4.6798 | LR: 1.61e-04 | Time: 318.75s

Epoch 1 average loss: 5.1911
Model saved to model_checkpoints/gpt_model.pt
Epoch 2/3 | Step 0/2640 | Loss: 4.7526 | LR: 1.59e-04 | Time: 324.61s
Epoch 2/3 | Step 100/2640 | Loss: 4.8661 | LR: 1.56e-04 | Time: 336.86s
Epoch 2/3 | Step 200/2640 | Loss: 3.6523 | LR: 1.52e-04 | Time: 349.37s
Epoch 2/3 | Step 300/2640 | Loss: 5.0215 | LR: 1.49e-04 | Time: 361.65s
Epoch 2/3 | Step 400/2640 | Loss: 3.1650 | LR: 1.45e-04 | Time: 373.90s
Epoch 2/3 | Step 500/2640 | Loss: 4.7052 | LR: 1.41e-04 | Time: 386.14s
Epoch 2/3 | Step 600/2640 | Loss: 4.7030 | LR: 1.37e-04 | Time: 398.76s
Epoch 2/3 | Step 700/2640 | Loss: 4.0044 | LR: 1.33e-04 | Time: 411.12s
Epoch 2/3 | Step 800/2640 | Loss: 3.5728 | LR: 1.29e-04 | Time: 423.67s
Epoch 2/3 | Step 900/2640 | Loss: 4.9750 | LR: 1.25e-04 | Time: 436.14s
Epoch 2/3 | Step 1000/2640 | Loss: 3.8117 | LR: 1.21e-04 | Time: 448.52s
Epoch 2/3 | Step 1100/2640 | Loss: 3.6213 | LR: 1.17e-04 | Time: 460.90s
Epoch 2/3 | Step 1200/2640 | Loss: 5.3448 | LR: 1.13e-04 | Time: 473.58s
Epoch 2/3 | Step 1300/2640 | Loss: 3.8046 | LR: 1.09e-04 | Time: 485.89s
Epoch 2/3 | Step 1400/2640 | Loss: 4.0316 | LR: 1.05e-04 | Time: 498.29s
