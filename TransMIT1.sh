export CUDA_VISIBLE_DEVICES=0

fix_seed=${1:-2025}  # default = 2025 if no argument provided

python -u run.py \
  --fix_seed $fix_seed \
  --data_name 'FTD' \
  --miss_rate 0.1 \
  --batch_size 32\
  --lr 0.0005 \
  --epochs 200 \
  --alpha 0.5 \
  --s 6 \
  --d_model 128 \
  --d_q 64 \
  --num_layers 2 \
  --num_heads 2

# python -u run.py \
#   --fix_seed $fix_seed \
#   --data_name 'FTD' \
#   --miss_rate 0.2 \
#   --batch_size 100\
#   --lr 0.0005 \
#   --epochs 200 \
#   --alpha 0.5 \
#   --s 16 \
#   --d_model 128 \
#   --d_q 64 \
#   --num_layers 4 \
#   --num_heads 4

#   python -u run.py \
#   --fix_seed $fix_seed \
#   --data_name 'FTD' \
#   --miss_rate 0.3 \
#   --batch_size 100\
#   --lr 0.0005 \
#   --epochs 200 \
#   --alpha 0.5 \
#   --s 16 \
#   --d_model 128 \
#   --d_q 64 \
#   --num_layers 4 \
#   --num_heads 4

# python -u run.py \
#   --fix_seed $fix_seed \
#   --data_name 'FTD' \
#   --miss_rate 0.4 \
#   --batch_size 100\
#   --lr 0.0005 \
#   --epochs 200 \
#   --alpha 0.5 \
#   --s 16 \
#   --d_model 128 \
#   --d_q 64 \
#   --num_layers 4 \
#   --num_heads 4

# python -u run.py \
#   --fix_seed $fix_seed \
#   --data_name 'FTD' \
#   --miss_rate 0.5 \
#   --batch_size 100\
#   --lr 0.0005 \
#   --epochs 200 \
#   --alpha 0.5 \
#   --s 16 \
#   --d_model 128 \
#   --d_q 64 \
#   --num_layers 4 \
#   --num_heads 4
