python -m torch.distributed.launch --nproc_per_node=1 --master_port=44248  train.py --config ./configs/FA_Res18_CCRI.yml
