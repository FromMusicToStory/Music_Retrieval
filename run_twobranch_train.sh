python3 train.py --text_dir data/Story_dataset \
--audio_dir data/MTG \
--checkpoint_dir checkpoint \
--ckp_per_step 10 \
--log_dir result/tensorboard \
--epoch 2000 \
--model TwoBranch \
--batch_size 16
