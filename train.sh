CUDA_VISIBLE_DEVICES=0 nohup python train.py --model_type 'ConvNeXt'\
                                             --mae true \
                                             --head regression \
                                             --dataset_root /Data/LLVIP > training.log & 