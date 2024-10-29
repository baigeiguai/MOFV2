#!/bin/bash
#SBATCH --job-name=MOFV2
#SBATCH --output=Hope_Con_AG.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=25

# Load any necessary modules (e.g., Anaconda, CUDA)

#module load /opt/MODULES/python/anaconda/3/22.05
#module load /opt/MODULES/compiler/cuda/12.1

module load /opt/MODULES/python/anaconda/3/22.05
module load /opt/MODULES/compiler/cuda/12.1

# Run your Torch script
# python train.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=RawEmbedConvNoTrain_extend  --learning_rate=5e-3 --min_learning_rate=1e-6 --start_scheduler_step=0 --batch_size=2048 --model_save_path=./checkpoints/RawEmbedConvNoTrain_extend  --device=0  --epoch_num=100  --weight_decay=2e-6 --model_path=./checkpoints/RawEmbedConvNoTrain/RawEmbedConvNoTrain_epoch_7.pth --num_workers=10
#python train.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=ConcatEmbedConv_extend --learning_rate=1e-2 --min_learning_rate=1e-6 --start_scheduler_step=0 --batch_size=2048 --model_save_path=./checkpoints/ConcatEmbedConv_extend --device=0  --epoch_num=100  --weight_decay=1e-6 --num_workers=30 --model_path=./checkpoints/ConcatEmbedConv/ConcatEmbedConv_epoch_11.pth
# python train.py --data_path=./data/Pymatgen_Wrapped_Short/0  --train_name=AtBase --learning_rate=5e-3 --min_learning_rate=1e-6 --start_scheduler_step=0 --batch_size=512 --model_save_path=./checkpoints/AtBase --device=0,1,2,3  --epoch_num=100  --weight_decay=1e-6 --num_workers=30 
# python train.py --data_path=./data/Pymatgen_Wrapped_Short/0  --train_name=AtLBase_extend --learning_rate=1e-2 --min_learning_rate=1e-3 --start_scheduler_step=0 --batch_size=1024 --model_save_path=./checkpoints/AtLBase_extend --device=0,1,2,3  --epoch_num=100  --weight_decay=0 --num_workers=30 --model_path=./checkpoints/AtLBase/AtLBase_epoch_5.pth
# python train.py --data_path=./data/Pymatgen_Wrapped_Plus/0/  --train_name=AtLBase --learning_rate=5e-3 --min_learning_rate=1e-6 --start_scheduler_step=0 --batch_size=1024  --model_save_path=./checkpoints/AtLBase  --device=0,1,2,3   --epoch_num=200  --weight_decay=1e-6 --num_workers=40
#python train.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=ConvAtt --learning_rate=1e-4 --min_learning_rate=1e-5 --start_scheduler_step=10 --batch_size=128 --model_save_path=./checkpoints/ConvAtt --device=0,1,2,3  --epoch_num=200 --weight_decay=1e-6  --num_workers=30 --model_path=../temp_files/ConvAtt_extend_epoch_22.pth
#python train.py --data_path=./data/Pymatgen_Wrapped/2  --train_name=AtLV2 --learning_rate=5e-4 --min_learning_rate=1e-5 --start_scheduler_step=10 --batch_size=64  --model_save_path=./checkpoints/AtLV2  --device=0,1,2,3  --epoch_num=200 --weight_decay=1e-6  --num_workers=30
#python train.py --data_path=./data/Pymatgen_Wrapped/2  --train_name=AtLV2_16layer --learning_rate=5e-4 --min_learning_rate=1e-5 --start_scheduler_step=10 --batch_size=2048  --model_save_path=./checkpoints/AtLV2_16L  --device=0  --epoch_num=200 --weight_decay=1e-6  --num_workers=30
#python train.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=HopeV1 --learning_rate=5e-4 --min_learning_rate=1e-6 --start_scheduler_step=10 --batch_size=512 --model_save_path=./checkpoints/HopeV1   --device=0,1  --epoch_num=200 --weight_decay=1e-6  --num_workers=30
#/opt/Anaconda3/2022.05/bin/python3 train_con.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=HopeV1_SupConLoss  --learning_rate=5e-4 --min_learning_rate=1e-6 --start_scheduler_step=10 --batch_size=1024  --model_save_path=./checkpoints/HopeV1_SupConLoss  --device=0  --epoch_num=120  --weight_decay=1e-5 --num_workers=40
#pip list 
#python train_sub.py --data_path=./data/All_Wrapped --train_name=Hope_SubClass --model_save_path=./checkpoints/Hope_SubClass  --batch_size=768  --device=0 --learning_rate=5e-4 --min_learning_rate=1e-5 
python train_con.py --data_path=./data/Pymatgen_Wrapped/0  --train_name=HopeV1_Con_AG --learning_rate=5e-4 --min_learning_rate=1e-6  --start_scheduler_step=30 --batch_size=768  --model_save_path=./checkpoints/HopeV1_Con_AG --device=0  --epoch_num=200 --weight_decay=1e-4


