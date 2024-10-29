#!/bin/bash
#SBATCH --job-name=Test_MOFV2
#SBATCH --output=TestHopeCon.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20

# Load any necessary modules (e.g., Anaconda, CUDA)

module load /opt/MODULES/python/anaconda/3/22.05
module load /opt/MODULES/compiler/cuda/12.1

# Run your Torch script
# python test.py --data_path=./data/Pymatgen_Wrapped/0/ --model_path=./checkpoints/ConvAtt/ConvAtt_epoch_16.pth  --device=0 --test_name=TestConvAtt  --batch_size=32
python test_con.py --data_path=./data/Pymatgen_Wrapped/0/ --model_path=checkpoints/HopeV1_con/HopeV1_con_epoch_90.pth --device=0 --test_name=TestHopeCon --batch_size=1024


