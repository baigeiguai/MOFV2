#!/bin/bash
#SBATCH --job-name=Test_MOFV2
#SBATCH --output=TestAtLV2.log
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64

# Load any necessary modules (e.g., Anaconda, CUDA)

module load /opt/MODULES/python/anaconda/3/22.05
module load /opt/MODULES/compiler/cuda/12.1

# Run your Torch script
# python test.py --data_path=./data/Pymatgen_Wrapped/0/ --model_path=./checkpoints/ConvAtt/ConvAtt_epoch_16.pth  --device=0 --test_name=TestConvAtt  --batch_size=32
python test.py --data_path=./data/Pymatgen_Wrapped/2/ --model_path=./checkpoints/AtLV2_16L/AtLV2_16layer_epoch_32.pth  --device=0 --test_name=AtLV2_16Layer --batch_size=256


