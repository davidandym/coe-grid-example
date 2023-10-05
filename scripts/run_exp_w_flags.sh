#$ -cwd
#$ -q gpu.q -l gpu=1
#$ -l h_rt=1:00:00,mem_free=64G
#$ -j y
#$ -N fashion-mnist

# The path to the conda environment bin
PATH_TO_ENV=~/.conda/envs/grid-example/bin

# Loading the latest cuda modules
module load cuda11.7/toolkit
module load cudnn/8.4.0.27_cuda11.x
module load nccl/2.13.4-1_cuda11.7

${PATH_TO_ENV}/python src/train.py --output_dir experiment_outputs/single_job_run