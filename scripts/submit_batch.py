import os
import argparse as ap
import subprocess
import shutil


# point to the conda bin for our environment.
conda_bin = '~/.conda/envs/grid-example/bin'

# create a temporary directory to dump the scripts and job stdout and stderr.
tmp_script_dir = './tmp_outputs/batch_jobs'
if os.path.exists(tmp_script_dir):
    shutil.rmtree(tmp_script_dir)
os.makedirs(tmp_script_dir)

# loop over the hyperparameters of interest.
for rs in range(5):

    # use the hyperparameters to determine the experiment directory (where models and training logs will be saved).
    output_dir = os.path.join(
        './experiment_outputs/batch_jobs',
        f'random_seed_{rs}'
    )

    # create a script file, which is the .sh file that will be submitted to qsub
    script_file_path = os.path.join(tmp_script_dir, f'random_seed_{rs}.sh')
    script_file = open(script_file_path, 'w')

    # write the .sh file contents
    sh_file = f'''\
#$ -cwd
#$ -q gpu.q -l gpu=1
#$ -l h_rt=1:00:00,mem_free=64G
#$ -j y
#$ -N fashion-mnist

module load cuda11.7/toolkit
module load cudnn/8.4.0.27_cuda11.x
module load nccl/2.13.4-1_cuda11.7

{conda_bin}/python src/train.py \
    --output_dir {output_dir} \
    --random_seed {rs}
'''
    
    # write the script to the file
    script_file.write(sh_file)
    script_file.close()

    # define the qsub command (-o flag sets where the stdout will go)
    submit_string = "qsub -o {} {}"
    command = submit_string.format(
        tmp_script_dir,   # where to output stdout
        script_file_path  # where the script to run is
    )

    # run the qsub command for each experiment in our loop.
    process = subprocess.Popen(command, shell=True, stderr=None)
    process.communicate()