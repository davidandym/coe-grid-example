import os
import subprocess
import shutil


# point to the conda bin for our environment.
conda_bin = '~/.conda/envs/grid-example/bin'

# Define how many jobs can run concurrently in this array
max_concurrent = 3

# create a temporary directory to dump the scripts and job stdout and stderr.
tmp_script_dir = 'tmp_outputs/array_batch_jobs'
if os.path.exists(tmp_script_dir):
    shutil.rmtree(tmp_script_dir)
os.makedirs(tmp_script_dir)

# define initial uid. We want one uid per job to run, and it should increment by 1.
uid = 1

# loop over the hyperparameters of interest.
for rs in range(5):

    # use the hyperparameters to determine the experiment directory (where models and training logs will be saved).
    output_dir = os.path.join(
        './experiment_outputs/array_batch_jobs',
        f'random_seed_{rs}'
    )

    # create a script file, which is the .sh file that will be submitted to qsub.
    # Note that rather than defining the script with the hyperparameters, we are using the UID only.
    script_file_path = os.path.join(tmp_script_dir, f'{uid}.sh')
    script_file = open(script_file_path, 'w')

    # write the .sh file contents
    sh_file = f'''\

{conda_bin}/python src/train.py \
    --output_dir {output_dir} \
    --random_seed {rs}
'''
    
    # write the script to the file
    script_file.write(sh_file)
    script_file.close()

    # Note that we aren't running a qsub command for each experiment, unlike in the batch_submit.
    # Instead, we will run a single qsub command at the end.
    uid += 1

# Array jobs work by passing a variable, named $SGE_TASK_ID, to the environment when running the job.
# So here I'm going to create a generic bash script that defines my qsub flags, loads my modules, and then
# calls the script associated with $SGE_TASK_ID.
generic_script_file_path = os.path.join(tmp_script_dir, f'run_array_script.sh')
gen_script_file = open(generic_script_file_path, 'w')
sh_file = f'''\
#$ -cwd
#$ -q gpu.q -l gpu=1
#$ -l h_rt=1:00:00,mem_free=64G
#$ -j y
#$ -N fashion-mnist
#$ -o {tmp_script_dir}

module load cuda11.7/toolkit
module load cudnn/8.4.0.27_cuda11.x
module load nccl/2.13.4-1_cuda11.7

sh {tmp_script_dir}/${{SGE_TASK_ID}}.sh
'''
    
gen_script_file.write(sh_file)
gen_script_file.close()

# here we define the qsub array command.
# -t 1-N indicates that this is an array job with IDs 1 through N.
# -tc M tells qsub that this array job can run M jobs concurrently.
# Finally, we pass the generic run_array_script.sh to qsub as the starting point of our jobs.
submit_string = "qsub -t 1-{} -tc {} {}"
command = submit_string.format(
    uid,
    max_concurrent,
    generic_script_file_path,
)

# run the qsub command once, submitting all jobs as a single array job.
process = subprocess.Popen(command, shell=True, stderr=None)
process.communicate()
