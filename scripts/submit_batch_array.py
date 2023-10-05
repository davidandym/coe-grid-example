""" Script for training cifar-100 ensembles. """


import os
import argparse as ap
import subprocess
import shutil


exp_dir_temp = "grad_{}_optim_{}_lr_{}_epochs_{}_bs_{}_rs_{}_sha_{}"

p = ap.ArgumentParser()
p.add_argument("--grid_sub", type=str,
               default="experiments/array_grid_sub.sh")
p.add_argument("--train_script_dir", type=str,
               default="experiments/cifar-100/multi-task/post-icml-trying-to-overfit")
p.add_argument("--experiment_dir", type=str,
               default="/exp/dmueller/multi-opt/cifar-100/multi-task/post-icml-trying-to-overfit")

p.add_argument("--num_random_seeds", type=int, default=3)
p.add_argument("--max_concurrent", type=int, default=15)
args = p.parse_args()

git_process = subprocess.Popen("git rev-parse --short=5 HEAD",
                               shell=True,
                               stderr=None,
                               stdout=subprocess.PIPE)

output = git_process.communicate()
git_head = output[0].decode('utf-8').strip()

code_copy_dir = "src/git_{}".format(git_head)
if not os.path.exists(code_copy_dir):
    shutil.copytree("src/main", code_copy_dir)


run_dir = os.path.join(
    args.train_script_dir,
    git_head
)
if not os.path.exists(run_dir):
    os.makedirs(os.path.join(run_dir, 'outputs'))

uid = 0

bs_16_lr = 4e-3

for grad in ['avg']:
    for lr_scale in [1, 5, 20]:
        for optim in ['momentum']:
            for epochs in [75]:
                for bs in [128, 256, 512]:
                    for rs in range(args.num_random_seeds):
                        uid += 1

                        output_dir = os.path.join(
                            args.experiment_dir,
                            exp_dir_temp.format(grad, optim, lr_scale, epochs, bs, rs, git_head)
                        )
                        if os.path.exists(output_dir):
                            shutil.rmtree(output_dir)
                        os.makedirs(output_dir)

                        epoch_steps = 2000 // bs 
                        max_steps = epoch_steps * epochs

                        lr = lr_scale * bs_16_lr

                        script_file = open(os.path.join(run_dir, f'{uid}.sh'), 'w')
                        sh_file = f'\
#! /usr/bin/env bash \n\n\
python src/git_{git_head}/train_multitask.py \\\n\
    --seed {rs} \\\n\
	--dataset "CIFAR-100" \\\n\
    --tasks "all-tasks" \\\n\
    --data_dir "/exp/dmueller/data/cifar-100/cifar-100-python" \\\n\
	--eval_with_model "current" \\\n\
    --output_dir {output_dir} \\\n\
    --do_train \\\n\
    --eval_test \\\n\
    --eval_dev \\\n\
    --eval_train \\\n\
    --gradient {grad} \\\n\
    --optimizer {optim} \\\n\
    --classifier_lr "{lr}" \\\n\
    --enc_lr "{lr}" \\\n\
    --lr_gamma 0.99 \\\n\
    --train_batch_size {bs} \\\n\
    --log_steps 10 \\\n\
    --steps_in_epoch {epoch_steps + 1} \\\n\
    --max_steps {max_steps} \\\n\
    --overwrite_output_dir \n'
                        script_file.write(sh_file)
                        script_file.close()

submit_string = "qsub -N cifar-mt -o {} -t 1-{} -tc {} {} {}"

command = submit_string.format(
    "{}/outputs".format(run_dir),
    uid,
    args.max_concurrent,
    args.grid_sub,
    os.path.join(args.train_script_dir, git_head)
)
process = subprocess.Popen(command, shell=True, stderr=None)
process.communicate()