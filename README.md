# COE Grid Example

A small example for submitting and running experiments on the HLTCOE grid.
This is not really meant to be an extensive or exhaustive guide to grid usage, just a small sample of what I regularly use to run my own experiments.

## Conda Envs

I'd recommend using conda to manage environments on the grid. If you're interested in running the code and examples in this repo, you can just copy the environment from the `.yaml` file using the following command:

```bash
conda env create --file ./environment.yaml
```

The reason I recommend using conda is that you can point to the environment bin to explicitly tell the scripts which python version to use when running your experiment.
This is helpful to avoid weird conda or venv activate issues (I don't understand them, but they happen to me a lot when submitting jobs), and also allows other users to run your code on the grid without needing to recreate your virtual environment (shout-out to [Aleem](https://aleemkhan62.github.io/) for this tip).
The correct python version for the environment should be found somewhere like `~/.conda/envs/grid-example/bin/python`, which is the path I use in all of the examples below.

## Running Jobs on Interactive Sessions

The `qrsh` command can be used to ssh directly into a GPU node with access to one or more GPU.
This is a really nice way to test and debug your code since you can just run your code and see it's output without the hassle of queues and stuff.
You can run:
```bash
sh scripts/interactive_session.sh
```
to ssh into a GPU node with a single GPU allocated. In the file `scripts/interactive_session.sh`, you can change type and number of GPUs with the two variables defined at the top of the file. The following commands should get you into a GPU node, activate the correct environment, and then run a test job (assuming you're in the correct directory).

```bash
sh scripts/interactive_session.sh
conda activate grid-example
python src/train.py --output_dir experiment_outputs/test
```

## Submitting a Job to the Grid

When you're ready to run a full experiment, it's time to submit the job to the scheduler.
Job submissions are done using the command `qsub`; the most basic usage of qsub is something like
```bash
qsub scripts/run_exp.sh
```
where `scripts/run_exp.sh` is an sh file that runs your experiment.
However, there are a number of flags that `qsub` takes, some of which are mandatory for GPU jobs.
#### Qsub Flags

The most important of these is the `-q` flag, which will tell the schedule what kind of queue to put your job into.
Since we're interested in experiments that require GPUs, that queue is `-q gpu.q`.
If want to specify a specific type of GPU we can request to be put on only the queue for those GPUs, e.g. `-q gpu.q@@rtx` puts us on the queue for RTX 2080 GPUs only.
We also need to specify how many GPUs we want for our job, which is done with a `-l` switch after the queue command, in the following way: `-q gpu.q -l gpu=1` tells the scheduler that we only want one gpu for our job.

The queues that I'm aware of, in order of GPU quality, are
| Queue | GPU | GPU Mem |
| ---- | ---- | ---- | 
| @@2080 | GeForce GTX 2080 Ti | 11GB |
| @@rtx  | GeForce Titan RTX |  24GB  | 
| @@v100 | Tesla V100S | 32GB | 
| @@dgx | Tesla V100 (Optimized for Multi-GPU training)| 32GB |
| @@a100 | Tesla A100 (I think you need special permission for these) | 40GB |

There are other, generic job resources that are useful to specify before hand. The ones that I use the most are working memory (RAM) and how much time the job needs. These are set with `-l` switches as well. For example `-l h_rt:1:000:000` tells the scheduler that my job will take no more than one hour, and `-l mem_free=64G` tells the scheduler to give me 64GB of RAM.

We also want our experiment to be run from the directory that we're currently working in; the `-cwd` flag tells qsub to run the experiment from the same directory that you are in when you run the qsub command.
Assuming you're at the top level of this directory, then all of the relative paths that we pass in or use in our experiment will be relative from this directory, rather than your home directory.

The final flag that I use regularly is `-N`, which tells the scheduler what to name my job. This is useful if you're running a bunch of jobs simultaneously, and you want to easily keep track of which job is which.
So, in reality, submitting a single job to the grid looks something like:
```bash
qsub -q gpu.q -l gpu=1 -cwd -N fashionMNIST -l h_rt=1:000:000,mem_free=64G scripts/run_exp.sh
```

This is somewhat annoying and a lot to remember; luckily, we can push these flags into the top of the script we want to call using `#$` to specify to qsub that these are qsub flags, rather than comments.
You can see an example of this in `scripts/run_exp_w_flags.sh`.
If we do this, then we can simply run
```bash
qsub scripts/run_exp_w_flags.sh
```
without worring about setting flags in the CLI.


## Running Multiple Jobs Simultaneously

If we want to run a sweep over a set of hyperparameters, the most obvious way to do this is to run a qsub command for each value of our hyperparameter.
While this could be done manually, I prefer to use python scripts to manage my job creation and submissions.
To see an example of how I do this, see `scripts/submit_batch.py`.

The gist of my workflow here is that I will create a separate `.sh` file for each set of hyperparameters, and then call `qsub` on each of those files.
You could instead create a single `.sh` file which takes arguments, and then pass in the hyperparameters as arguments in the `qsub` command. Either way should work just fine!

To test this out, you can submit 5 jobs, for a sweep over 5 random seeds, by running
```bash
python scripts/submit_batch.py
```

## Array Jobs: A Better Way To Run Multiple Jobs

One issue with running `qsub` for each hyperparameter setting is that it can saturate your GPU queue really fast. If you have a GPU quota of 4 GPUs and you submit a hyperparameter search over 4 different values, then you are essentially locked out of using any GPUs until one of those jobs finish. There isn't really a way to control how many jobs run at the same time, because they're all individual jobs and are managed independently.

In cases like this, it is better to submit an Array Job, which essentially treats a set of jobs as a single submission that can be managed together; you can delete all the jobs with just a single jodIB and, more importantly, you can set an upper limit on how many of these jobs can run concurrently.

Array jobs are specified by the `-t` flag, which is invoked like `qsub -t {i}:{i+n}` where i is just some arbitrary minimum number (I use 0 or 1) and n is the number of jobs. Array Jobs function by passing a special environment variable, `$SGE_TASK_ID`, to each job when it starts running, which is just a number in the range `i:i+n`. The `-tc` flag can be used to set the maximum number of concurrent jobs.

Because you have to handle the `$SGE_TASK_ID` variable, array job submissions require a little more complexity than submitting each job independently.
My way of handling this variable (probably not the best way!) is to create a `.sh` file for each number in the range (e.g. `1.sh`). Then, to the `qsub` command, I pass in a _generic_ `.sh` script which simply does all of the standard qsub flags and module loading and then calls `$SGE_TASK_ID.sh`. Here's an example of a `generic.sh` script:
```bash
#$ -cwd
#$ -q gpu.q -l gpu=1
#$ -l h_rt=1:00:00,mem_free=64G
#$ -j y
#$ -N fashion-mnist
#$ -o tmp_outputs

module load cuda11.7/toolkit
module load cudnn/8.4.0.27_cuda11.x
module load nccl/2.13.4-1_cuda11.7

sh tmp_outputs/${SGE_TASK_ID}.sh
```
Then I would call something like
```bash
qsub -t 1:10 -tc 3 generic.sh
```
to run an array job of 10 jobs, with a max of 3 concurrent jobs, which will run each individual job by calling `tmp_outputs/{1-10}.sh`.
I wrote up a quick sample of how I do this in python. To run an example array job with a max concurrent limit of 3 jobs, run
```bash
python scripts/submit_batch_array.py
```

## Monitoring Experiments

Once the jobs are kicked off, you can monitor them using the command `qstat`. 
This will print out a bunch of info about what jobs are currently running, what is queued up, their status, etc.
Here is an example qstat output
```bash
job-ID     prior   name       user         state submit/start at     queue                          jclass                         slots ja-task-ID
------------------------------------------------------------------------------------------------------------------------------------------------
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:04 gpu.q@r8n03.cm.gemini                                             1 97
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:04 gpu.q@r8n04.cm.gemini                                             1 98
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:19 gpu.q@r7n03.cm.gemini                                             1 99
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:19 gpu.q@r8n04.cm.gemini                                             1 100
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:19 gpu.q@r8n04.cm.gemini                                             1 101
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:19 gpu.q@r8n03.cm.gemini                                             1 102
  10952548 0.00010 celeba-mt  dmueller     r     10/04/2023 21:05:34 gpu.q@r7n03.cm.gemini                                             1 103
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 14:14:01 gpu.q@r8n03.cm.gemini                                             1 83
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 14:33:31 gpu.q@r8n04.cm.gemini                                             1 84
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 15:07:16 gpu.q@r8n03.cm.gemini                                             1 85
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 15:17:31 gpu.q@r8n03.cm.gemini                                             1 86
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 15:23:16 gpu.q@r8n04.cm.gemini                                             1 87
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 15:37:01 gpu.q@r8n03.cm.gemini                                             1 91
  10967880 0.00010 cifar-st   dmueller     r     10/06/2023 16:06:46 gpu.q@r8n04.cm.gemini                                             1 92
  10952548 0.00000 celeba-mt  dmueller     qw    09/29/2023 17:11:05                                                                   1 104-144:1
  10967880 0.00000 cifar-st   dmueller     qw    10/05/2023 18:30:06                                                                   1 93-108:1
  ```
  From this, we can see that I have 2 array jobs running (cifar and celeba), each of which has 7 individual jobs running concurrently. I can see what node they're on, what time they started running, and what state they're in.
  You can also run `qstat -u user` if you want to snoop on other people's jobs.

  For a more whollistic view of what's going on on the grid, use `qinfo` which prints more generic statistics for every user currently running jobs.