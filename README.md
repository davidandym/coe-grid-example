# COE Grid Example

A small example for submitting and running experiments on the HLTCOE grid.
This is not really meant to be an extensive or exhaustive guide to grid usage, just a small sample of what I regularly use to run my own experiments.

## Which Python

I'd recommend using conda to manage environments on the grid, here's the command I used to create this one: `conda create --name grid-example python=3.10` If you're interested in running the code, you can just copy the environment from the `.yaml` file using the following commannd:

```bash
conda env create --file ./environment.yaml
```

The reason I recommend using conda is that you can point to the environment bin to explicitly tell the scripts which python version to use when running your experiment.
The correct python version for our environment should be found somewhere like `~/.conda/envs/grid-example/bin/python`;
this will become relevant in the `qsub` command sections.

## Running Jobs on Interactive Sessions

The `qrsh` command can be used to ssh directly into a GPU node with access to one or more GPU.
This is a really nice way to test and debug your code.

An example script can be found in `scripts/`. You can run
```sh scripts/interactive_session.sh```
to ssh into a GPU node with a single GPU allocated. In the file you can change type and number of GPUs with the two variables defined at the top of hte file. The following commands should get you into a GPU node, activate the correct environment, and then run a test job (assuming you're in the correct directory).

```bash
sh scripts/interactive_session.sh
conda activate grid-example
python src/train.py --output_dir experiment_outputs/test
```

## Submitting a Job to the Grid

While an interactive session is a great option to test and debug jobs, it's likely not ideal to run a full experiment in an interactive session if that experiment will take more than a few minutes to run.
The proper way to run a full experiment is to submit that experiment as a job to the grid's job scheduler.

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

We also want our experiment to be run from the directory that we're currently working in; the `-cwd` flag tells qsub to run the experiment from the same directory that you are in when you run the qsub command.
Assuming you're at the top level of this directory, then all of the relative paths that we pass in or use in our experiment will be relative from this directory, rather than your home directory.

There are generic job resources that are useful to specify before hand. The ones that I use the most are working memory (RAM) and how much time the job needs. These are set with `-l` switches as well. For example `-l h_rt:1:000:000` tells the scheduler that my job will take no more than one hour, and `-l mem_free=64G` tells the scheduler to give me 64GB of RAM.

The final flag that I use regularly is `-N`, which tells the scheduler what to name my job. This is useful if you're running a bunch of jobs simultaneously, and you want to easily keep track of which job is which.


So, in reality, submitting a single job to the grid looks something like:
```bash
qsub -q gpu.q -l gpu=1 -cwd -N fashionMNIST -l h_rt=1:000:000,mem_free=64G scripts/run_exp.sh
```

This is somewhat annoying and a lot to remember; luckily, we can actually push these flags into the top of the script we want to call using `#$` to specify to qsub that these are qsub flags, rather than comments.
You can see an example of this in `scripts/run_exp_w_flags.sh`.
If we do this, then we can simply run
```bash
qsub scripts/run_exp_w_flags.sh
```
without worring about setting flags in the CLI.


## Running Multiple Jobs Simultaneously

Suppose we want to run a job that sweeps over a set of hyperparameters (in our case, the random seed).
The proper way to do this is to submit each run to the queue manager, allowing 

## Array Jobs: A Better Way To Run Multiple Jobs

## Monitoring Experiments