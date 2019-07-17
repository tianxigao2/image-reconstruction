### Tutorial of Sources ###

* log into virtual env for tensorflow: ```conda activate tf-gpu```

* Using Bridges:
  
  * ``` ssh -l *account* login.xsede.org ```

  * ``` gsissh bridges ```

  * ``` interact ```

  * (if going to use gpu, use ```interact -gpu -egress ```, but with smaller memory)

  * ``` module load anaconda3/2019.03 ```

  * (if haven't build a new env yet, build by ``` conda create -y -n envName ```;

  * ``` source activate ``` (now should get into (base) env)

  * ``` conda activate *envName* ```

  * can then install packages using pip or conda

* Using scp:

  ```
  scp separator.py janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/image-reconstruction-2019/
  ```

  ```
  scp -r /local/directory janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/image-reconstruction-2019/
  ```

  ```
  scp janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/running_output/slurm-5921791.out ../results/500_64_128_5_assign_8_hours_changed_PSNR.out
  ```

* Run a job on batch:

  * create a new file as batch script([how to write a sample batch script](https://www.psc.edu/bridges/user-guide/sample-batch-scripts));

  * batch script example:

		#!/bin/bash
		#SBATCH -p RM
		#SBATCH -t 2:50:00
		#SBATCH -N 1
		#SBATCH --ntasks-per-node 16

		#echo commands to stdout
		set -x

		#move to working directory
		cd /pylon5/ac5610p/janegao/image-reconstruction-2019/code/

		#run python file
		python main.py

		# USE "sbatch -t 2:50:00 -N 1 batch_scripts" to submit job

  * **-t** and **-N** must be stated before batch script name

  * call a job to run by typing ``` sbatch batch_script ```

  * check all running (pending) jobs: ```squeue -u janegao```

  * check status: ```sacct -X -j nnnnnnnn ```  (nnnn stands for the proj-id)

  * result will be automatically generated to the same directory with batch script

  * be sure to store code under pylon5 directory
