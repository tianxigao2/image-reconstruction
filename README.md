### Tutorial of Sources ###

* log into virtual env for tensorflow: *__conda activate tf-gpu__*

* Using Bridges:
  
  * ```
    ssh -l *account* login.xsede.prg
    ```

  * ```
    ssh -l *account* login.xsede.prg
    ```

  * ```
    gsissh bridges
    ```

  * ```
    interact
    ```

  * (if going to use gpu, use ```interact -gpu -egress ```, but with smaller memory)

  * ```
    module load anaconda3/2019.03
    ```

  * (if haven't build a new env yet, build by 
    ```
    conda create -y -n envName
    ```
    ;

  * ```
    source activate
    ```
    (now should get into (base) env)

  * ```
    conda activate *envName*
    ```

  * can then install packages using pip or conda

* Using scp:

  ```
  scp separator.py janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/image-reconstruction-2019/
  ```

  ```
  scp -r /local/directory janegao@bridges.psc.edu:pylon5/ac5610p/janegao/image-reconstruction-2019/
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

		# USE "sbatch batch_scripts" to submit job

  * call a job to run by typing 
  ```
  sbatch batch_script
  ```

  * check status: 
  ```
  sacct -X -j nnnnnnnn
  ```
  (nnnn stands for the proj-id)

  * result will be automatically generated to the same directory with batch script

  * be sure to store code under pylon5 directory

### Concept and Theory ###

**data**:

* Using shepp-logan phantom to generate org pictures -> can know the compression rate (and thus the performance of NN) exactly

* Using compression rate = 50 and uncompressed videos, extracting pictures out and stored as TIFF(lossless)

* To run on local machine, use the code "getPath.py" here; to run on Bridges, use the code stored in there

* Performance evaluated by PSNR and SSIM

**SRCNN single image model**

**3D conv layer model with no padding for sequence of images**

**Transfer Learning**

### Steps ###

* Develop model following the experiment in SRCNN

  * 3 conv layers

  * input and output are simgle images

* Run on local machine

  * run on GPU with AMOUNT <= 100; BATCH <= 15; KERNAL_SIZE = (64, 32, 1)

  * run on CPU with AMOUNT <= 1000; BATCH <=30; KERNAL_SIZE = (64, 32, 1)

* See performance

  * Determine if the problem is overfitting or not by plotting learning curve

### Structure in BitBucket ###

**code** stores all the python code files

* *extractor.py* can run to extract frames out from video; only need to run once at the very beginning;

* to run the nerual network, run *main.py*;

  * *load_data.py* load images with YUV model's Y channel to the main_function;

  * *separator.py* separates training set and test set, changes them into ndarry;

  * *model_build_up.py* determines the structure of nerual network and compiles it;

  * *train.py* specifys the callback functions and performance detecting metrics;

  * *acc.py* self-defines the callback functions and performance evaluators for *train.py*.

**data** stores all the data pictures;

* *compressdSample_xx.avi* and *uncompressedSample.avi* are developed using matlab code;

* *Frame_cmp* stores the images extracted from compressed video (rate = 50);

* *Frame_org* stores the images extracted from uncompressed video;

* two file directories are used for the input of train_x and train_y separately in fit function.


**datasetMake** store matlab code for shepp-logan phantom and video generation;

* *phantom.m* is a function to generate shepp-logan phantom;

* *seqGenerator.m* is a function to generate sequence of similar matrix E to pass to *phantom.m*.

* *make_dataset_org.m* generates sequence of images and stores as video.

### Reference ###

* [Overview of Transfer Learning](https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/)(smth for general adv of using transfer learning)

* [CS231n transfer learning](http://cs231n.github.io/transfer-learning/)

* Handbook of Research on Machine Learning Application ...[(Chapter 11 Transfer Learning)](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/doc/transferLearning.md)

* Tensorflow Transfer Learning [page](https://www.tensorflow.org/tutorials/images/transfer_learning)

* Tensorflow tutorials -> regression page and basic classification page

* SRCNN


