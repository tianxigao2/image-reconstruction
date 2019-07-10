### Tutorial of Sources ###

<<<<<<< HEAD
* log into virtual env for tensorflow: ```conda activate tf-gpu```
=======
* log into virtual env for tensorflow: *__conda activate tf-gpu__*
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c

* Using Bridges:
  
  * ``` ssh -l *account* login.xsede.org ```

<<<<<<< HEAD
=======
  * ``` ssh -l *account* login.xsede.org ```

>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c
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
<<<<<<< HEAD
  scp -r /local/directory janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/image-reconstruction-2019/
  ```

  ```
  scp janegao@bridges.psc.edu:/pylon5/ac5610p/janegao/running_output/slurm-5921791.out ../results/500_64_128_5_assign_8_hours_changed_PSNR.out
=======
  scp -r /local/directory janegao@bridges.psc.edu:pylon5/ac5610p/janegao/image-reconstruction-2019/
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c
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

  * call a job to run by typing ``` sbatch batch_script ```

<<<<<<< HEAD
  * check all running (pending) jobs: ```squeue -u janegao```

=======
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c
  * check status: ```sacct -X -j nnnnnnnn ```  (nnnn stands for the proj-id)

  * result will be automatically generated to the same directory with batch script

  * be sure to store code under pylon5 directory

### Concept and Theory ###

**data**:

<<<<<<< HEAD
Using shepp-logan phantom to generate org pictures -> for *sequence of images reconstruction*

* Using compression rate = 50 and uncompressed videos, extracting pictures out and stored as TIFF(lossless)

* Use 3D model

Using Timofte dataset with size of 3288 independent image for *single image reconstruction*

* Use 2D model

Performance evaluated by PSNR and SSIM

**(2D) SRCNN single image model**

* Doing feature extracting in the first two layers

**3D conv layer model with no padding for sequence of images**

* Subnet the paper mentioned: "video SR subnet" and "scene change detection and frame replacement subnet", but we will only use videp ST subnet

* Point: input packages should be concatinated

**Transfer Learning**

* First two layers of SRCNN single image model (with org fine-tuned parameters) + the whole 3D conv layer

[?]: should we train 3D conv layer before combining the two;

[?]: should we use the whole 3d model;

[?]: how to determine (by concept) if the transfer learning shows better performance. 

=======
* Using shepp-logan phantom to generate org pictures -> can know the compression rate (and thus the performance of NN) exactly

* Using compression rate = 50 and uncompressed videos, extracting pictures out and stored as TIFF(lossless)

* To run on local machine, use the code "getPath.py" here; to run on Bridges, use the code stored in there

* Performance evaluated by PSNR and SSIM

**SRCNN single image model**

**3D conv layer model with no padding for sequence of images**

**Transfer Learning**

>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c
### Steps ###

* Develop model following the experiment in SRCNN

  * 3 conv layers

  * input and output are simgle images

<<<<<<< HEAD
* Run on local machine and [Bridges](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/RECORD.md)

* Develop 3D conv model

* Separate 2D functions and 3D functions

* Load fine-tuned parameter from .mat file

* [load the parameters into compilable format](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/doc/transferLearning.md#How%20to%20use%20pre-trained%20data%20saved%20as%20.mat%20file%20and%20load%20as%20a%20model)

* freeze the first two layers and add another 2 layers

* fine tune the learning rate of transfer learning model

* compare preformance of:

  - self-trained SRCNN

  - typical SRCNN with trained parameters

  - basic SRCNN model with added 2 layers

=========================================================================

  - (basic SRCNN model with 3d model)

* choose different file to load base-model's parameters

* append 3d model to the transfer learning model

=======
* Run on local machine

  * run on GPU with AMOUNT <= 100; BATCH <= 15; KERNAL_SIZE = (64, 32, 1)

  * run on CPU with AMOUNT <= 1000; BATCH <=30; KERNAL_SIZE = (64, 32, 1)

* See performance

  * Determine if the problem is overfitting or not by plotting learning curve
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c

### Structure in BitBucket ###

**code** stores all the python code files

* *extractor.py* can run to extract frames out from video; only need to run once at the very beginning;

<<<<<<< HEAD
**code_2D**:

* to run the nerual network, run *main.py*;

  * *getpath.py* calls Timofte dataset

  * *load_data.py* load images with YCbCr models' Y channel to the main_function;

  * *separator.py* separates training set and test set, changes them into ndarry;

  * *model_build_up.py* build SRCNN model;

  * *train.py* specifys the callback functions and performance detecting metrics;

  * *acc.py* self-defines the callback functions and performance evaluators for *train.py*;

**code_3D**:

* to run the neural network, run *main_3d.py*;

  * *getpath.py* calls Shepp-logan phantom dataset

  * *load_data.py* load images with YUV models' Y channel to the main function, X set images packed as packages by 5; TARGET SIZE is smaller than org TARGET SIZE (64 -> 60)

  * *separator.py* separates training set and test set, changes them into ndarry;

  * *model_build_up.py* build 3d model;

  * *train.py* specifys the callback functions and performance detecting metrics;

  * *acc.py* self-defines the callback functions and performance evaluators for *train.py*;

**data** stores all the data pictures;

Under 'shepp logan phantom':

* Used for testing 3d conv CNN model;

=======
* to run the nerual network, run *main.py*;

  * *load_data.py* load images with YUV model's Y channel to the main_function;

  * *separator.py* separates training set and test set, changes them into ndarry;

  * *model_build_up.py* determines the structure of nerual network and compiles it;

  * *train.py* specifys the callback functions and performance detecting metrics;

  * *acc.py* self-defines the callback functions and performance evaluators for *train.py*.

**data** stores all the data pictures;

>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c
* *compressdSample_xx.avi* and *uncompressedSample.avi* are developed using matlab code;

* *Frame_cmp* stores the images extracted from compressed video (rate = 50);

* *Frame_org* stores the images extracted from uncompressed video;

* two file directories are used for the input of train_x and train_y separately in fit function.

<<<<<<< HEAD
Under 'Timofte_dataset':

* Currently only use Y channel;

* Load the CSV file as ndarray;

* Used for single image reconstruction.
=======
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c

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

<<<<<<< HEAD
* 3d Model


=======
>>>>>>> 35e709b2621290a314ef0129e11dec8ea6b0db4c

