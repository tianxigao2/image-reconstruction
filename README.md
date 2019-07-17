### Concept and Theory ###

#### data ####

Different dataset for different model:

* Using shepp-logan phantom to generate org pictures -> for *sequence of images reconstruction*

  - Using compression rate = 50 and uncompressed videos, extracting pictures out and stored as TIFF(lossless)

* Use 3D model

  - Using Timofte dataset with size of 3288 independent image for *single image reconstruction*

  - Compress by using resize: downsampling + upsampling, information missing

  - PSNR, SSIM, MAE, MSE: 33.24609813224299, 0.9121916197185661, 0.016739558613043767, 0.0009313933013429778

* Use 2D model

  - Using short movie sequence, frames extracted one by one from pictures

  - Compress by using resize: downsampling + upsampling, information missing

#### (2D) SRCNN single image model ####

* Typical model: Build model as paper stated, using Timofte dataset only.

  - PSNR, SSIM, MAE, MSE: 33.9774122589296, 0.9298915633331887, 0.01597587670676589, 0.0007675648953920149

* All parameters frozen inside model:

  - PSNR, SSIM, MAE, MSE: 34.39533863298893, 0.9314247043105195, 0.015128582491306834, 0.0007203916113815921

* Transfer Learning model: Freeze the first one or two layers with fine-tuned parameters, provided by paper.

  - Freeze only one layer makes better performance.

  - For 9-1-5 model, the second layer is doing non-linear mapping, thus only the first layer frozen helps.

  - For 9-3-5 model or 9-5-5 model, freeze the second layer may also help.	//TODO

  - Test to add another non-linear mapping layer to the 9-3-5 or 9-5-5 model.	//TODO

  - PSNR, SSIM, MAE, MSE: 34.568111435612344, 0.9339550069793681, 0.014882425023144312, 0.0006896958805634528

  - Not significant improvement: hyperparameter? Raw data too good?

* Tuning structure: will add some more non-linear mapping layers help?

  - Not significant improvement, sometimes 3 layers, sometimes 5 or 6 layers gave better result.

#### 3D conv layer model with no padding for sequence of images ####

* Using movie sequence to train

  - Shepp-logan phantom not generalized enough;

  - Dataset for one movie too small: only 300 frames even before test-set separation.

  - Set stop points at the end of each movie when doing 5-frame packages, and then conbine packages for all movies to make a larger dataset.	//TODO

* Typical model: 3 conv3d layers with padding, 2 conv3d layers w/o padding, package size = 5

* Training with typical model + 2 model conv3d layers with padding.

#### 2D model + 3D model: individual reconstruction + movie reconstruction ####

* Process data frames independently for the first model (aka 2d model), store the prediction as interim results;

* Pack interim results as input for the second model (aka 3d model), store the prediction for final results.

  - When processing movie sequence directly, no need to do earlystopping and the performance will keep being improved;

  - After processing interim result, will experience a [sudden decrease](https://github.com/tianxigao2/image-reconstruction/tree/to_merge/results/slurm-6026838.out): [Earlystopping? Checkpoint model save?](https://github.com/tianxigao2/image-reconstruction/tree/to_merge/results/slrum-6028662.out)
  - 	Running on local machine, with the second model take batch size = 2
  -	Running on Bridges, with batch size = 20	=>	is it a problem of hyper-parameter?

* Does not help?

		x.shape:(300, 348, 284) y.shape(300, 348, 284)
		psnr_raw, ssim_raw, mae_raw, mse_raw: 25.366856127420387, 0.9064812078136102, 111.63283933004146, 36.877505632453726

		x_set.shape: (300, 352, 288, 1) y_set.shape: (300, 352, 288, 1)
		psnr_interim, ssim_interim, mae_interim, mse_interim: 24.245902882606238, 0.8765903830874335, 9.984719794591268, 244.69206939697267

		AMOUNT, TOTAL_AMOUNT, HEIGHT, WIDTH, TARGET_HEIGHT, TARGET_WIDTH, DEPTH:  296 300 288 352 352 288 5
		x_set.shape:  (300, 352, 288, 1) y_set.shape:  (296, 348, 284, 1) x_interim.shape:  (296, 5, 352, 288, 1)
		mse_final, ssim_final, psnr_final:  2603.812209670608 0.20025187650242368 13.975267255628431

  - Loading fine-tuned parameters from previous trained 3d model

  - Problem of reloading interim prediction?	//todo

  - Missing backpropogation if split the model by half: using functional model to conbine the two structure into one model.	->code under folder 'functional'

### Structure in BitBucket ###

**code** stores all the python code files

* *extractor.py* can run to extract frames out from video; only need to run once at the very beginning;

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

**test_3d**:

* Directly load interim prediction after SRCNN pre-processing

  - check any possible errors in reloading function!	//TODO

**functional**:

* Using functional model to combine the two network inside one model

  - Avoid data reload, add back propogation between the two networks

**data** stores all the data pictures;

Under 'shepp logan phantom':

* Used for testing 3d conv CNN model;

* *compressdSample_xx.avi* and *uncompressedSample.avi* are developed using matlab code;

* *Frame_cmp* stores the images extracted from compressed video (rate = 50);

* *Frame_org* stores the images extracted from uncompressed video;

* two file directories are used for the input of train_x and train_y separately in fit function.

Under 'Timofte_dataset':

* Currently only use Y channel;

* Load the CSV file as ndarray;

* Used for single image reconstruction.

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

* 3d Model




