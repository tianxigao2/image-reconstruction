### Recording for results got from Bridges ###

#### For SRCNN (2D) model ####
- 1000_64_128_5_assign_5_hours.out

  - Try to run 1000 pictures from 64*64 to 128*128, batch size = 5

  - Assign 5 hours, but seems command error, and goes to default; the program only runs to 30 min.

  - Goes to epochs around 20

- 500_64_128_5_assign_8_hours.out

  - Start few mins before 17:00, finished before 20:00

  - Assign 8 hours, can successfully run to the end of job

  - 500 pictures, resolution from 64 to 128, batch size = 5

  - PSNR and SSIM are according to compressed image directly resized from org image

- 500_64_128_5_assign_8_hours_changed_PSNR.out

  - Start at 09:53, end at 12:35	=>	assign 3 hours will be enough

  - PSNR and SSIM are according to compressed image resized from ground truth

  - Similar performance with previous PSNR and SSIM function

- T_set_500_5.out

  - Using Timofte dataset to test SR 2d model

  - Dataset size = 100

  - Assign the job 5 hours

- T_set_3288_assign_8_hr.out

  - Using Timofte dataset, input all the data (3288)

  - Assign 8 hours for the jobslurm-5925987.out

  - Stuck at Epoch = 43

  - Exceed time limit

- running output from local machine:

  - T_set, size = 100, batch = 5

  - Output from hardware comes from computing PSNR and SSIM

		without processed by neural network:
		psnr =  32.32374505996704
		ssim =  0.8904345609247685
		performance after processed by nerual network:
		psnr:  30.371594619750976
		ssim:  0.8996250867843628
		performance of predicted images:
		psnr, ssim for raw data: 33.71692161560058 0.9066464126110076
		psnr, ssim for predicted result: 31.398320960998536 0.9159058421850205
		mae, mse for raw data: 0.016919827661132814 0.0010573680347035646
		mae, mse for predicted result 0.023058389639043077 0.0012274294390675808

- running on gpu-small (Bridges):

  - 100, 5

		psnr =  32.32374605310659
		ssim =  0.9335277163055442
		performance after processed by nerual network:
		psnr:  30.336954879760743
		ssim:  0.8968415021896362
		performance of predicted images:
		psnr, ssim for raw data: 33.71692237367288 0.9429522818613986
		psnr, ssim for predicted result: 31.469573836794893 0.9468661789488848
		mae, mse for raw data: 0.016919827661132814 0.0010573680347035646
		mae, mse for predicted result 0.022500586560944343 0.0012215527042980434


- 5925120

  - Set size = 1500, going to see if the stagnation is because of data size, assigned 6 hours

  - Canceled due to time limits, epoch = 15

- slurm-5925987.out

  - Set size = 1000; assign 6 hrs to the job

  - Cancelled due to time limit

- 5929162

  - Set size = 1000, assign 8 hours to the job

  - Going to see how much time a 1000-pic dataset needs

  - Cancelled due to time limit

- 5931148 (checked, for 2d model for sure)

  - Set size = 500, assign 8 hours

  - to make sure the logic of code is OK

  - error: im_true has intensity values outside the range expected for its data type

- 5933310

  - Running on GPU-small, assign 1 hour

  - size = 3288

  - Error: ValueError: im_true has intensity values outside the range expected for its data type.  Please manually specify the data_range

  - Dynamic_range not specified

- 5933527

  - Running on GPU_small, assign 2 hours

  - size = 3288

  - Dynamic_range specified as 255

  - Can run, but wrong PSNR value

- 5933529

  - Running on GPU_small, assign 2 hours

  - size = 3288

  - Dynamic_range specified as 256

  - Successfully run, not sure if correct or not

- 5933832

  - Running on GPU_small, assign 2 hours

  - size = 3288

  - Dynamic_range specified as 1

  - Successfuly run

#### for 3d model ####

- shepp-logan_100_3D.out

  - 8 hours, 3D model using Shepp-logan Phantom

  - amount = 100, resolution = 32 -> 64, batch = 5, epoch = 1000

  - SSIM performs well, PSNR is negative	

- shepp-logan_100_3D_uint8.out

  - Test why PSNR is negative: change float32 to uint8

  - Problem fixed	

  - amount = 100, resolution = 32 -> 64, batch = 5, epoch = 1000

Error: ValueError: im_true has intensity values outside the range expected for its data type.  Please manually specify the data_range

- print min and max of im_true and im_predict

("true_min, true_max = 0 255; predict_min, predict_max = -0.0 255")

  - remove the last picture (AMOUNT 500 -> 499)	=>5933361 (gpu-small, assign 2 hours)

		set range = 1 or range = 256

  - if min < 0, let min = 0	=> failed

		true_min, true_max =  0 255
		predict_min, predict_max = 0 255
		/home/janegao/.conda/envs/env/lib/python3.7/site-packages/skimage/measure/simple_metrics.py:127: UserWarning: Inputs have mismatched dtype.  Setting data_range based on im_true.
		warn("Inputs have mismatched dtype.  Setting data_range based on "
		/home/janegao/.conda/envs/env/lib/python3.7/site-packages/skimage/measure/_structural_similarity.py:155: UserWarning: Inputs have mismatched dtype.  Setting data_range based on X.dtype.
		warn("Inputs have mismatched dtype.  Setting data_range based on "
		true_min, true_max =  0 255


- dataset = Container

  - [trail0_3d.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/trail0_3d.out), psnr improved from 18 to 20

  - [trail1_3d.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/trail1_3d.out), psnr imporved from 22 to 26

  - [trail2_3d.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/trail2_3d.out), add 2 more layers, psnr from 22 to 28

  - [trail3_3d.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/trail3_3d.out), add 2 more layers, epoch = 500. *overfit heavily*

#### for transfer learning model ####

- ADAM(lr=0.001, decay=1e-6), AMOUNT = 3288, BATCH = 30

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  32.49794277147648
		ssim:  0.9214905683985228
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 32.73852918303381 0.9286519562827201
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.01871146150820401 0.0008543146827991486

- Adam(lr = 0.001, decay = 1e-6), AMOUNT = 3288, BATCH = 30

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  33.32229467761834
		ssim:  0.9202151557791822
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 33.55565694249675 0.9273953696676059
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.016553640714301678 0.0007906101653951263

- ADAM(lr=0.0005, decay=1e-6), AMOUNT = 3288, BATCH = 30

  - Stop at epoch = 44
		
		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  33.82051299062518
		ssim:  0.9233639388483287
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 34.05868559085807 0.9307882727034198
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.015724144987448493 0.0007393778400121262

- ADAM(lr=0.0003, decay=1e-6), AMOUNT = 3288, BATCH = 30

  - Stop at epoch = 66

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  33.51058461004337
		ssim:  0.9211136328856755
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 33.75302700726272 0.9281626818464954
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.016175298532042373 0.0007707692998916953

- ADAM(lr=0.0005, decay=5e-5), AMOUNT = 3288, BATCH = 30

  - stop at epoch = 63

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  33.52701753391513
		ssim:  0.9216833415140218
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 33.75460223710975 0.9287758222775475
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.01604342261427898 0.0007679279522831855

#### for fine-tuned model ####
- with weight transposed

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by 2+2 nerual network:
		psnr:  31.951406993793444
		ssim:  0.9159237690751544
		performance after processed by 3-layer nerual network:
		psnr:  6.622407911394939
		ssim:  0.011128140966312344
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 32.17825415051682 0.9221916967474031
		psnr, ssim for predicted result of fine-tuned network: 6.795688858851343 0.011859018611827535
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.019519571950382137 0.0009214781621304264
		mae, mse for predicted result of fine-tuned network 0.4483490363596167 0.2366072314525427

- without tranpose

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by 2+2 nerual network:
		psnr:  33.00753919797252
		ssim:  0.9199809412992499
		performance after processed by 3-layer nerual network:
		psnr:  6.617755762252518
		ssim:  0.010819070715591765
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 33.25247463881564 0.92716406017897
		psnr, ssim for predicted result of fine-tuned network: 6.787714054864546 0.011629721859646043
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.017299860248577444 0.000812325985232138
		mae, mse for predicted result of fine-tuned network 0.4486307370061262 0.23681689582752222

- fit the fine-tuned model with new data, without transpose

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939 

		performance after processed by 2+2 nerual network:
		psnr:  32.94420984072377
		ssim:  0.9171418770184535 

		performance after processed by 3-layer nerual network:
		psnr:  34.156309721043804
		ssim:  0.922475484523483 

		performance after processed by self-trained SRCNN:
		psnr:  33.71093183452186
		ssim:  0.9203401137214197  

		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 33.18705389486758 0.9246176868762175
		psnr, ssim for predicted result of fine-tuned network: 34.39533863298893 0.9314247043105195
		psnr, ssim for predicted result of self-trained network: 33.9774122589296 0.9298915633331887 

		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.017436507321962554 0.0008346723536093918
		mae, mse for predicted result of fine-tuned network 0.015128582491306834 0.0007203916113815921
		mae, mse for predicted result of self-trained network 0.01597587670676589 0.0007675648953920149 

- load parameter (9-1-5 Imagenet), see if generalization makes a difference in result

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939 

		performance after processed by 2+2 nerual network:
		psnr:  32.233695619822456
		ssim:  0.9071991144024374 

		performance after processed by 3-layer nerual network:
		psnr:  33.85454175535717
		ssim:  0.9199593357713504 

		performance after processed by self-trained SRCNN:
		psnr:  34.402345197554325
		ssim:  0.9236550118986645  

		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 32.46258969054824 0.9141279453711311
		psnr, ssim for predicted result of fine-tuned network: 34.10821711559255 0.9290720760063075
		psnr, ssim for predicted result of self-trained network: 34.64851190298278 0.9332298432584628 

		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.018946358168595332 0.0009523153326665808
		mae, mse for predicted result of fine-tuned network 0.015761210642809217 0.000757946626476562
		mae, mse for predicted result of self-trained network 0.014745022115220156 0.0006984175749830964

  - Performance not better than (9-1-5 91 images), may be caused by similarity using the same data in 91-image example.


#### Experiment ####

##### SRCNN with all layers loaded from fine-tuned data #####

[Feature Extraction]

[Non-linear Mapping]

[Reconstruction]

- Results as shown before

##### 1 + 2 model #####

[Feature Extraction]

Non-linear Mapping

Reconstruction

- 5949519

- 5949539

- gpu: [model_1+2.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/model_1%2B2.out)

##### self-trained 4 layers (added non-linearity) #####

Feature Extraction

Non-linear Mapping

Non-linear Mapping

Reconstruction

- Result: [model_add_non-linearity.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/model_add_non-linearity.out)

##### 1 + 3 model #####

[Feature Extraction]

Non-linear Mapping

Non-linear Mapping

Reconstruction

- Resulut: [model_1+3.out](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/results/model_1%2B3.out)

##### hyper-parameter tuning #####

		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
				
		The list of survivals' performances:  [0.0007595433733259019, 0.033145709863459084, 0.03309217990463809, 0.0006453759830634712, 0.0006698216822941004, 0.0007930509409921012, 0.0007399136943977586, 0.0007027420440000853, 0.0008301682438344714, 0.0008305879766234738, 0.000644087276066226, 0.0032694027037289005, 0.0009244858902259121, 0.0006611345186872029, 0.0007373944210103716, 0.0007947931486524438, 0.0014104050818383466]
		The number of survivals:  17
		The best performance is  0.000644087276066226
		The best hyperparameter set is 
		lr :  0.0029861308377023593
		epochs :  113
		tolerate :  4
		decay :  5e-05
		batch_size :  25
		layers_num :  2

		--result stores in test1.h5


		The number of survivals:  6
		The best performance is  0.0006474627531841284
		The best hyperparameter set is 
		lr :  0.003709810182711635
		epochs :  65
		tolerate :  3
		decay :  3e-05
		batch_size :  20
		layers_num :  4
		 
		finish prediction
		without processed by neural network:
		psnr =  33.01512476511432
		ssim =  0.9094088004024939
		performance after processed by nerual network:
		psnr:  34.62898522438658
		ssim:  0.9277384522749897
		performance of predicted images:
		psnr, ssim for raw data: 33.24609813224299 0.9121916197185661
		psnr, ssim for predicted result: 34.86560016003114 0.9366768077909863
		mae, mse for raw data: 0.016739558613043767 0.0009313933013429778
		mae, mse for predicted result 0.014345458238951557 0.0006474627596842397
		
		--result stores in test2.h5
