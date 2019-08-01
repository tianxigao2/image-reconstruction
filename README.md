### Concept and Theory ###

#### command ####

[`command.md`](./command.md): Commands about how to use bridges

#### data ####

Test Cases:

* Using [shepp-logan phantom](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/data/shepp%20logan%20phantom/) to generate org pictures -> for sequence of images reconstruction

  - Using code [`datasetMake`](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/datasetMake/) to make the data set; run `make_dataset_org.m` to generate sequence of images and stores as video.

  - Using compression rate = 50 and uncompressed videos, extracting pictures out and stored as TIFF(lossless)

* Use 2D model

  - Using [Timofte dataset](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/data/Timofte_dataset/) with size of 3288 independent image for single image reconstruction

  - Compress by using resize: downsampling

* Use 3D model

  - Using [short movie sequence](https://bitbucket.org/EDKLW/image-reconstruction-2019/src/master/Video_Stream/), frames extracted one by one from pictures

  - Compress by using resize: downsampling + upsampling, information missing

Climate Data:

* [Org data video](https://www.dropbox.com/s/15zmk2fxf5ggkeu/H_Field_Nl5.mat?dl=0)

* [Processed and stored in h5 format](https://www.dropbox.com/s/zj2695fogmu0uu1/climate_data.h5?dl=0)

### Code ####

- To generate dataset:

  - `extractor.py` can run to extract frames out from video; only need to run once at the very beginning;

- [Code](./code/): run the `main.py` file under each folder (or file named similar to this)

  - [code_2d](./code/code_2d/): basic model named SRCNN

  - [transfer_learning](./code/transfer_learning/): transfer learning on SRCNN model

  - [test_transfer_learning](./code/test_transfer_learning_acce/): 2D FSRCNN with transfer learning

  - [code_3d](./code/code_3d/): 3D SRnet model

  - [two_d+three_d](./code/two_d%2Bthree_d/): combined 2d and 3d model

  - [functional](./code/functional/): combined 2d and 3d model with a lambda data-reprocessing layer in between (not succeed)

  - [test2](./code/test2/): using the code of hyper-parameter tuning team to optimize hyper-parameter -- for SRCNN model

  - [test_3d](./code/test_3d): using the code of hyper-parameter tuning team to optimize hyper-parameter -- for 3d SRnet


To choose different dataset:
  - Check the `getpath.py` file under the exact directory of code, there are different choices for dataset (for example, shepp-logan phantom).

  - Change the size constant in main file according to the selected dataset `HEIGHT, WIDTH, AMOUNT, TOTAL AMOUNT`

  - Choose the data loading function inside main file `(x_set, y_set, ...)`


  

`code` folder gives general idea of models ==========================================

SUGGESTION: start from running `final` folder. Though other dataset are not stored as h5 files like climate data, the `final` folder is more clearly and nicely structured. There are more versions of functions under `code` folder. You may want to refer to the most updated ones. 

(`test_transfer_learning` is named as `fsrcnn` under `final`)

(`functional` folder under `code` is not runnable -- not fully developed)

 run `final` folder for final result with climate data =====================================

- [Experiment with climate data](./final/): run `main.py` and choose which model to run inside the code

  - `two_d.py`: transfer learing based SRCNN

  - `fsrcnn.py`: fsrcnn model with transfer learning

  - `code_3d.py`: 3D SRnet model, trained with raw data

  - `code_3d_processed.py`: 3D SRnet, trained with FSRCNN pre-processed data

  - `two_d_three_d.py`: combined 2d network and 3d raw-data-trained network

  - `two_d_processed_three_d.py`: combined 2d network and 3d fsrcnn-pre-trained network

  - models and images are stored in the folders under this direction





