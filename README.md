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

  - [functional](./two-in-one_attempt/): combined 2d and 3d model with a lambda data-reprocessing layer in between (not succeed)

Run `final` folder for final result with climate data =====================================

- [Experiment with climate data](./final/): run `main.py` and choose which model to run inside the code

  - `two_d.py`: transfer learing based SRCNN

  - `fsrcnn.py`: fsrcnn model with transfer learning

  - `code_3d.py`: 3D SRnet model, trained with raw data

  - `code_3d_processed.py`: 3D SRnet, trained with FSRCNN pre-processed data

  - `two_d_three_d.py`: combined 2d network and 3d raw-data-trained network

  - `two_d_processed_three_d.py`: combined 2d network and 3d fsrcnn-pre-trained network

  - models and images are stored in the folders under this direction





