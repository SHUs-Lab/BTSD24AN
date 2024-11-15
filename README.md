# Transformer_OD_TPU

Exploration of TPU Architectures for Optimized Transformer Performance in Image Detection of Drainage Crossings

## Repository Structure

A working structure for a Dockerized application of this repository, with results, visualizations, and the dataset stored outside the code repository, should follow this structure. The result_folder and visualizations directories are created by running ```main.py``` and the notebooks under the visualization subdirectory.
```
/workspace/
├── Transformer_OD_TPU
│   ├── ScaleSim
│   │   └── Experiment Results
│   ├── datasets
│   ├── models
│   ├── notebooks
│   │   ├── data_preparation
│   │   ├── tests
│   │   └── visualization
│   └── util
|
├── processed_data
│   ├── initial_data
│   │   ├── annotations
│   │   ├── test
│   │   ├── train
│   │   └── validate
│   └── transfer_data
│       ├── annotations
│       └── test
|
├── result_folder
└── visualizations
```

## Data Preprocessing, Testing, and Visualization using Notebooks

Raw data were preprocessed and data statistics were extracted using the notebooks under the data_preparation subdirectory. Data are preprocessed and object jsons are created for each crop level to conform to the coco dataset spec. These notebooks should not be used on the preprocessed dataset, and are included to show the adaptation from the initial raw dataset.

Testing performance is measured using the notebook under the tests subdirectory, and visualizations and graphs are generated using the visualization subdirectory.

## Running the Model in Docker

Ensure that the data folder and Transformer_OD_TPU repository are in the same work directory, ```/path/to/work/directory/```.

Navigate to the ```Transformer_OD_TPU``` repository and build the docker image by running:

```docker build -t transformer_od_tpu .```

Then, after building the Docker image, run a jupyter lab server using:

```docker run --gpus all -it -v /path/to/work/directory:/workspace/ -p 8888:8888 transformer_od_tpu```

Then, the program can be run using the following command in the ```Transformer_OD_TPU``` directory, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output --num_workers 0 --batch_size 8 --crop 400 --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth```

This will run the script with default parameters, loading in pretrained object detection weights.

For the experiments performed in the accompanying paper 'Exploration of TPU Architectures for the Optimized Transformer in Drainage Crossing Detection', experiments were run using the above command, changing the ```--crop``` argument to 256, 400, 600, and 800.

To train from scratch, a binary DETR model can be built by passing ```num_classes 2``` as an argument, e.g.

```torchrun main.py --coco_path /workspace/processed_data --output_dir /workspace/detr_output_scratch --num_workers 0 --batch_size 8 --crop 400 --num_classes 2```

## Dataset

The full preprocessed dataset is available for download here: https://figshare.com/articles/dataset/Processed_Data_for_Exploration_of_TPU_Architectures_for_the_OptimizedTransformer_in_Drainage_Crossing_Detection_/27711249?file=50460957

In order to test model transferrability to an unseen watershed, we first partition the data into the initial dataset and the transfer dataset.

The initial dataset, which includes watersheds from NE, CA, and IL, is randomly paritioned into training (70%), validation (20%), and testing (10%) sets.

The transfer dataset, which includes data from the ND watershed, is used in its entirety to test model transferrability to unseen HRDEM data.

The initial and transfer dataset are normalized according to their corresponding dataset mean and standard deviation to simulate the model's use in inference. It is expected that for a given inference dataset, the user will first find the mean and standard deviation of the inference dataset before inputting these in the dataset script.

The dataset statistics used for the initial dataset were as follows: ```mean: 6.6374, std: 10.184```

The statistics for the transfer dataset were as follows: ```mean: 0.7294, std: 9.3929```




