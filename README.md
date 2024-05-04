# Real Time Object Detection: A comparative Analysis Of YOLOv1 And ResNet50 Archtectures
## Overview
The content of this repository related to the coursework of INM705. This readme file provides the detailed instruction on how to set-up and training procedure for the baseline model and also for the improved model. An implementation of "yolov1" for baseline and "ResNet50" for advanced model for object detection. Pytorch is served as a programming language for this project.
## Dataset 
I used a subset of PASCAL VOC 2007. After preprocessing, I split the data into train and test folders.
The datset is provided by the link https://1drv.ms/f/s!AviR1kFjoZOZbaMRIhYQlvfTXqA?e=nxzFIY
## Prerequisistes
Before running the code, ensure you have sucessfully installed the following files
- requirement.txt
- pip install -r requirement.txt
- Python 3.8
- GPU + CUDA
## Setup
### Configuration 
Set up the configuration file first. You need to put your API key in the configuration file if you want to load the weights and biases.
- config = load_config("config.yaml")
- print(config)
### Load model
Load the pretrained modle from the file 
- LOAD_MODEL_FILE= r'saved_models/model_full_data_80_tensor(0.9161).pth.tar' : put your path directory here.
### Running or Testing
For further development or testing you need to run it locally.You need to put your path directory in the file " inference models.ipynb"for baseline model 
and in the file " inference models.ipynb" for improved one.
## Training 
-  seed : 123
-  batch_size : 64
-  test_batch_size : 16
-  epochs : 100
-  weight_decay : 0
-  checkpoint_mAP : 0.85
- optimizer : Adam
-  baseline:
    learning_rate: 2e-5

-  improved:
    learning_rate: 1e-3
### TRain baseline model
- To train baseline model put your path directories in the file " train_baseline_model.py "
- To train imroved model put your path directories in the file " train_improved_model.py "
## Checkpoints 
- save_checkpoint(state, filename="my_checkpoint.pth.tar"):  Put your path directory
-    print("=> Saving checkpoint")
  -  torch.save(state, filename) :  To save checkpoints.
-  checkpoint_mAP 0.0 means save model at every epoch.
- checkpoint_mAP above 0.0 means model will only be saved when this threshold is met. 

