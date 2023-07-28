<a name="readme-top"></a>


<!-- PROJECT LOGO -->
[![Pytorch][Pytorch.org]][PyTorch-url]
<br />
<div align="center">

  <h3 align="center">Laboratory 1</h3>

  <p align="center">
    Depth, Residual Connections and Convolutions

  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#functionalities">Functionalities</a></li>
      </ul>
    </li>
    <li><a href="#excercise1">Excercise 1</a></li>
    <li><a href="#excercise2">Excercise 2</a></li>
    <li><a href="#excercise3">Excercise 3</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this Laboratory we explore the effects of Residual connections on deep architectures and we analyze the predictions of a convolutional architecture by exploiting Class Activation Maps.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

After downloading this directory and using it as the working directory, you can run the code in this folder. There are two executable files pipelinecopy.py and cam.py.

### Prerequisites

To execute this code you have to create the right virtual environment, following the instructions on the README in the main folder.

### Functionalities
Here are all the possible arguments to run pipelinecopy.py
```
Laboratory 1

  --model_name MODEL_NAME
                        Name of the Model
  --val_size VAL_SIZE   Validation Size
  --num_epochs NUM_EPOCHS
                        Number of Epochs
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch Size
  --dataset DATASET     Dataset
  --num_classes NUM_CLASSES
                        Number of Classes
  --residual_step RESIDUAL_STEP
                        Residual Step
  --input_size INPUT_SIZE
                        Flattened Size of Input Image (H*W*C)
  --width WIDTH         Number of Neurons in the hidden layers
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        Number of Hidden Layers
  --in_channels IN_CHANNELS
                        Number of Input Image Channels
  --out_channels OUT_CHANNELS
                        Number of Output Channels for the first block
  --num_levels NUM_LEVELS
                        Number of ResNet Block
  --num_conv_per_level NUM_CONV_PER_LEVEL
                        Number of Convolutions per Block
  --wandb WANDB         Logging with wandb; default: disabled; Pass online to log
  --wandb_proj_name WANDB_PROJ_NAME
                        Project name on wandb platform
  --train TRAIN         If True train the model; If False Evaluate the model
  --load_model_as LOAD_MODEL_AS
                        Specify a <name>.pt to load the model; used if train==False
  --save_model_as SAVE_MODEL_AS
                        Specify a <name>.pt to save the model
```


_Here are to examples to run the code_

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install NPM packages
   ```sh
   npm install
   ```

Here are all the possible arguments to run pipelinecopy.py
```
Laboratory 1

  --model_name MODEL_NAME
                        Name of the Model
  --val_size VAL_SIZE   Validation Size
  --num_epochs NUM_EPOCHS
                        Number of Epochs
  --lr LR               Learning rate
  --batch_size BATCH_SIZE
                        Batch Size
  --dataset DATASET     Dataset
  --num_classes NUM_CLASSES
                        Number of Classes
  --residual_step RESIDUAL_STEP
                        Residual Step
  --input_size INPUT_SIZE
                        Flattened Size of Input Image (H*W*C)
  --width WIDTH         Number of Neurons in the hidden layers
  --num_hidden_layers NUM_HIDDEN_LAYERS
                        Number of Hidden Layers
  --in_channels IN_CHANNELS
                        Number of Input Image Channels
  --out_channels OUT_CHANNELS
                        Number of Output Channels for the first block
  --num_levels NUM_LEVELS
                        Number of ResNet Block
  --num_conv_per_level NUM_CONV_PER_LEVEL
                        Number of Convolutions per Block
  --wandb WANDB         Logging with wandb; default: disabled; Pass online to log
  --wandb_proj_name WANDB_PROJ_NAME
                        Project name on wandb platform
  --train TRAIN         If True train the model; If False Evaluate the model
  --load_model_as LOAD_MODEL_AS
                        Specify a <name>.pt to load the model; used if train==False
  --save_model_as SAVE_MODEL_AS
                        Specify a <name>.pt to save the model
```


_Here an example to run the code_

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Excercise1

In this excercise we study the effect on increasing the depth of an MLP, trained on the MNIST dataset.


immagini


As we can see, increasing the depth doesn't mean increase in performance. In particular we see in this case that the performance when the depth increase degenerate. This can be seen comparing the gradients in the first layers between two MLPs at different depths.

immagini

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise2

Here we compare the performance of a convolutional model with and without residual connections.

immagini

We can observe that as with the MLPs in excercise 1, increasing the depth doesn't lead to an increase in performance. The use of skip connections(or residual connections), allows the information of the layers to be seen also in last layers. So we avoid the problem seen in the previous excercise.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise3

Following the "CAM paper" we implemented CAM acivation maps for our convolutional model. Activation maps we'll give us a sense of what part of the images the network attended to give us a prediction of the class it belongs to.

immagini

Of course if the prediction is correct, we can see that the activation maps usually corresponds to tghe area of interest that also we can discern; if instead the prediction of the network is wrong, this assumption is not true anymore.

immagini

We can observe that as with the MLPs in excercise 1, increasing the depth doesn't lead to an increase in performance. The use of skip connections(or residual connections), allows the information of the layers to be seen also in last layers. So we avoid the problem seen in the previous excercise.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[PyTorch-url]: https://pytorch.org/
[Pytorch.org]:https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white