<a name="readme-top"></a>


<!-- PROJECT LOGO -->
[![Pytorch][Pytorch.org]][PyTorch-url]
<br />
<div align="center">

  <h3 align="center">Laboratory 4</h3>

  <p align="center">
    Out of Distribution Detection, Adversarial Attack and Adversarial Training

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

In this Laboratory we implemented a simple OOD detection ppipeline and the FGSM(Fast Gradient Sign Method) attack to evaluate the produced adversarial samples and the robustness of a convolutional model to adversarial attacks. Then we used this attack to perform adversarial training and implemented also a Targeted FGSM attack.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

After downloading this directory and using it as the working directory, you can run the code in this folder. There are two executable files pipeline_FGSM and pipeline_OOD.py.
To run the OOD pipeline you will need a previously trained model.

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

Here we implemented a simple OOD detection pipeline. We used the logits of the predictions as a score to evaluate if a sample is in or out odf distribution. We built histograms using different metrics and also ROC cure and Precision-Recall curves.


immagini


We can see that the mean on logits doesn't give any kind of distinction between the two distributions, as one could imagine knowing the smoothing effect of the mean operator. The max and the variance though being respectively an existence operator and a distribution metric can achieve a certain level of discernment between IN and OUT of distribution samples.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise2

We implemented the FGSM attack to experiment with adversarial attacks and adversarial training.

immagini


As we can see attacking the model brings down the performance of the model. But we can also observe that beyond a certain epsilon value we see a degradation of the image that is pretty visible to a human observer. So the attack is more effective but can be easily be detected.

Enanching the model with adversarial training makes it more robust to this kind of attacks. We can see that the performance for epsilon = ? are better than the previous case, when no adversarial training was performed.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise3

A variant of FGSM is the Targeted FGSM. Here we can choose a target class to guide the adversarial attack. The objective is make the model classify the input images as the target class.

immagini

In the figures above we can see the progression of the attack. As we can see we needed to perform more than one iteration of the attack to achieve better results overall. But this came at the cost of the images' quality, as the effect is similar to the previous excercise in which increasing epsilon degraded the images.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[PyTorch-url]: https://pytorch.org/
[Pytorch.org]:https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white