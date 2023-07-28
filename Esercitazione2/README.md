<a name="readme-top"></a>


<!-- PROJECT LOGO -->
[![Pytorch][Pytorch.org]][PyTorch-url]
<br />
<div align="center">

  <h3 align="center">Laboratory 3</h3>

  <p align="center">
    Use of Large Language Models
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

In this Laboratory we experimented with LLM(Large Language Models). The objective was to become familiar with these models as they are the current major topic of AI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

After downloading this directory and using it as the working directory, you can run the code in this folder.

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

Following Karapathy's video, we played with his implementation of gpt2 to predict text. In particular we choose Dante's Divina Commedia as input, to generate Dante's style text with this model. Here there is an extract of the result:

```
Inferno: Canto IVII


Loco e` in parlan, due perche' ntrariva
  un pennto anto` rie metto al ciel pete;
  per che mi pur mo ch'a prio puntio.

Moltro al far, se benen di Guido altro mondo,
  aveder la men tre che tu siegu` forte
  verribi due miglia fede avanni

dipio` de l'altima bolgiate dal cozzo,
  si` fatto che tu vedi; edi la`, si` sazio?
  diche' di qual disio loco atte alquatta?

Se lungi la siete sta prima, a cotando
  la prima che tal piu` ti spadigne>>.

Poscia comincia' io a lui: <<Ose lezzo
  sotto lo scente ristra 'l mondo cinge da la pieta.
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise2

Here we used the Hugging Face library to achieve results similar to the previous excercise. We downloaded a pretrained model of GPT2 from Hugging Face and gave it a short input to make it predict the next tokens and generate a new part of the sentence.

Obtained the predictions, the decoding phase is fundamental to obtain good text.

```
The future of Artificial Intelligence is in the hands of the next generation of AI.

The future of Artificial Intelligence is in the hands of the next generation of AI.

The future of Artificial Intelligence is in the hands of the next generation of
```

The greedy search doesn't give us a good result overall as it chooses the highest probability token at each step, leaving behind possible sequences that can have higher probability in total. to resolve this problem we have for example, the Beam Search

```
0: The future of Artificial Intelligence is in the hands of the next generation of intelligent machines.

This article was originally published on Wired.com.
1: The future of Artificial Intelligence is in the hands of the next generation of intelligent machines.

This article was originally published on Wired.com
2: The future of Artificial Intelligence is in the hands of the next generation of intelligent machines.

This article was originally published on Medium.
3: The future of Artificial Intelligence is in the hands of the next generation of intelligent machines.

This article was originally published on TechRepublic.com
4: The future of Artificial Intelligence is in the hands of the next generation of intelligent machines.

This article was originally published on TechRepublic.com.
```

Here we choose to use 5 beams and to produce 5 sequences. To avoid repeating words a parameter no_repeat_ngram_size is used. We can see this method can give us various results to choose from. But they are not that different from the greedy method.

```
The future of Artificial Intelligence is coming up. But the real revolution has yet to take place – even at the most basic level of human interaction.

According toGerman research chief Andreas Thorne (d. 2005), "we
```

This last example is more human-like. This decoding method is Top-k, the one used in GPT2.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Excercise3

We used LLMs to perform sentence classification. In particular we employed two different LLMs: GPT2 and DistillBERT. Different from BERT that as a special class token to perform classification, GPT2 was not designed for that. We decided to use the last hidden state(excluding padding) to perform classification as in theory contains information on the whole sentence. Than we tried a weighted average on all the tokens, in which the weights decrease going left to right.

To perform classification we considered a sentiment analysis task on a dataset composed of Twitter posts.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[PyTorch-url]: https://pytorch.org/
[Pytorch.org]:https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white