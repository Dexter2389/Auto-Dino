# Auto Dino
A Deep Convolutional Neural Network to play Google Chrome's Dino Run game by learning action patterns using a model-less Reinforcement Learning Algorithm using Nanand Optimizer.

## Requirements
* Set up a python environment with required dependencies installed.
* If you are familiar with Docker, you can use this [container]() that comes preinstalled with everything you need.

## Usage
### For Python Environment:
  #### 1. Downloading this Respository
  Start by [downloading](https://github.com/Dexter2389/Auto-Dino/archive/master.zip) or clone the repository:
  
  ```
  $ git clone https://github.com/Dexter2389/Auto-Dino.git
  $ cd Auto-Dino
  ```
  
  #### 2. Install Dependencies and get Chromedriver
  * If you are running this without the [Docker image](), you will need to get the chromedriver and place it in the working directory. This is a requirement to make the python script interact wih the Google Chrome. You can download the and extract by running:
  
  ```
  $ cd Auto-Dino
  $ wget https://chromedriver.storage.googleapis.com/2.41/chromedriver_linux64.zip
  $ unzip chromedriver_linux64.zip
  ```
  * You will also need to install specific python dependencies for this project:
  
  ```
  pip install -r requirements.txt
  ```
  #### 3. Start the training
  1. Run ```init_cache.py``` first time to initialize the file system structure.
  
  2. Run ```RLDinoRun.py``` to start the training of the Dino Run game.
  
  #### 4. Results
  Run ```DinoTrainingProgress.py``` to see the results of the training process
  
### For Docker Container:
  #### 1. xxxxx
  

## Acknowledgements
* Thanks to Ravi Munde for his [awsome article](https://blog.paperspace.com/dino-run/) because which this project as possible
