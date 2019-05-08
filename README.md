# Auto Dino
A 2013 publication by DeepMind titled [‘Playing Atari with Deep Reinforcement Learning’](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) introduced a new deep learning model for reinforcement learning, and demonstrated its ability to master difficult control policies for Atari 2600 computer games, using only raw pixels as input. In this repository, a Deep Convolutional Neural Network will learn to play Google Chrome's Dino Run game by learning action patterns using a model-less Reinforcement Learning Algorithm using Nadam Optimizer. Moreover, this is using the latest TensorFlow 2.0. So there will be no version confilts while running this project on TF 2.0.

## Requirements
* Set up a python environment with required dependencies installed.
* If you are familiar with Docker, you can use this [container]() that comes preinstalled with everything you need (Coming Soon).

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
  
  2. Run ```RLDinoRunTF_2.0.py``` to start the training of the Dino Run game.
  
  #### 4. Results
  Run ```DinoTrainingProgress.py``` to see the results of the training process
  
### For Docker Container (Coming Soon!):
  #### 1. xxxxx
  

## Acknowledgements
* Thanks to Ravi Munde for his [awesome article](https://blog.paperspace.com/dino-run/) because which this project as possible
