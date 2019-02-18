import numpy as np
from PIL import Image
import cv2
import io
import time
import pandas as pd
from random import randint
import os

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD, Adam, Nadam
from tensorflow.keras.callbacks import TensorBoard

from collections import deque
import random
import pickle
import base64
from io import BytesIO
import json

# Path Variables
GAME_URL = "http://wayou.github.io/t-rex-runner/"
CHROME_DRIVER_PATH = "./chromedriver"
LOSS_FILE_PATH = "./objects/loss_df.csv"
ACTIONS_FILE_PATH = "./objects/actions_df.csv"
Q_VALUE_FILE_PATH = "./objects/q_values.csv"
SCORE_FILE_PATH = "./objects/scores_df.csv"

# Script to create id for canvas for faster selections from Document Object MOdel (DOM)
init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"

# Script to get image from canvas
getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
return canvasRunner.toDataURL().substring(22)"

# Game Parameter Constants
ACTIONS = 2  # Possible actions: "Jump" or "Do Nothing"
GAMMA = 0.9  # Decay rate of past observations, original 0.9
OBSERVATION = 100.  # Timesteps to observe before training
EXPLORE = 100000  # Frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # Final value of epsilon
INITIAL_EPSILON = 0.1  # Initial value of epsilon
REPLAY_MEMORY = 80000  # Number of previous transitions to remember
BATCH = 32  # Size of minibatch
FRAME_PER_ACTION = 1
LEARNING_RATE = 0.0003

img_rows, img_cols = 80, 80
img_channels = 4  # We stack 4 frames

# Initialize log structures from file if they exist or else create new
loss_df = pd.read_csv(LOSS_FILE_PATH) if os.path.isfile(
    LOSS_FILE_PATH) else pd.DataFrame(columns=["loss"])
score_df = pd.read_csv(SCORE_FILE_PATH) if os.path.isfile(
    SCORE_FILE_PATH) else pd.DataFrame(columns=["Scores"])
actions_df = pd.read_csv(ACTIONS_FILE_PATH) if os.path.isfile(
    ACTIONS_FILE_PATH) else pd.DataFrame(columns=["Actions"])
q_values_df = pd.read_csv(Q_VALUE_FILE_PATH) if os.path.isfile(
    Q_VALUE_FILE_PATH) else pd.DataFrame(columns=["qvalues"])

# Some basic pre-processing function


def save_object(object, name):
    """
    Dump file into objects folder
    """
    with open("objects/" + name + ".pkl", "wb") as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


def load_object(name):
    """
    Loads file Dump
    """
    with open("objects/" + name + ".pkl", "rb") as f:
        return pickle.load(f)


def process_image(image):
    """
    Processes the image to use futher
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # RGB to Gray scale
    image = image[:300, :500]  # Crop Region of Interest(ROI)
    image = cv2.resize(image, (80, 80))
    return image


def grab_screen(_driver):
    """
    Grabs the screen
    """
    image_b64 = _driver.execute_script(getbase64Script)
    screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
    image = process_image(screen)  # Processing image is required
    return image


def show_image(graphs=False):
    """
    Shows images in new window
    """
    while True:
        screen = (yield)
        window_title = "Logs" if graphs else "Game_play"
        cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
        image_size = cv2.resize(screen, (800, 400))
        cv2.imshow(window_title, screen)
        if (cv2.waitKey(1) & 0xFF == ord("q")):
            cv2.destroyAllWindows()
            break

# Trainig varialbes saed as checkpoints to filesystem to resume training from the same step

class Game():
    """
    Selenium interfacing between the python and browser
    """

    def __init__(self, custom_config=True):
        """
        Launch the broswer window using the attributes in chrome_options
        """
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        chrome_options.add_argument("--mute-audio")
        self._driver = webdriver.Chrome(
            executable_path=CHROME_DRIVER_PATH, chrome_options=chrome_options)

        self._driver.set_window_position(x=-10, y=0)
        self._driver.get("chrome://dino")
        self._driver.execute_script("Runner.config.ACCELERATION=0")
        self._driver.execute_script(init_script)

    def get_crashed(self):
        """
        return True if the agent as crashed on an obstacles. Gets javascript variable from game decribing the state
        """
        return self._driver.execute_script("return Runner.instance_.crashed")

    def get_playing(self):
        """
        returns True if game in progress, false is crashed or paused
        """
        return self._driver.execute_script("return Runner.instance_.playing")

    def restart(self):
        """
        Sends a signal to browser-javascript to restart the game
        """
        self._driver.execute_script("Runner.instance_.restart()")

    def press_up(self):
        """
        Sends a single to press up get to the browser
        """
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    def get_score(self):
        """
        Gets current game score from javascript variables
        """
        score_array = self._driver.execute_script(
            "return Runner.instance_.distanceMeter.digits")
        # the javascript object is of type array with score in the formate[1,0,0] which is 100.
        score = ''.join(score_array)
        return int(score)

    def pause(self):
        """
        Pause the game
        """
        return self._driver.execute_script("return Runner.instance_.stop()")

    def resume(self):
        """
        Resume a paused game if not crashed
        """
        return self._driver.execute_script("return Runner.instance_.play()")

    def end(self):
        """
        Close the browser and end the game
        """
        self._driver.close()


class DinoAgent:
    """
    Reinforcement Agent
    """

    def __init__(self, game):  # takes game as input for taking actions
        self._game = game
        self.jump()  # to start the game, we need to jump once

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()


class Game_State:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game
        # Display the processed image on screen using openCV, implemented using python coroutine
        self._display = show_image()
        self._display.__next__()  # Initilize the display coroutine

    def get_state(self, actions):
        """
        Returns the Experience of one itereationas a tuple
        """
        actions_df.loc[len(actions_df)
                       ] = actions[1]  # Storing actions in a dataframe
        score = self._game.get_score()
        reward = 0.1
        is_over = False  # Game Over
        if actions[1] == 1:
            self._agent.jump()
        image = grab_screen(self._game._driver)
        self._display.send(image)  # Display the image on screen
        if self._agent.is_crashed():
            # Log the score when the game is over
            score_df.loc[len(loss_df)] = score
            self._game.restart()
            reward = -1
            is_over = True
        return image, reward, is_over

def buildModel():
    print("Building Convolutional Neural Network")
    model = Sequential()
    model.add(Conv2D(32, (8, 8), padding="same", strides=(4, 4), input_shape=(
        img_cols, img_rows, img_channels)))  # First layer of 80*80*4 with 32 filters
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    # Second layer of 40*40*4 with 64 filters
    model.add(Conv2D(64, (4, 4), strides=(2, 2),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    # Third layer of 30*30*4 with 64 filters
    model.add(Conv2D(64, (3, 3), strides=(1, 1),  padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(ACTIONS))
    #adam = Adam(lr=LEARNING_RATE)
    nadam = Nadam(lr=LEARNING_RATE)
    model.compile(loss="mse", optimizer=nadam)

    # Creating model file if not present
    if not os.path.isfile(LOSS_FILE_PATH):
        model.save_weights("model.h5")
    print("Finished building the Convolutional Neural Network")
    return model


def trainNetwork(model, game_state, observe=False):
    """
    Main Training module

    Parameters:
        model => Keras Model to be trained
        game_state => Game State module with access to game environment and dino
        observe => Flag to indicate if the model is to be trained(weights updates), else just play
    """
    last_time = time.time()  # Store the previous observations in replay memory
    D = load_object("D")  # Load from file system

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1  # 0 => Do Nothing ; 1 => Jump

    # Get next step after performing the action
    x_t, r_0, terminal = game_state.get_state(do_nothing)

    # Stack 4 images to create a placeholder input
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  # 1*20*40*4

    initial_state = s_t

    if observe:  # We keep observing, never train
        OBSERVE = 99999
        epsilon = FINAL_EPSILON
        print("Loading weights to the CNN")
        model.load_weights("model.h5")
        #adam = Adam(lr=LEARNING_RATE)
        nadam = Nadam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=nadam)
        print("Loading weights Successful")

    else:  # We go to training mode
        OBSERVE = OBSERVATION
        epsilon = load_object("epsilon")
        model.load_weights("model.h5")
        #adam = Adam(lr=LEARNING_RATE)
        nadam = Nadam(lr=LEARNING_RATE)
        model.compile(loss="mse", optimizer=nadam)

    # Resume from the previous time step stored in the file system
    t = load_object("time")

    while True:  # Endless running
        loss = 0
        Q_sa = 0
        action_index = 0
        r_t = 0  # Reward at 4
        a_t = np.zeros([ACTIONS])  # Actions at t

        # Choose an action epsilon greedy
        if t % FRAME_PER_ACTION == 0:  # Parameter to skip frames for actions
            if random.random() <= epsilon:  # Randomly explore an action
                print("---------Random Action---------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:  # Predict the output
                # Input a stack of 4 images, get the prediction
                q = model.predict(s_t)
                max_Q = np.argmax(q)  # Choosing index with maximum "q" value
                action_index = max_Q
                a_t[action_index] = 1  # 0 => Do Nothing, 1 => Jump

        # We reduce the epsilon (exploration parameter) gradually
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        # Run the selected action and observed next state and reward
        x_t1, r_t, terminal = game_state.get_state(a_t)

        # FPS of the game
        print("FPS: {0}".format(1/(time.time()-last_time)))
        last_time = time.time()

        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)  # 1x20x40x1

        # Append the new image to input stack and remove the first one
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

        # Store the transition in D
        D.append((s_t, action_index, r_t, s_t1, terminal))
        if len(D) > REPLAY_MEMORY:
            D.popleft()

        # Only train if done observing
        if t > OBSERVE:
            # Sample a minibatch to train on
            minibatch = random.sample(D, BATCH)
            inputs = np.zeros(
                (BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  # 32x20x40x4
            targets = np.zeros((inputs.shape[0], ACTIONS))

            # Now we do the experience replay
            for i in range(0, len(minibatch)):
                state_t = minibatch[i][0]  # 4D stack of images
                action_t = minibatch[i][1]  # This is the action index
                reward_t = minibatch[i][2]  # Reward at state_t due to action_t
                state_t1 = minibatch[i][3]  # Next State
                # Wheather the agent died or survided due to the action
                terminal = minibatch[i][4]

                inputs[i:i+1] = state_t

                targets[i] = model.predict(state_t)  # Predicted "q" value
                # Predict "q" value for next step
                Q_sa = model.predict(state_t1)

                if terminal:
                    # If terminated, only equal to reward
                    targets[i, action_t] = reward_t
                else:
                    targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)

            loss += model.train_on_batch(inputs, targets)
            loss_df.loc[len(loss_df)] = loss
            q_values_df.loc[len(q_values_df)] = np.max(Q_sa)

        # Reset game to initial frame if terminated
        s_t = initial_state if terminal else s_t1
        t += 1

        # Save progress every 500 iterations
        if t % 500 == 0:
            print("Now we save model during training")
            game_state._game.pause()  # Pause game while saving to filesystem
            model.save_weights("model.h5", overwrite=True)
            save_object(D, "D")  # Saving episodes
            save_object(t, "time")  # Caching time steps
            # Cache epsilon to avoide repeated randomness in actions
            save_object(epsilon, "epsilon")
            loss_df.to_csv(LOSS_FILE_PATH, index=False)
            score_df.to_csv(SCORE_FILE_PATH, index=False)
            actions_df.to_csv(ACTIONS_FILE_PATH, index=False)
            q_values_df.to_csv(Q_VALUE_FILE_PATH, index=False)

            with open("model.json", "w") as outfile:
                json.dump(model.to_json(), outfile)

            game_state._game.resume()

        # Print Info
        state = ""
        if t <= OBSERVE:
            state = "observe"
        elif t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, "/ EPSILON", epsilon, "/ ACTION",
              action_index, "/ REWARD", r_t, "/ Q_MAX ", np.max(Q_sa), "/ Loss ", loss)

    print("Episode finished!")
    print("-------------------------------")

# Main Function

def playGame(observe=False):
    try:
        game = Game()
        dino = DinoAgent(game)
        game_state = Game_State(dino, game)
        model = buildModel()
        try:
            trainNetwork(model, game_state, observe=observe)
        except FileNotFoundError:
            print("Looks like init_cache.py was not executed ever.\nDoing that for you!!! Sit back and relax.....")
            os.system('python init_cache.py')
            trainNetwork(model, game_state, observe=observe)
        except StopIteration:
            game.end()
    except selenium.common.exceptions.WebDriverException:
        print("No driver")

playGame(observe=False)    