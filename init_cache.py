import pickle
from collections import deque

INITIAL_EPSILON = 0.1  # Initial value of epsilon

def save_object(object, name):
    """
    Dump file into objects folder
    """
    with open("objects/" + name + ".pkl", "wb") as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)

def main():     
    """
    Initiate variable caching. Run this just once before running RLDinoRun.py.
    """
    save_object(INITIAL_EPSILON, "epsilon")
    t = 0
    save_object(t, "time")
    D = deque()
    save_object(D, "D")

main()