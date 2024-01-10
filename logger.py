from numpy import array
import json
class logger(object):
    def __init__(self, n_robots = 1):
        self.__n_robots = n_robots
        self.__log = dict()
        for i in range(n_robots):
            self.__log[i] = []

    def update(self, robot : int, information:list):
        self.__log[robot].append(information)

    def save(self, name=''):
        # Serializing json
        json_object = json.dumps(self.__log,
                                 indent = self.__n_robots)
        # Writing to sample.json
        with open("sample" + name + ".json", "w") as outfile:
            outfile.write(json_object)
    def reset(self):
        self.__log = dict()
        for i in range(self.__n_robots):
            self.__log[i] = []