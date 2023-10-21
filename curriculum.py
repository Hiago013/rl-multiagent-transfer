import numpy as np


class curriculum:
    def __init__(self, states:np.array, obstacles:list, elevator, col:int, row:int):
        self.states = states
        self.obstacles = obstacles
        self.col = col
        self.row = row
        self.axis_grid_position, self.axis_pick_up,\
        self.axis_drop_off, self.axis_flag, self.axis_dynamic = 4, 3, 2, 1, 0
        self.elevator = elevator
        self.__load_stages()
    
    def __load_stages(self):
        remove_first_axis = set(np.arange(self.col * self.row, self.col * self.row * 2))
        remove_first_axis.update(set(self.obstacles))
        remove_first_axis.update(self.elevator)

        self.first_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.first_stage = np.delete(self.first_stage, (1, 2), axis=self.axis_flag)
        self.first_stage = self.first_stage.flatten()


        self.second_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.second_stage = np.delete(self.second_stage, (0, 2), axis=self.axis_flag)
        self.second_stage = self.second_stage.flatten()


        remove_third_axis = set(np.arange(0, self.col * self.row))
        remove_third_axis.update(set(self.obstacles))

        self.third_stage = np.delete(self.states, list(remove_third_axis), axis = self.axis_grid_position)
        self.third_stage = np.delete(self.third_stage, (0, 2), axis = self.axis_flag)
        self.third_stage =  self.third_stage.flatten()

        self.forth_stage = np.delete(self.states, list(remove_third_axis), axis = self.axis_grid_position)
        self.forth_stage = np.delete(self.forth_stage, (0, 1), axis = self.axis_flag)
        self.forth_stage = self.forth_stage.flatten()


        self.fifth_stage = np.delete(self.states, list(remove_first_axis), axis=self.axis_grid_position)
        self.fifth_stage = np.delete(self.fifth_stage, (0, 1), axis=self.axis_flag)
        self.fifth_stage = self.fifth_stage.flatten()

        self.stages = {0:self.first_stage,
                       1:self.second_stage,
                       2:self.third_stage,
                       3:self.forth_stage,
                       4:self.fifth_stage}
    
    def get_stage(self, stage = 5):
        if stage >= 5:
            return self.states.flatten()
        return self.stages[stage]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from brain import brain
    from GridWorld import GridWorld

    env = GridWorld(5, 5, -1, 5, 10, 100, 1)
    env.set_pick_up([1, 2, 3])
    env.set_drop_off([35, 39])
    env.set_obstacles([0, 4, 5, 9, 10, 14, 15, 19, 20, 24, 36, 38, 41, 43])
    env.possible_states()
    env.load_available_action()
    env.load_available_flag_dynamic()
    agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    agent.filter_q_table(env.state_action)
    num_epochs = 10000
    score = np.zeros(num_epochs)
    sum_score = 0

    crr = curriculum(env.all_states, env.obstacles, {21, 23, 26, 28}, 5, 5)
    print(len(crr.first_stage))
    print(len(crr.second_stage))
    print(len(crr.third_stage))
    print(len(crr.forth_stage))
    print(len(crr.fifth_stage))
    print(len(crr.states.flatten()))

