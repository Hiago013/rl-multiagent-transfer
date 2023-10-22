from stack import Stack
from GridWorld import GridWorld
from brain import brain
import numpy as np
from matplotlib import pyplot as plt
from transfer import transfer

def state2cartesian(state):
    x, y = divmod(state, 9)
    return x * 50, y * 50

def cartesian2state(cartesian_point):
    x, y = cartesian_point
    x = x // 50
    y = y // 50
    return 9 * x + y
#####

class multi_agent():
    def __init__(self, agent : brain, grid_world : GridWorld, n_agents = 1):
        # Initialize a multi-agent with a main agent, the environment, and the number of agents
        self.main_agent = agent
        self.env = grid_world
        self.n_agents = n_agents
        self.q_table = agent.get_q_table()
        self.reward = [0 for n_agnt in range(n_agents)]
        self.done = [False for n_agnt in range(n_agents)]
        self.stack_stay = [Stack() for i in range(n_agents)]
        self.metrics = dict()
        for agent in range(n_agents):
            self.metrics[agent] = {'step': 0,
                                   'pick': 0,
                                   'drop': 0,
                                   'up':0,
                                   'down':0,
                                   'left':0,
                                   'right':0,
                                   'wait':0}
        self.hard_n_agents = n_agents
        self.hard_done = self.done
        self.ID = [[agent, agent] for agent in range(n_agents)]
    
    def update_drop(self, agent):
        # Update the 'drop' metric for a specific agent
        self.metrics[agent]['drop'] += 1
    
    def update_pick(self, agent):
        # Update the 'pick' metric for a specific agent
        self.metrics[agent]['pick'] += 1

    def update_step(self, agent):
        # Update the 'step' metric for a specific agent
        self.metrics[agent]['step'] +=  1
    
    def update_up(self, agent):
        # Update the 'up' metric for a specific agent
        self.metrics[agent]['up'] +=  1

    def update_down(self, agent):
        # Update the 'down' metric for a specific agent
        self.metrics[agent]['down'] +=  1

    def update_left(self, agent):
        # Update the 'left' metric for a specific agent
        self.metrics[agent]['left'] +=  1

    def update_right(self, agent):
        # Update the 'right' metric for a specific agent
        self.metrics[agent]['right'] +=  1

    def update_wait(self, agent):
        # Update the 'wait' metric for a specific agent
        self.metrics[agent]['wait'] +=  1

    def set_q_table(self, q_table:np.array):
        # Set the Q-table for the multi-agent
        self.q_table = q_table
        
    def set_n_agents(self, n_agents:int):
        # Set the number of agents in the multi-agent
        self.n_agents = n_agents
        self.reward = [0 for n_agnt in range(n_agents)]
        self.done = [False for n_agnt in range(n_agents)]
        self.stack_stay = [Stack() for i in range(n_agents)]

    def get_q_table(self):
        # Get the Q-table of the multi-agent
        return self.q_table
    
    def save(self, filename):
        # Save the Q-table to a file
        if not '.txt' in filename:
            filename += '.txt'
        np.savetxt(filename, self.main_agent.get_q_table())
    
    def load(self, filename):
        # Load the Q-table from a file
        if not '.txt' in filename:
            filename += '.txt'
        self.set_q_table(np.loadtxt(filename))
    
    def colision_actions(self):
        # Define collision actions in the Q-table
        self.main_agent.colision_actions()
        start = 36450
        left = 3
        right = 2
        up = 1
        down = 0
        self.q_table[start: 2 * start, left] = -100  # Remove left action
        self.q_table[2 * start: 3 * start, right] = -100  # Remove right action
        self.q_table[3 * start: 4 * start, [right, left]] = -100  # Remove left and right actions
        self.q_table[4 * start: 5 * start, up] = -100  # Remove up action
        self.q_table[5 * start: 6 * start, [up, left]] = -100  # Remove up and left actions
        self.q_table[6 * start: 7 * start, [up, right]] = -100  # Remove up and right actions
        self.q_table[7 * start: 8 * start, [up, right, left]] = -100  # Remove up, left, and right actions
        self.q_table[8 * start: 9 * start, down] = -100  # Remove down action

        self.q_table[9 * start: 10 * start, [down, left]] = -100  # Remove down and left actions
        self.q_table[10 * start: 11 * start, [down, right]] = -100  # Remove down and right actions
        self.q_table[11 * start: 12 * start, [down, right, left]] = -100  # Remove down, left, and right actions
        self.q_table[12 * start: 13 * start, [down, up]] = -100  # Remove down and up actions
        self.q_table[13 * start: 14 * start, [down, up, left]] = -100  # Remove down, up, and left actions
        self.q_table[14 * start: 15 * start, [down, up, right]] = -100  # Remove down, up, and right actions
        self.q_table[15 * start: 16 * start, [down, up, right, left]] = -100  # Remove down, up, left, and right actions


    def reset(self):
        # Reset the environment and agents to their initial state
        self.n_agents = self.hard_n_agents
        n_agents = self.n_agents
        self.observations = [self.env.reset() for i in range(self.n_agents)]
        self.grid_positions = [self.env.what_position(i) for i in self.observations]
        self.data = dict()

        # Ensure that initial grid positions are unique for all agents
        while len(self.grid_positions) != len(set(self.grid_positions)):
            self.observations = [self.env.reset() for i in range(self.n_agents)]
            self.grid_positions = [self.env.what_position(i) for i in self.observations]

        # Store observations and related data for each agent
        for i in range(self.n_agents):
            states = self.env.get_states(self.observations[i])
            self.data[i] = [self.observations[i]] + list(states)

        # Initialize metrics and other attributes
        self.atualizar_flag_all_agents()
        self.metrics = dict()
        for agent in range(n_agents):
            self.metrics[agent] = {'step': 0,
                                'pick': 0,
                                'drop': 0,
                                'up':0,
                                'down':0,
                                'left':0,
                                'right':0,
                                'wait':0}
        self.hard_n_agents = n_agents
        self.hard_done = self.done
        self.ID = [[agent, agent] for agent in range(n_agents)]

        return self.observations

    
    def zero_flag(self):
    # Reset the flag for all agents
        observations_copy = self.observations
        for idx, observation in enumerate(observations_copy):
            flag = 0
            self.data[idx][1] = flag
            self.env.current_dynamic, self.env.current_flag, \
            self.env.current_drop_off, self.env.current_pick_up,\
            self.env.grid_position = self.data[idx][1:]
            self.observations[idx] = self.data[idx][0] = self.env.att_state(self.data[idx][-1])

    def _att_flag(self, n_agent, observation):
    # Update and synchronize the flag for a specific agent based on other agents' positions
        flag = [0, 0, 0, 0]
        self.env.grid_position = self.env.what_position(observation)
        available_action = self.env.available_action(observation)[:-1]
        possibles_states = [(act, self.env.what_position(self.env.move(act))) for act in available_action]
        for i in range(self.n_agents):
            if i != n_agent:
                for act, state in possibles_states:
                    if self.data[i][-1] == state:
                        flag[act] = 1
        current_flag = self.env.binary2decimal(''.join(map(str, flag)))
        
        self.data[n_agent][1] = current_flag

        self.env.current_dynamic, self.env.current_flag, \
        self.env.current_drop_off, self.env.current_pick_up,\
        self.env.grid_position = self.data[n_agent][1:]
        return self.env.att_state(self.data[n_agent][-1])
    
    
    def step(self):
        for agent in range(self.n_agents):
            observation = self.observations[agent]
            self.env.set_state(observation)
            available_actions = self.env.available_action(observation)
            action = self.main_agent.choose_best_action(observation, available_actions)
            observation_, reward, done = self.env.step(action)

            # Update the environment state and check for flag-related conditions
            self.env.set_state(observation_)
            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2 and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up = temporary_state[1]
                    observation_ = self.env.att_state(gp)

            # Update various agent metrics based on the action and rewards
            self.update_step(agent)
            if 99 == reward:
                self.update_drop(self.ID[agent][1])
            if 49 == reward:
                self.update_pick(self.ID[agent][1])
            
            # Update specific action counters for the agent
            if action == 0:
                self.update_down(self.ID[agent][1])
            elif action == 1:
                self.update_up(self.ID[agent][1])
            elif action == 2:
                self.update_right(self.ID[agent][1])
            elif action == 3:
                self.update_left(self.ID[agent][1])
            else:
                self.update_wait(self.ID[agent][1])
            
            # Update the agent's observation, states, grid positions, and related data
            self.observations[agent] = observation_
            states = self.env.get_states(observation_)
            self.grid_positions[agent] = gp
            self.data[agent] = [observation_]+ list(states)
            self.done[agent] = done
            self.hard_done[self.ID[agent][1]] = done
            self.reward[agent] = reward
            self.atualizar_flag_all_agents()

        # Try to handle agents that are marked as "done"
        try:
            [self.done_agent(agent) for agent in range(self.n_agents) if self.done[agent] == True]
        except:
            pass

        # Return the updated observations, grid positions, rewards, and done status
        return self.observations, [self.env.what_position(a) for a in self.observations], self.reward, self.done

    
    def set_ep(self, min_ep, max_ep, maxEpisode):
    # Set the exploration rate (epsilon) using an epsilon-decay function
        self.main_agent.epfunction(max_ep, min_ep, maxEpisode)

    def cut_actions_dynamic(self, observation, potential_actions):
        # Get the dynamic flag from the observation
        flag_dynamic = self.env.get_states(observation)[0]
        
        # Convert the dynamic flag to binary representation
        binary = self.env.decimal2binary(flag_dynamic)

        # Iterate through the binary representation
        for idx, item in enumerate(binary):
            if int(item) == 1:
                # If the flag is set (1), remove the corresponding action from potential actions
                potential_actions = np.setdiff1d(potential_actions, idx)
        
        return potential_actions

    
    def step_agents(self, episode, maxEpisode):
        actions = np.zeros(self.n_agents, dtype=np.uint8)
        for agent in range(self.n_agents):
            observation = self.observations[agent]
            self.env.set_state(observation)
            available_action = self.env.available_action(observation)
            action = self.main_agent.choose_action(observation, episode, maxEpisode, available_action)
            actions[agent] = action
            observation_, reward, done = self.env.step(action)

            # Update the Q-table and the environment state
            self.main_agent.learn(observation, action, reward, observation_, done)
            self.env.set_state(observation_)

            # Check for flag-related conditions and handle them
            dynamic, flag, drop, pick, gp = self.env.get_states(observation_)
            if flag == 2:  # and gp < self.env.col * self.env.col:
                if len(self.book) > 0:
                    temporary_state = self.book.pop()
                    self.env.current_flag = 0
                    self.env.current_drop_off = temporary_state[0]
                    self.env.current_pick_up = temporary_state[1]
                    observation_ = self.env.att_state(gp)
            
            # Update the agent's observation, states, grid positions, rewards, and done status
            self.observations[agent] = observation_
            states = self.env.get_states(observation_)
            self.grid_positions[agent] = gp
            self.data[agent] = [observation_] + list(states)
            self.done[agent] = done
            self.reward[agent] = reward
            self.atualizar_flag_all_agents()

        # Try to handle agents that are marked as "done"
        try:
            [self.done_agent(agent) for agent in range(self.n_agents) if self.done[agent] == True]
        except:
            pass

        # Create info with grid positions and chosen actions and return it
        info = {'grid_position': [self.env.what_position(a) for a in self.observations],
                'action': actions}
        return self.observations, self.reward, self.done, info

        

    def done_agent(self, n):
        if self.n_agents > 1:
            # Remove the agent's information from the lists
            del self.grid_positions[n]
            del self.observations[n]

            # Reconstruct the data dictionary and update the number of agents
            self.data = dict()
            for i in range(len(self.grid_positions)):
                states = self.env.get_states(self.observations[i])
                self.data[i] = [self.observations[i]] + list(states)
            self.n_agents = len(self.grid_positions)

            # Reset the 'done' status and update flag information for remaining agents
            self.done = [False for n_agents in range(self.n_agents)]
            self.atualizar_flag_all_agents()

            # Handle changes in the ID list and update 'hard_done'
            idx = np.where(self.hard_done == 1)[0][0]
            self.hard_done[idx] = 2

            if idx + 1 != self.n_agents + 1:
                # Adjust the ID list for remaining agents
                for item in range(idx + 1, len(self.n_agents + 1)):
                    self.ID[item][0] -= 1

            # Remove the corresponding entry from the ID list
            del self.ID[idx]

    
    def books(self, n):
        # Generate a demand for books in the environment
        self.book = self.env.generate_demand(n)
    
    def atualizar_flag_all_agents(self):
        observations_copy = self.observations
        for idx, observation in enumerate(observations_copy):
            # Initialize a flag list to track available actions
            flag = [0, 0, 0, 0]

            # Set the state of the environment to the current observation
            self.env.set_state(observation)

            # Get the available actions for the current observation
            available_action = self.env.available_action(observation)

            # If "4" (wait action) is in the available actions, remove it
            if 4 in available_action:
                available_action = np.setdiff1d(available_action, 4)

            # Check for other agents at possible states
            for act in available_action:
                if self.env.move(act) in self.grid_positions:
                    flag[act] = 1

            # Convert the flag to a decimal value
            flag = self.env.binary2decimal(''.join(map(str, flag)))

            # Update the flag and related attributes in the environment
            self.data[idx][1] = flag
            self.env.current_dynamic, self.env.current_flag, \
            self.env.current_drop_off, self.env.current_pick_up, \
            self.env.grid_position = self.data[idx][1:]

            # Update the observation to reflect the new state
            self.observations[idx] = self.data[idx][0] = self.env.att_state(self.data[idx][-1])



def transfer_learning_kevin(env : GridWorld, agent:brain, tl = 1):
    k = 0
    v = 0
    agent.load('qtable.txt')
    if tl == 1:
        # Transfer to reach all drop locations on the 2nd floor
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(8, len(env.drop_off)):
                    if not (pick == 2 and drop == 14):
                        aux.append(env.get_observation((0, 1, drop, pick, gp)))
            k += 1
            train_states[env.get_observation((0, 1, 14, 2, gp))] = aux
            v += len(aux)
            aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state)

    if tl == 2:
        # Transfer to reach nearby drop locations on the 1st floor
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in np.arange(0, 8):
                    if not (pick == 2 and drop == 3):
                        aux.append(env.get_observation((0, 1, drop, pick, gp)))
            k += 1
            train_states[env.get_observation((0, 1, 3, 2, gp))] = aux
            v += len(aux)
            aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state, default=.5)


    if tl == 3:
        # Transfer to reach all states of all bays on both floors
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for drop in range(len(env.drop_off)):
                for pick in [0,1,3,4]:
                    aux.append(env.get_observation((0, 1, drop, pick, gp)))
                k += 1
                train_states[env.get_observation((0, 1, drop, 2, gp))] = aux
                v += len(aux)
                aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state, default=.9)
    
    if tl == 4:
        # Transfer to facilitate learning to reach all bays
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(len(env.drop_off)-1):
                    aux.append(env.get_observation((0, 0, drop, pick, gp)))
                k += 1
                v += len(aux)
                train_states[env.get_observation((0, 1, 14, pick, gp))] = aux
                aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to_reverse(agent, state=key, state_=state)

    if tl == 5:
        # Transfer to reach the central bay
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(len(env.drop_off)):
                    if not(pick == 2 and drop == 0):
                        aux.append(env.get_observation((0, 0, drop, pick, gp)))
            k += 1
            v += len(aux)
            train_states[env.get_observation((0, 0, 0, 2, gp))] = aux
            aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state, default=.5)
        
    if tl == 6:
        # Transfer to reach any bay
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(1, len(env.drop_off)):
                    aux.append(env.get_observation((0, 0, drop, pick, gp)))
                k += 1
                v += len(aux)
                train_states[env.get_observation((0, 0, 0, pick, gp))] = aux
                aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state)

    if tl == 7:
        # Transfer to reach the central bay
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(len(env.drop_off)):
                    if not (pick == 2 and drop == 0):
                        aux.append(env.get_observation((0, 2, drop, pick, gp)))
            k += 1
            v += len(aux)
            train_states[env.get_observation((0, 0, 0, 2, gp))] = aux
            aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state)

    if tl == 8:
        # Transfer to reach the central bay
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(len(env.drop_off)):
                    if not (pick == 2 and drop == 0):
                        aux.append(env.get_observation((0, 2, drop, pick, gp)))
            k += 1
            v += len(aux)
            train_states[env.get_observation((0, 2, 0, 2, gp))] = aux
            aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state)

    if tl == 9:
        # Transfer to reach the central bay with different dynamic and flag values
        # Stage 1
        train_states = dict()
        aux = []
        for gp in env.get_possibles_grid_positions():
            for pick in range(len(env.pick_up)):
                for drop in range(len(env.drop_off)):
                    for flag in np.arange(3):
                        for dynamic in np.arange(1, 16):
                            aux.append(env.get_observation((dynamic, flag, drop, pick, gp)))
                        k += 1
                        v += len(aux)
                        train_states[env.get_observation((0, flag, drop, pick, gp))] = aux
                        aux = []

        # Transfer knowledge from the first stage
        transfer_learning = transfer()
        for key in train_states.keys():
            for state in train_states[key]:
                agent = transfer_learning.from_to(agent, state=key, state_=state)

    agent.save('qtable.txt')