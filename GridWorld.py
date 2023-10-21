import numpy as np
from curriculum import curriculum
from stack import Stack
#from new_curriculum import new_curriculum
from curriculum_kevin import new_curriculum
class GridWorld:
  def __init__(self, row, col, kl, kp, kd, kg, num_agents):
    self.row = row
    self.col = col
    self.kl = kl
    self.kp = kp
    self.kd = kd
    self.kg = kg
    self.kgback = kg
    self.num_agents = num_agents
    self.actions = np.array([0, 1, 2, 3, 4])
    self.num_flags = 3
    self.flag_dynamic = 16
    self.floors = 2
    self.stage = 3

    self.last_stage = None
    

    self.state, self.action, self.reward, self.state_, self.done = (0, 0, 0, 0, 0)
    self.elevator = [74, 78, 83, 87]

    self.agents = dict()
    self.stack_states = Stack()
    for agent in range(num_agents):
      self.agents[agent] = [] # lista com state, action, reward, next_state, done
  
  def set_obstacles(self, obstacles: np.array):
    self.obstacles = obstacles

  def set_pick_up(self, pick_up : np.array):
    self.pick_up = pick_up
  
  
  def set_drop_off(self, drop_off : np.array):
    self.drop_off = drop_off
  
  def set_stage(self, stage : int):
    self.stage = stage
  
  def set_progressive_curriculum(self, value:int):
    self.progressive = value
  
  def possible_states(self):
    all_grid_position = self.row * self.col * self.floors # floors é o numero de andares
    all_pick_up = len(self.pick_up)
    all_drop_off = len(self.drop_off)
    all_flag = self.num_flags
    all_agents = self.num_agents
    flag_dynamic = self.flag_dynamic


    self.all_states = np.arange(all_grid_position**all_agents * all_pick_up\
                                * all_drop_off * all_flag * flag_dynamic)
    
    shape = [flag_dynamic, all_flag, all_drop_off, all_pick_up, all_grid_position]
    for i in range(1, all_agents):
      shape.insert(0, all_grid_position)
    shape = tuple(shape)
    

    self.all_states = self.all_states.reshape(shape)
    self.initial_states()
    self.crr = curriculum(self.all_states, self.obstacles, self.elevator, self.col, self.row)
    self.ncrr = new_curriculum(self.all_states, self.obstacles, self.elevator, self.col, self.row, self.pick_up, self.drop_off)
    
  def initial_states(self):
    shape = self.all_states.shape
    n_axis = len(shape)
    states = np.copy(self.all_states)
    if n_axis > 5:
      del_axis = [i for i in range(n_axis) if n_axis-i > 5]
      del_axis.append(n_axis - 1)
      del_axis = tuple(del_axis)

    del_pick_up = tuple([i for i in range(shape[-1]) if i not in self.pick_up])
    states = np.delete(states, del_pick_up, axis=n_axis - 1)
    states = np.delete(states, (1, 2), axis=n_axis - 4)
    self.start_state = states.flatten()

  def reset(self):
    #if len(self.stack_states) == 0:
    #  for item in self.ncrr.get_stage(self.stage)[self.progressive]:
    #    self.stack_states.push(item)
  # # #if self.stage >= 3:
  # # #    self.state = np.random.choice(self.ncrr.get_stage(3))
  # # #else:
    self.state = np.random.choice(self.ncrr.get_stage(self.stage)[self.progressive]) # falta arrumar esse
    self.current_dynamic, self.current_flag, self.current_drop_off, self.current_pick_up, \
    self.grid_position = np.array(np.where(self.state == self.all_states)).squeeze()

    if self.last_stage != self.stage:
      self.last_stage = self.stage
      self.aux = np.array(list(map(self.get_states, self.ncrr.get_stage(self.stage)[self.progressive])))
    return self.state
    # Antigo Curriculum
    #if self.stage == 5:
    #    self.state = np.random.choice(self.crr.get_stage(0))
    #else:
    #  self.state = np.random.choice(self.crr.get_stage(self.stage))
    #self.current_dynamic, self.current_flag, self.current_drop_off, self.current_pick_up, \
    #self.grid_position = np.array(np.where(self.state == self.all_states)).squeeze()
    #return self.state

  def move(self, action):
    grid_position_ = self.grid_position
    if action == 0: #baixo
        grid_position_ += self.col
    elif action == 1: # cima
        grid_position_ -= self.col
    elif action == 2 and grid_position_ % self.col != self.col - 1: #direita
        grid_position_ += 1
    elif action == 3 and grid_position_ % self.col != 0: # esquerda
        grid_position_ -= 1
    elif action == 4: # parado
        pass
    if not self.on_map(grid_position_):
      return self.grid_position
    if self.on_obstacle(grid_position_):
      return self.grid_position
    #if self.filter_move(grid_position_):
    #  return self.grid_position
    return grid_position_
  
  def on_map(self, grid_position):
    if grid_position < 0 or grid_position >= self.row * self.col * self.floors:
      return False
    return True
  
  def on_obstacle(self, grid_position):
    if grid_position in self.obstacles:
      return True
    return False
  
  def on_goal(self, grid_position):
    if self.stage == 0:
      self.kg = self.kp
      if self.current_flag == 1 or \
         (self.current_flag == 0 and grid_position == self.pick_up[self.current_pick_up]):
         self.kg = self.kp
         return True
      return False

    #if self.stage == 1:
    #  return self.on_elevator2(grid_position)
    
    if self.stage == 1:
    #   if self.progressive > 5  and self.progressive < 9:
    #     self.kg = 0
    #     return self.on_elevator2(grid_position)

      if self.current_flag == 2 or grid_position == self.drop_off[self.current_drop_off]:
        self.kg = self.kd
        return True
      return False


    if self.current_flag == 2 and grid_position in self.pick_up:
      self.kg = self.kgback
      return True
    return False


    # Curriculum antigo
    #if self.stage == 0:
    #  if self.current_flag == 1 or \
    #     self.current_flag == 0 and grid_position == self.pick_up[self.current_pick_up] :
    #    self.kg = self.kp
    #    return True
    #  return False
   # 
   # elif self.stage == 1:
   #   if grid_position in self.elevator[0:2]:
   #     self.kg = self.kp
   #     return True
   #   return False
   # 
   # elif self.stage == 2:
   #   if self.current_flag == 2:# and grid_position == self.drop_off[self.current_drop_off]:
   #     self.kg = self.kd
   #     return True
   #   return False
   # 
   # elif self.stage == 3:
   #   if self.current_flag == 2 and grid_position in self.elevator[2:]:
   #     self.kg = self.kd
   #     return True
   #   return False
   # 
   # if self.current_flag == 2 and grid_position in self.pick_up:
   #     self.kg = self.kgback
   #     return True
   # return False

  def on_elevator2(self, grid_position):
    if grid_position in self.elevator[2:]:
      return True
    return False
  
  def on_elevator3(self, grid_position):
    if grid_position in self.elevator[:2]:
      return True
    return False

  def on_drop_off(self, grid_position):
    if self.current_flag == 1 and grid_position == self.drop_off[self.current_drop_off]:
      return True
    return False
  
  def on_pick_up(self, grid_position):
    if self.current_flag == 0 and grid_position == self.pick_up[self.current_pick_up]:
      return True
    return False
  
  def on_terminate(self, grid_position):
    if self.current_flag == 2 and grid_position in self.pick_up:
      return True
    return False
  
  def on_done(self, grid_position):
    if self.on_goal(grid_position):
      return True
    #if self.on_dynamic(self.action):
    # return True
    return False

  def on_elevator(self, grid_position):
    if grid_position in self.elevator:
      return True
    return False
  
  def att_flag(self, grid_position):
    if self.current_flag == 0 and grid_position == self.pick_up[self.current_pick_up]:
      self.current_flag = 1

    if self.current_flag == 1 and grid_position == self.drop_off[self.current_drop_off]:
      self.current_flag = 2
  
  def att_dynamic(self, state):
    if state >= self.col * self.row * 2:
      state = self.what_position(state)
    self.current_dynamic = np.random.choice(self.state_dynamic_flag[state])
    return self.att_state(self.grid_position_)
    
  def step(self, action):
    self.action = action
    self.grid_position_ = self.move(action=action)
    self.reward = self.get_reward(self.grid_position, self.grid_position_)
    self.done = self.on_done(self.grid_position_)
    self.att_flag(self.grid_position_)
    self.state_ = self.att_state(self.grid_position_)
    self.grid_position = self.grid_position_
    self.state = self.state_
    return self.state_, self.reward, self.done
  
  def action_space(self):
    return self.actions
  
  def state_space(self):
    return self.all_states.flatten()
  
  def att_state(self, grid_position):
    return self.all_states[self.current_dynamic, self.current_flag, \
                           self.current_drop_off, self.current_pick_up, grid_position]
  
  def on_dynamic(self, action):
    binary_flag_dynamic = self.decimal2binary(self.current_dynamic)
    if action == 0 and binary_flag_dynamic[0] == '1':
      return True
    if action == 1 and binary_flag_dynamic[1] == '1':
      return True
    if action == 2 and binary_flag_dynamic[2] == '1':
      return True
    if action == 3 and binary_flag_dynamic[3] == '1':
      return True
    return False
  
  def set_state(self, observation):
    self.current_dynamic, self.current_flag, \
    self.current_drop_off, self.current_pick_up, self.grid_position = self.get_states(observation)

  def get_position(self):
    return self.grid_position
  
  def get_states(self, observation):
    return np.array(np.where(observation == self.all_states)).squeeze()
  
  def get_possibles_grid_positions(self):
    return np.delete(np.arange(self.col * self.row * 2), self.obstacles)
  
  def get_observation(self, states : tuple):
    return self.all_states[states]
  
  def what_position(self, state):
    dynamic, flag, drop, pick, gp = np.array(np.where(state == self.all_states)).squeeze()
    return gp

  def get_reward(self, state, state_):
    reward = self.kl
    if self.on_dynamic(self.action):
      reward -= self.kg
      if self.current_flag == 2:
        reward -= self.kg // 2

    if self.on_drop_off(state_):
      reward += self.kd

    if self.on_pick_up(state_):
      reward += self.kp
    
    if self.on_terminate(state_):
      reward += self.kgback

    #if self.on_goal(state_):
    #  reward += self.kg
    #  if self.action == 4:
    #    reward += self.kg // 4
        
    if self.action == 4:
      reward -= 6
    
    
   # if self.action == 2 and self.current_flag == 0:
   #   reward -= 1
    
   # if self.action == 3 and self.current_flag == 2:
   #   reward -= 1
    
   # if self.action == 0 and (self.current_flag == 2 or self.current_flag == 0):
   #   reward -= 1
    
    if self.current_flag == 0 :
      position_goal = self.pick_up[self.current_pick_up]
      xg, yg = divmod(position_goal, self.row)
      
      position_agent = self.what_position(state_)
      xa, ya = divmod(position_agent, self.row)

      distance = np.abs(xg - xa) + np.abs(yg - ya)
      reward -= distance / (2 * (2 * self.row + self.row ))

    elif self.current_flag == 1:
      position_goal = self.drop_off[self.current_drop_off]
      xg, yg = divmod(position_goal, self.row)
      
      position_agent = self.what_position(state_)
      xa, ya = divmod(position_agent, self.row)

      distance = np.abs(xg - xa) + np.abs(yg - ya)
      reward -= distance / (2 * (2 * self.row + self.row ))
    
    else:
      position_agent = self.what_position(state_)
      min_dist = 100

      for pick_up in self.pick_up:
        position_goal = pick_up
        xg, yg = divmod(position_goal, self.row)
        position_agent = self.what_position(state_)
        xa, ya = divmod(position_agent, self.row)
        distance = np.abs(xg - xa) + np.abs(yg - ya)
        if distance < min_dist:
          min_dist = distance
        distance = min_dist

      reward -= distance / (2 * (2 * self.row + self.row ))

    return reward
  
  def decimal2binary(self, decimal):
    binary = bin(decimal).replace("0b", "")
    while len(binary) < 4:
      binary = '0' + binary
    return binary

  def binary2decimal(self, binary):
    return int(binary, 2)
  
  def load_available_flag_dynamic(self):
    self.state_dynamic_flag = dict()
    states = np.delete(self.all_states, self.obstacles, axis=4)
    for state in states.flatten():
      possible_actions = self.available_action(state)
      possible_actions = possible_actions[:-1]
      aux_possibles = []
      for action in self.actions[:-1]:
        if action in possible_actions:
          aux_possibles.append(2)
        else:
          aux_possibles.append(1)
      flags_dynamic = []
      for down in range(aux_possibles[0]):
        for up in range(aux_possibles[1]):
          for right in range(aux_possibles[2]):
            for left in range(aux_possibles[3]):
              decimal = self.binary2decimal(str(down) + str(up) + str(right) + str(left))
              flags_dynamic.append(decimal)
      self.state_dynamic_flag[state] = np.array(flags_dynamic)

  
  def load_available_action(self):
    self.state_action = dict()
    states = np.delete(self.all_states, self.obstacles, axis=4)
    for state in states.flatten():
      data = np.array(np.where(state == self.all_states)).squeeze()
      grid_position = data[-1]
      aux = []

      if (grid_position < self.col*self.row):
        if (((grid_position + self.col) < self.col*self.row) or grid_position == 74 or grid_position == 78) and \
          (grid_position + self.col) not in self.obstacles:
          aux.append(0)
        
        if ((grid_position - self.col) >= 0) and \
         (grid_position - self.col) not in self.obstacles:
          aux.append(1)

      else:
        if ((grid_position + self.col) < self.col*self.row*self.floors) and \
         (grid_position + self.col) not in self.obstacles:
          aux.append(0)

        if (((grid_position - self.col) >= self.row*self.col) or grid_position == 83 or grid_position == 87) and \
         (grid_position - self.col) not in self.obstacles:
          aux.append(1)
          

      if ((grid_position % self.col != self.col - 1)) and \
         (grid_position + 1) not in self.obstacles:
          aux.append(2)

      if ((grid_position % self.col != 0)) and \
         (grid_position - 1) not in self.obstacles:
          aux.append(3)

      aux.append(4)

      self.state_action[state] = np.array(aux)
  
  def available_action(self, state):
    state = self.what_position(state)
    return self.state_action[state]

  def available_action2(self, state):
    state = self.what_position(state)
    return self.state_action[state]

  def load_available_flag_dynamic2(self):
    self.state_dynamic_flag = dict()
    states = np.arange(self.col * self.row * 2)
    for state in states.flatten():
      possible_actions = self.available_action2(state)
      possible_actions = possible_actions[:-1]
      aux_possibles = []
      for action in self.actions[:-1]:
        if action in possible_actions:
          aux_possibles.append(2)
        else:
          aux_possibles.append(1)
      flags_dynamic = []
      for down in range(aux_possibles[0]):
        for up in range(aux_possibles[1]):
          for right in range(aux_possibles[2]):
            for left in range(aux_possibles[3]):
              decimal = self.binary2decimal(str(down) + str(up) + str(right) + str(left))
              flags_dynamic.append(decimal)
      self.state_dynamic_flag[state] = np.array(flags_dynamic)

  def load_available_action2(self):
    self.state_action = dict()
    states = np.arange(self.row * self.col * 2)
    for state in states:
      grid_position = state
      aux = []

      if (grid_position < self.col*self.row):
        if (((grid_position + self.col) < self.col*self.row) or grid_position == 74 or grid_position == 78) and \
          (grid_position + self.col) not in self.obstacles:
          aux.append(0)
        
        if ((grid_position - self.col) >= 0) and \
         (grid_position - self.col) not in self.obstacles:
          aux.append(1)

      else:
        if ((grid_position + self.col) < self.col*self.row*self.floors) and \
         (grid_position + self.col) not in self.obstacles:
          aux.append(0)

        if (((grid_position - self.col) >= self.row*self.col) or grid_position == 83 or grid_position == 87) and \
         (grid_position - self.col) not in self.obstacles:
          aux.append(1)
          

      if ((grid_position % self.col != self.col - 1)) and \
         (grid_position + 1) not in self.obstacles:
          aux.append(2)

      if ((grid_position % self.col != 0)) and \
         (grid_position - 1) not in self.obstacles:
          aux.append(3)

      aux.append(4)

      self.state_action[state] = np.array(aux)
  
  def generate_demand(self, n):
    stack_books = Stack()
    for i in range(n):
      state = np.random.choice(self.ncrr.get_stage(self.stage)[self.progressive])
      dynamic, flag, drop, pick, gp = self.get_states(state)

      #pick_up = np.random.choice(np.arange(len(self.pick_up)))
      #drop_off = np.random.choice(np.arange(len(self.drop_off)))
      stack_books.push([drop, pick])
    return stack_books
  
  def filter_move(self, grid_position):
    if grid_position in np.arange(np.max(self.aux[:,4])):
      return False
    return True

      

    
