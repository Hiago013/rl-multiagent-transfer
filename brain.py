import numpy as np
class brain():
    def __init__(self, epsilon, gamma, alpha, n_actions, n_states):
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.actions = np.arange(n_actions)
        self.q_table = np.zeros((n_states, n_actions))
        self.epsilonFunction = lambda episode, maxEp: (max(1 - 1/maxEp * episode, .1))
    
    def epfunction(self, max_ep, min_ep, maxEpisode):
        self.epsilonFunction = lambda episode, mx: (max(max_ep + episode * (min_ep - max_ep)/maxEpisode, .1))

    def filter_q_table(self, state_action):
        for state in state_action.keys():
            for action in self.actions:
                if action not in state_action[state]:
                    self.q_table[state][action] = -1000
    
    def colision_actions(self):
        start = 36450
        left = 3
        right = 2
        up = 1
        down = 0
        self.q_table[start: 2 * start, left] = -100 # tira esquerda
        self.q_table[2 * start: 3 * start, right] = -100 # tira direita
        self.q_table[3 * start: 4 * start, [right, left]] = -100 # tira esquerda direita
        self.q_table[4 * start: 5 * start, up] = -100 # tira cima
        self.q_table[5 * start: 6 * start, [up, left]] = -100 # tira cima e esquerda
        self.q_table[6 * start: 7 * start, [up, right]] = -100 # tira cima e direita
        self.q_table[7 * start: 8 * start, [up, right, left]] = -100 # tira cima, esquerda e direita
        self.q_table[8 * start: 9 * start, down] = -100 # tira baixo

        self.q_table[9 * start: 10 * start, [down, left]] = -100 # tira baixo
        self.q_table[10 * start: 11 * start, [down, right]] = -100 # tira baixo
        self.q_table[11 * start: 12 * start, [down, right, left]] = -100 # tira baixo
        self.q_table[12 * start: 13 * start, [down, up]] = -100 # tira baixo
        self.q_table[13 * start: 14 * start, [down, up, left]] = -100 # tira baixo
        self.q_table[14 * start: 15 * start, [down, up, right]] = -100 # tira baixo
        self.q_table[15 * start: 16 * start, [down, up, right, left]] = -100 # tira baixo

    def choose_action(self, state, episode, maxEpisode, actions):
        if np.random.uniform() < self.epsilonFunction(episode, maxEpisode):
            return np.random.choice(actions)
        best_action = self.choose_best_action(state, actions)#np.argmax(self.q_table[state,:])
        return best_action
    
    def set_qvalue(self, state, values):
        self.q_table[state] = values

    def choose_best_action(self, state, available_action = []):
        if len(available_action) == 0:
            available_action = self.actions
        best_action = available_action[0]
        max_q = self.q_table[state, best_action]
        for action in available_action:
            if self.q_table[state, action] > max_q:
                best_action = action
                max_q = self.q_table[state, action]
        #print(best_action)
        #best_action = np.argmax(self.q_table[state,:])
        return best_action
  
    def learn(self, state, action, reward, state_, done):
        self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + self.alpha * (reward + (1-done) * self.gamma * np.max(self.q_table[state_,:]))
    
    def save(self, file_name):
        np.savetxt(file_name, self.q_table)
        
    def load(self, file_name):
        self.q_table = np.loadtxt(file_name)
    
    def get_q_table(self):
        return self.q_table