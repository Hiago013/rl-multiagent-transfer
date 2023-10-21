import numpy as np
from stack import Stack

class new_curriculum:
    def __init__(self, states:np.array, obstacles:list, elevator, col:int, row:int, pick_up:list, drop_off:list):
        self.states = states
        self.obstacles = obstacles
        self.col = col
        self.row = row
        self.axis_grid_position, self.axis_pick_up,\
        self.axis_drop_off, self.axis_flag, self.axis_dynamic = 4, 3, 2, 1, 0
        self.elevator = elevator
        self.stage = {}
        self.drop_off = drop_off
        self.pick_up = pick_up
        self.load_stages()

    
    def load_stages(self):
        all_grid_positions = np.arange(self.col * self.row * 2)
        # Primeiro Estagio
        # Vale apenas (0, 1, 14, 2, grid_position)
        # Aprender a levar até a última estante em qualquer posição do mapa
        states = np.array_split(all_grid_positions[::-1], 6)
        stages_aux = []
        del_drop_off = np.arange(len(self.drop_off) - 1) # Entregar apenas no ultimo
        del_dynamic = np.arange(1,16) # considerar que não há nenhum outro robo
        del_flag = [0, 2] # Flag 1, ou seja, entregar
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        for idx, item in enumerate(states): # Entregar 2 andar progressivo
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx, aux.flatten()))

        # Primeiro Estagio
        # Vale apenas (0, 1, [8-14], [2], grid_position)
        # Aprender a levar até a última estante em qualquer posição do mapa
        del_drop_off = np.arange(0, 8) # Entregar apenas no 2 andar
        del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        del_flag = [0, 2] # flag 1, ou seja, entregar
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        for idx2, item in enumerate(states): # Entregar 2 andar progressivo com
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx + idx2 + 1, aux.flatten()))

        
        # Primeiro Estagio
        # Vale apenas (0, 1, 0, 2, grid_position)
        # Aprender a levar até a última estante em qualquer posição do mapa
        del_drop_off = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14] #np.arange(1,len(self.drop_off)) # Entregar apenas no primeiro
        del_dynamic = np.arange(1,16)                  # considerar que não há nenhum outro robo
        del_flag = [0, 2]                              # Flag 1, ou seja, entregar
        del_pick = [0, 1, 3, 4]                        # Saindo apenas da baia central
        for idx3, item in enumerate(states[::-1]):           # Entregar 2 andar progressivo
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx + idx2 + idx3 + 2, aux.flatten()))
        
        # Primeiro Estagio
        # Vale apenas (0, 1, [0-7], [2], grid_position)
        # Aprender a levar até a última estante em qualquer posição do mapa
        del_drop_off = np.arange(8, len(self.drop_off)) # Entregar apenas no 2 andar
        del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        del_flag = [0, 2] # flag 1, ou seja, entregar
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        for idx4, item in enumerate(states[::-1]): # Entregar 2 andar progressivo com
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx + idx2 + idx3 + idx4 + 3, aux.flatten()))

        
        del_grid_position = np.setdiff1d(all_grid_positions, [0,1,2,3,4])
        aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
        aux = np.delete(aux, del_flag, axis = self.axis_flag)
        aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
        stages_aux.append((idx + idx2 + idx3 + idx4 + 4, aux.flatten()))

        self.stage[1] = dict(stages_aux)

        ### Estagio de sair de qualquer lugar e ir até a baia
        stages_aux.clear()
        del_flag = [1, 2] # Flag 1, ou seja, entregar
        del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        del_drop_off = np.arange(1, len(self.drop_off))
        for idx, item in enumerate(states[::-1]):
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx, aux.flatten()))

        del_flag = [1, 2] # Flag 1, ou seja, entregar
        del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        del_drop_off = np.arange(1, len(self.drop_off))
        for idx2, item in enumerate(states[::-1]):
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            stages_aux.append((idx + idx2 + 1, aux.flatten()))

        
        self.stage[0] = dict(stages_aux)

        # ### Estagio de voltar até a baia
        # stages_aux.clear()
        # del_flag = [0, 1] # Flag 2, ou seja, voltar
        # del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        # del_grid_position = self.obstacles
        # aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
        # aux = np.delete(aux, del_flag, axis = self.axis_flag)
        # aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
        # stages_aux.append((0, aux.flatten()))

        # self.stage[2] = dict(stages_aux)
        
        ### Estagio de sair de qualquer lugar e ir até a baia
        stages_aux.clear()
        del_flag = [0, 1] # Flag 2, ou seja, voltar (finalizar)
        del_dynamic = np.arange(1,16) # Não há nenhum outro robo
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        del_drop_off = np.arange(1, len(self.drop_off))
        for idx, item in enumerate(states[::-1]):
            del_grid_position = list(set(self.obstacles + list(np.setdiff1d(all_grid_positions, item))))
            aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
            aux = np.delete(aux, del_flag, axis = self.axis_flag)
            aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
            aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
            aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
            stages_aux.append((idx, aux.flatten()))
        
        self.stage[2] = dict(stages_aux)

        ### Treino multi-agentes
        stages_aux.clear()
        del_flag = [2, 1] # Flag 2, ou seja, voltar
        del_dynamic = np.arange(1, 16) # Não há nenhum outro robo
        del_grid_position = self.obstacles
        del_pick = [0, 1, 3, 4] # Saindo apenas da baia central
        del_drop_off = np.arange(1, len(self.drop_off))
        aux = np.delete(self.states, del_grid_position, axis=self.axis_grid_position)
        aux = np.delete(aux, del_flag, axis = self.axis_flag)
        aux = np.delete(aux, del_dynamic, axis=self.axis_dynamic)
        #aux = np.delete(aux, del_drop_off, axis=self.axis_drop_off)  # posso tirar depois
        #aux = np.delete(aux, del_pick, axis = self.axis_pick_up) # posso tirar depois
        stages_aux.append((0, aux.flatten()))

        self.stage[3] = dict(stages_aux)

    
    def get_stage(self, stage):
        return self.stage[stage]
        
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from brain import brain
    from GridWorld import GridWorld

    env = GridWorld(9, 9, -1, 50, 100,150,1)
    env.set_pick_up([2, 3, 4, 5, 6])
    env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
    env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
                       46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
                       103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
                       121,124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
                       160, 161])
    env.possible_states()
    #env.load_available_action2()
    #env.load_available_flag_dynamic2()

    #agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
    #agent.load('qtable.txt')

    crr = new_curriculum(env.all_states, env.obstacles, env.elevator, 9, 9, env.pick_up, env.drop_off)

    #print('ok')
    print(len((crr.stage[0][0])))
    soma = 0
    for i in range(12):
        soma += len(crr.stage[0][i])
    print(soma)
    #a = np.array(list(map(env.get_states, crr.stage[3][0])))
    #print((a[:,4]))

    #for item in crr.stage[3][0]:
    #    print(env.get_states(item))
    #    print('')
    #    for state in crr.stage[1][key]:
    #        print(env.get_states(state))
    #    break
    #    print(' ')

   # print(((crr.stage[0])))