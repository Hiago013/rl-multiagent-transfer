import numpy as np
import cv2
import time
from brain import brain
from GridWorld import GridWorld
from multi_agent import multi_agent
from tqdm import tqdm
from logger import logger

def state2cartesian(state):
    x, y = divmod(state, 9)
    return x * 50, y * 50

def cartesian2state(cartesian_point):
    x, y = cartesian_point
    x = x // 50
    y = y // 50
    return 9 * x + y

def get_informations(state, env, m_agents):
    _, flag, drop, pick, position = env.get_states(state)
    zp = zd = zsb = 0
    xp, yp = divmod(position, env.row)
    xd, yd = divmod(env.drop_off[drop], env.row)
    xsb, ysb = divmod(env.pick_up[pick], env.row)
    if xp > 8:
        xp = 8 - (xp - 9)
        zp = 1
    if xd > 8:
        xd = 8 - (xd - 9)
        zd = 1
    if xsb > 8:
        xsb = 8 - (xsb - 9)
        zsb = 1

    available_actions = env.available_action(state)
    action = m_agents.main_agent.choose_best_action(state, available_actions)

    return [xp, yp, zp, xsb, ysb, zsb, xd, yd, zd, action, flag]

def create_txtobstacles(env:GridWorld):
    obstacles = env.obstacles
    list_obs = []
    for obs in obstacles:
        x, y = divmod(obs, env.row)
        z = 0
        if x > 8:
            x = 8 - (x - 9)
            z = 1
        list_obs.append([str(x), str(y), str(z)])

    with open('obstaculos.txt', 'w') as f:
        for line in list_obs:
            f.write(' '.join(line))
            f.write('\n')

def create_txtbaias(env:GridWorld):
    baias = env.pick_up
    list_obs = []
    for obs in baias:
        x, y = divmod(obs, env.row)
        z = 0
        if x > 8:
            x = 8 - (x - 9)
            z = 1
        list_obs.append([str(x), str(y), str(z)])

    with open('baias.txt', 'w') as f:
        for line in list_obs:
            f.write(' '.join(line))
            f.write('\n')

def create_txtentrega(env:GridWorld):
    drop_off = env.drop_off
    list_obs = []
    for obs in drop_off:
        x, y = divmod(obs, env.row)
        z = 0
        if x > 8:
            x = 8 - (x - 9)
            z = 1
        list_obs.append([str(x), str(y), str(z)])

    with open('entrega.txt', 'w') as f:
        for line in list_obs:
            f.write(' '.join(line))
            f.write('\n')

def create_txtelevador(env:GridWorld):
    drop_off = env.elevator
    list_obs = []
    for obs in drop_off:
        x, y = divmod(obs, env.row)
        z = 0
        if x > 8:
            x = 8 - (x - 9)
            z = 1
        list_obs.append([str(x), str(y), str(z)])

    with open('elevador.txt', 'w') as f:
        for line in list_obs:
            f.write(' '.join(line))
            f.write('\n')

#####

def visualizar():
    obstacle = env.obstacles
    points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]
    drop_off = env.drop_off
    drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]
    pick_up = env.pick_up
    pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]

    elevador = [74, 78, 83, 87]
    elevador_point = [np.array((state2cartesian(state))) for state in elevador]

    current_pick_up = [ma.data[i][-2] for i in range(len(agent_position))]
    pick_point = [np.array(state2cartesian(pick_up[n_current_pick_up])) for n_current_pick_up in current_pick_up]
    current_drop_off = [ma.data[i][-3] for i in range(len(agent_position))]
    drop_point = [np.array(state2cartesian(drop_off[n_current_drop_off])) for n_current_drop_off in current_drop_off]
    #img = np.zeros((450, 900, 3), dtype='uint8')


    img = np.zeros((450, 900, 3), dtype='uint8')
##      # Desenhar elementos estaticos
    for point in points_obstacles:
        cv2.rectangle(img, (point[0], point[1]), (point[0]+50, point[1]+50), (0, 0, 255), 5)
    for point in drop_off_points :
        cv2.rectangle(img, (point[0], point[1]), (point[0]+50, point[1]+50), (0, 255, 255), 5)
    for item in drop_point:
        cv2.rectangle(img, (item[0], item[1]), (item[0]+50, item[1]+50), (0, 255, 255), -1)
    for point in pick_up_point:
        cv2.rectangle(img, (point[0], point[1]), (point[0]+50, point[1]+50), (0, 255, 0), 5)
    for item in pick_point:
        cv2.rectangle(img, (item[0], item[1]), (item[0]+50, item[1]+50), (0, 255, 0), -1)
    for point in elevador_point:
        cv2.rectangle(img, (point[0], point[1]), (point[0]+50, point[1]+50), (139, 61, 72), 5)
     #Takes step after fixed time
    t_end = time.time() + .1
    while time.time() < t_end:
        continue

    for idx, n_agnt in enumerate(agent_position):
        agent_state = n_agnt
        agent_point = np.array(state2cartesian(agent_state))
        cv2.rectangle(img, (agent_point[0], agent_point[1]), (agent_point[0]+50, agent_point[1]+50), [255, int(idx/2 * 255), idx*100], 3)

    cv2.imshow('Grid_World', img)
    cv2.waitKey(1)
    #cv2.imwrite(f"nova_imagem{step}.jpg", img)
    pass

def save_data(data : list, n_agents = 1):
    if n_agents == 1:
        text = 'single'
    elif n_agents == 2:
        text = 'double'
    elif n_agents == 3:
        text = 'triple'
    elif n_agents == 4:
        text = 'quadruple'
    elif n_agents == 5:
        text = '5uple'
    else:
        text = str(n_agents)+'uple'
    try:
        load = np.loadtxt("data_" + text +"_agent.txt")
        load = list(load.flatten())
        load += data
        data = load
    except:
        pass
    data = np.array(data).reshape(-1, 8 * n_agents)
    np.savetxt("data_" + text +"_agent.txt", data)




env = GridWorld(9, 9, -1, 50, 100, 150, 1)
env.set_pick_up([2, 3, 4, 5, 6])
env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
env.set_obstacles([19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44, \
                    46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102, \
                    103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120, \
                    121,124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,\
                    160, 161])
env.possible_states()
env.load_available_action2()
env.load_available_flag_dynamic2()
agent = brain(.1, .99, .1, len(env.action_space()), len(env.state_space()))
#agent.filter_q_table(env.state_action)
agent.load('qtable.txt')
env.set_stage(3)
env.set_progressive_curriculum(0)
obstacle = env.obstacles
points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

drop_off = env.drop_off
drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

pick_up = env.pick_up
pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]

create_txtobstacles(env)
create_txtbaias(env)
create_txtentrega(env)
create_txtelevador(env)
#observation = env.reset()
n_agents = 2
ma = multi_agent(agent, env, n_agents)
color_agents = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(n_agents)]
agent_position = [1, 2]

log = logger(n_agents + 1)

traffic_jam = 0
for w in tqdm( range(0, 100), desc='Test epochs...' ):
    step = 0
    reward = [10]
    ma.set_n_agents(n_agents)
    ma.books(10 - n_agents)
    print(ma.book)
    print(ma.book.get_all())
    observations = ma.reset()
    service_bays = [0, 0, 0, 0, 0]                   #
    for bays in range(5):                            #
        for _, ser_bay_idx in ma.book.get_all():     #
            service_bays[ser_bay_idx] = 1            #
    done = [False]
    aux = []
    while True:

        if not(False in done):
            break

        if len(set(agent_position)) < len(agent_position): # if true agent did a collision
            while True:
                pass
        for value in reward:
            if value < -50:# if true agent did a collision
                while True:
                    pass

        try:
            flag_falha = 0
            flaag = 0
            aux = get_informations(observations[0], env, ma)
            aux = [int(valor) for valor in aux]

            flaag = 1
            aux2 = get_informations(observations[1], env, ma)
            aux2 = [int(valor) for valor in aux2]
        except:
            flag_falha = 1
            aux = get_informations(observations, env, ma)
            aux = [int(valor) for valor in aux]

        observations, agent_position, reward, done = ma.step()

        service_bays = [0, 0, 0, 0, 0]                   #
        for bays in range(5):                            #
            for _, ser_bay_idx in ma.book.get_all():     #
                service_bays[ser_bay_idx] = 1            #
        #service_bays[ma.env.current_pick_up] = 2

        log.update(0, service_bays)
        if flag_falha == 1:
            if flaag == 1:
                log.update(1, aux)
            else:
                log.update(2, aux)
        else:
            log.update(1, aux)
            log.update(2, aux2)

        visualizar()
        step += 1
        if step > 500:
            traffic_jam += 1
            break
    log.save()
    metricas = [list(ma.metrics[i].values()) for i in range(n_agents)]
    metricas = np.array(metricas).flatten()
    metricas = metricas.tolist()
    save_data(metricas, n_agents)

print(traffic_jam)

cv2.destroyAllWindows()

# 1 agentes -> 0 falhas
# 2 agentes -> 0 falhas
# 3 agentes -> 1 falhas
# 4 agentes -> 1 falhas
# 5 agentes -> 1 falhas
# 6 agentes -> 4 falhas
# 7 agentes -> 5 falhas
# 8 agentes -> 3 falhas
# 9 agentes -> 6 falhas

## --- novo
# 3 agentes -> 0
