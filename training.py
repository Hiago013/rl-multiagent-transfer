from multi_agent import *
import time, cv2

# Function to visualize the environment
def visualizar():
    # Get obstacle locations
    obstacle = env.obstacles
    points_obstacles = [np.array((state2cartesian(state))) for state in obstacle]

    # Get drop-off locations
    drop_off = env.drop_off
    drop_off_points = [np.array((state2cartesian(state))) for state in drop_off]

    # Get pick-up locations
    pick_up = env.pick_up
    pick_up_point = [np.array((state2cartesian(state))) for state in pick_up]

    # Get the current pick-up and drop-off locations
    current_pick_up = ma.data[0][-2]
    pick_point = np.array(state2cartesian(pick_up[current_pick_up]))
    current_drop_off = ma.data[0][-3]
    drop_point = np.array(state2cartesian(drop_off[current_drop_off]))

    # Create an empty image
    img = np.zeros((450, 900, 3), np.uint8)

    # Draw static elements (obstacles, drop-off, and pick-up locations)
    for point in points_obstacles:
        point = tuple(point.tolist())
        cv2.rectangle(img, point, (point[0] + 50, point[1] + 50), (0, 0, 255), 5)
    for point in drop_off_points:
        point = tuple(point.tolist())
        cv2.rectangle(img, point, (point[0] + 50, point[1] + 50), (0, 255, 255), 5)

    drop_point = tuple(drop_point.tolist())
    cv2.rectangle(img, drop_point, (drop_point[0] + 50, drop_point[1] + 50), (0, 255, 255), -1)
    for point in pick_up_point:
        point = tuple(point.tolist())
        cv2.rectangle(img, point, (point[0] + 50, point[1] + 50), (0, 255, 0), 5)

    pick_point = tuple(pick_point.tolist())
    cv2.rectangle(img, pick_point, (pick_point[0] + 50, pick_point[1] + 50), (0, 255, 0), -1)

    # Pause for a fixed time
    t_end = time.time()
    while time.time() < t_end + .1:
        continue

    agent_position = info['grid_position']

    # Draw agents on the image
    for idx, n_agnt in enumerate(agent_position):
        agent_state = n_agnt
        agent_point = tuple(np.array(state2cartesian(agent_state)).tolist())

        cv2.rectangle(img, agent_point, (agent_point[0] + 50, agent_point[1] + 50), [255, int(idx/2 * 255), idx*100], 3)

    # Display the image
    cv2.imshow('Grid_World', img)
    cv2.waitKey(1)
    pass

# Dictionary of stage configurations
FECHAMENTOS = {
    0: {
        0: np.arange(27, 36),
        1: np.arange(54, 63),
        2: np.arange(81, 90),
        3: np.arange(108, 117),
        4: np.arange(135, 144),
        5: np.array([]),
    },
    1: {
        0: np.arange(126, 135),
        1: np.arange(99, 108),
        2: np.arange(72, 81),
        3: np.arange(45, 54),
        4: np.arange(18, 27),
        5: np.array([]),
    }
}

# List of obstacle locations
OBSTACULOS = [19, 20, 22, 23, 26, 28, 29, 31, 32, 35, 37, 38, 40, 41, 44,
              46, 47, 49, 50, 53, 90, 91, 93, 94, 97, 98, 99, 100, 102,
              103, 106, 107, 108, 109, 111, 112, 115, 116, 117, 118, 120,
              121, 124, 125, 149, 150, 151, 152, 153, 154, 155, 156, 158, 159,
              160, 161]

# Function to update obstacle locations
def att_obstaculos(new_obstaculos: np.array):
    env.set_obstacles(OBSTACULOS + new_obstaculos.tolist())
    env.possible_states()
    env.load_available_action2()
    env.load_available_flag_dynamic2()

# Function to save runtime information
def save_runtime(time):
    with open("runtime.txt", 'a', encoding='utf-8') as f:
        f.write(f"{time}\n")

# Initialize the environment
env = GridWorld(9, 9, -1, 50, 100, 150, 1)
env.set_pick_up([2, 3, 4, 5, 6])
env.set_drop_off([18, 25, 27, 30, 34, 39, 43, 48, 110, 113, 119, 122, 133, 142, 145])
env.set_obstacles(OBSTACULOS)
env.possible_states()
env.load_available_action2()
env.load_available_flag_dynamic2()

# Initialize the agent
agent = brain(.1, .99, .2, len(env.action_space()), len(env.state_space()))

n_agents = 1
ma = multi_agent(agent, env, n_agents)

# Configuration for training stages
control_trainning = {
    0: {
        'epoch': [50, 100, 150, 200, 250, 300],
        'epsilon': .1,
        'retrain': [200, 300, 400, 500, 600, 700],
        'n_agents': 1,
        'n_books': 0,
        'max_ep': lambda x: x,
    }
}

# Main loop for training stages
init = time.time()
for all_estagios in range(0, 42):
    print('\n', all_estagios)
    env.set_stage(1)

    # Stage-specific configurations
    if all_estagios < 6:
        att_obstaculos(FECHAMENTOS[1][all_estagios % 6])
        n_epoch = control_trainning[0]['epoch'][all_estagios]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - all_estagios/10, max_ep)

    elif all_estagios == 6:
        # Transfer learning from Kevin's model for stage 6
        transfer_learning_kevin(env, agent, 1)
        ma.load('qtable.txt')

        # Updates available position that the agent can move
        att_obstaculos(FECHAMENTOS[1][2])

        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios > 6 and all_estagios < 12:
        # Configuration for stages 7 to 11
        if all_estagios % 6 < 3:
            att_obstaculos(FECHAMENTOS[1][2])
        else:
            att_obstaculos(FECHAMENTOS[1][all_estagios % 6])

        # Set parameters
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep)

    elif all_estagios >= 12 and all_estagios < 18:
        # Configuration for stages 12 to 17
        if all_estagios % 6 < 2:
            att_obstaculos(FECHAMENTOS[0][1])
        else:
            att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set parameters
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep) #np.log10(10-all_estagios)


    elif all_estagios == 18:
        # Transfer learning from Kevin's model for a different scenario at stage 18
        transfer_learning_kevin(env, agent, 2)
        ma.load('qtable.txt')

        # Updates available position that the agent can move
        att_obstaculos(FECHAMENTOS[0][2])

        # Set other parameters
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep)

    elif all_estagios > 18 and all_estagios < 24:
        # Configuration for stages 19 to 23
        if all_estagios % 6 < 3:
            att_obstaculos(FECHAMENTOS[0][2])
        else:
            att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters as in previous sections...
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios)
        ma.set_ep(.1, 1 - (all_estagios % 6)/10, max_ep)

    elif all_estagios == 24:
        # Configuration for stage 24
        env.set_stage(0)
        transfer_learning_kevin(env, agent, 3)
        transfer_learning_kevin(env, agent, 4)
        ma.load('qtable.txt')
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters as in previous sections...
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6 )
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios > 24 and all_estagios < 30:
        # Configuration for stages 25 to 29
        env.set_stage(0)
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters
        n_epoch = control_trainning[0]['epoch'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 6)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios == 30:
        # Configuration for stage 30
        env.set_stage(0)
        transfer_learning_kevin(env, agent, 5)
        ma.load('qtable')
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set parameters
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios > 30 and all_estagios < 36:
        # Configuration for stages 31 to 35
        env.set_stage(0)
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters as in previous sections...
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios == 36:
        # Configuration for stage 36
        env.set_stage(2)
        transfer_learning_kevin(env, agent, 6)
        transfer_learning_kevin(env, agent, 7)
        ma.load('qtable')
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters as in previous sections...
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios > 36 and all_estagios < 42:
        # Configuration for stages 37 to 41
        env.set_stage(2)
        att_obstaculos(FECHAMENTOS[0][all_estagios % 6])

        # Set other parameters as in previous sections...
        n_epoch = control_trainning[0]['retrain'][all_estagios % 6]
        n_books = control_trainning[0]['n_books']
        n_agents = control_trainning[0]['n_agents']
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(all_estagios % 12)
        ma.set_ep(.1, 1 - (all_estagios % 6) / 10, max_ep)

    elif all_estagios == 42:
        # Transfer learning from Kevin's model for stages 42 and 43
        transfer_learning_kevin(env, agent, 8)
        transfer_learning_kevin(env, agent, 9)
        ma.load('qtable')
        ma.colision_actions()
        ma.save('qtable')
        env.set_stage(3)
        n_epoch = 5000
        n_books = 2
        n_agents = 2
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(0)
        ma.set_ep(.2, .2, 1)

    elif all_estagios == 43:
        # Configuration for stage 43
        env.set_stage(3)
        n_epoch = 3000
        n_books = 3
        n_agents = 3
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(0)
        ma.set_ep(.2, .2, 1)

    elif all_estagios == 44:
        # Configuration for stage 44
        env.set_stage(3)
        n_epoch = 2000
        n_books = 4
        n_agents = 4
        fmax_ep = control_trainning[0]['max_ep']
        max_ep = fmax_ep(n_epoch)
        agent.epsilon = control_trainning[0]['epsilon']
        env.set_progressive_curriculum(0)
        ma.set_ep(.2, .2, 1)


    # Training loop for the current stage
    reward_sum = np.zeros((n_agents, n_epoch))
    for epoch in range(n_epoch):
        ma.set_n_agents(n_agents)
        observations = ma.reset()
        ma.books(n_books)
        done = [False, False]
        while (False in done):
            observation_, reward, done, info = ma.step_agents(epoch + 1, max_ep)
            reward_sum[0][epoch] += reward[0]
            #visualizar()
        print(epoch, end='\r')

    end = time.time() - init
    ma.save('qtable')
    plt.figure()
    plt.plot(reward_sum[0])
    plt.savefig(f'graph{all_estagios}.png', dpi=600)
    plt.close()