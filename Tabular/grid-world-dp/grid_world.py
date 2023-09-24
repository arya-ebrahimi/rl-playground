import matplotlib.pyplot as plt
import numpy as np
from matplotlib.table import Table

C = 100 #constat to add to reward
WORLD_SIZE = 6
GOAL = [2, 5]
DISCOUNT = 0.9
GOAL_REWARD = +4.0+C
RED_REWARD = -4.0+C
DEFAULT_REWARD = -1+C

ACTIONS = [np.array([-1, 1]),
           np.array([1, 1])]

ACTIONS_BAHREVAR = [np.array([1, 0]), 
                  np.array([0, 1])]

UP = np.array([-1, 0])
DOWN = np.array([1, 0])

OPTIMAL_TRANSITION = np.zeros((WORLD_SIZE, WORLD_SIZE, 2))
RANDOM_TRANSITION = np.ones((WORLD_SIZE, WORLD_SIZE, 2))/2

red = []
for i in range(WORLD_SIZE):
    red.append([0, i])
    red.append([WORLD_SIZE-1, i])
red = red + [[1, 2], [1, 3], [4, 1], [4, 4]]


def transition(state, action, bahrevar=False):
    
    if state == GOAL:
        return state, GOAL_REWARD, True
    if state in red:
        return state, RED_REWARD, True
    
    next_state = (np.array(state) + action).tolist()
    x, y = next_state
    reward = DEFAULT_REWARD
    if y >= WORLD_SIZE:
        if bahrevar:
            next_state = state + DOWN
        else:
            next_state = state + UP

    if x < 0 or x >= WORLD_SIZE or y < 0:
        next_state = state

    return next_state, reward, False


def draw_image(image):
    fig, ax = plt.subplots()
    ax.set_axis_off()
    tb = Table(ax, bbox=[0, 0, 1, 1])

    nrows, ncols = image.shape
    width, height = 1.0 / ncols, 1.0 / nrows
    for (i, j), val in np.ndenumerate(image):
        if [i, j] == GOAL:
            val = str(val) + " (GOAL)"
        
        tb.add_cell(i, j, width, height, text=val,
                    loc='center', facecolor='white')
    for i in range(len(image)):
        tb.add_cell(i, -1, width, height, text=i+1, loc='right',
                    edgecolor='none', facecolor='none')
        tb.add_cell(-1, i, width, height/2, text=i+1, loc='center',
                    edgecolor='none', facecolor='none')

    ax.add_table(tb)

def value_evaluation(p=RANDOM_TRANSITION, bahrevar=False):
    value = np.zeros((WORLD_SIZE, WORLD_SIZE))
    for _ in range(50):
        new_value = np.zeros_like(value)
        for i in range(WORLD_SIZE):
            for j in range(WORLD_SIZE):
                values = []
                if bahrevar:
                    actions = ACTIONS_BAHREVAR
                else:
                    actions = ACTIONS
                    
                for index, action in enumerate(actions):
                    (next_i, next_j), reward, terminal = transition([i, j], action)
                    if terminal:
                        values.append(p[i, j, index]*(reward))
                    else:
                        values.append(p[i, j, index]*(reward + DISCOUNT * value[next_i, next_j]))
                
                new_value[i, j] = np.sum(values)
        value = new_value
    return value

def greedify_policy(value, bahrevar=False):
    pi_star = np.zeros((WORLD_SIZE, WORLD_SIZE, 2))
    
    for (i, j), val in np.ndenumerate(value):
        next_vals=[]
        if bahrevar:
            actions = ACTIONS_BAHREVAR
        else:
            actions = ACTIONS
        for action in actions:
            next_state, _, _= transition([i, j], action, bahrevar=bahrevar)
            next_vals.append(value[next_state[0],next_state[1]])

        best_actions=np.argmax(next_vals)
        pi_star[i, j, best_actions] = 1
    
    return pi_star
    

if __name__ == '__main__':

    value = value_evaluation(bahrevar=True)

    for i in range(5):
        pi_star = greedify_policy(value, bahrevar=True)
        value = value_evaluation(p=pi_star, bahrevar=True)

        
    draw_image(np.round(value, decimals=2))
    plt.savefig('figure_mdp_reward=' + str(DEFAULT_REWARD) +'.png')
    plt.close()   