import connect4_env

env = connect4_env.Connect4Env(6)

action_red = [1, 1, 1, 1, 1]# [0, 1, 2, 2, 3, 4, 5, 3] #

action_blue = [2, 3, 4, 3, 3] # [1, 2, 3, 3, 4, 5, 5, 4] #


for idx in range(len(action_blue)):
    a_action_red = {"player_name": "red", "value": action_red[idx]}
    state, reward, done = env.step(a_action_red)
    print(state.reshape(6, 6))
    print(done, reward)
    a_action_blue = {"player_name": "blue", "value": action_blue[idx]}
    state, reward, done = env.step(a_action_blue)


