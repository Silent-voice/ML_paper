# -*- coding: utf-8 -*-
from maze_env import Maze
from DQN import DeepQNetwork
# from DQN_modified import DeepQNetwork


def run_maze():
    step = 0

    for episode in range(300):
        # 初始化状态
        state = env.reset()

        while True:
            # fresh env
            env.render()

            # 根据当前状态s选取行动
            action = RL.choose_action(state)

            # 根据选取的行动更新状态，并计算reward和是否结束done
            next_state, reward, done = env.step(action)

            # 记录记忆
            RL.store_transition(state, action, reward, next_state)

            # 200步之后，每5步学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap state
            next_state = state

            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,             #
                      replace_target_iter=200,  # 每200训练次更新一次target_net
                      memory_size=2000,         # 记忆池上限
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()