import random
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import load_model, Sequential, model_from_json
from keras.optimizers import Adam


def get_epsilon(epsilon, epsilon_min=0.01, epsilon_decay=0.999):
    if epsilon < epsilon_min:
        return epsilon_min
    else:
        epsilon *= epsilon_decay
        return epsilon


def get_action(model, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(model.output_shape[1])  # model.output_shape[1] = action_size
    else:
        q = model.predict(state)
        return np.argmax(q[0])


def bellmann_eq(model, model_target, states, actions, states_next, rewards, dones, batch_size, discount=0.99):
    q = model.predict(states)
    q_target = model_target.predict(states_next)
    for j in range(batch_size):
        if dones[j]:
            q[j][actions[j]] = rewards[j]
        else:
            q[j][actions[j]] = rewards[j] + discount * np.amax(q_target[j])
    return q


def copy_model(model):
    model_copied = model_from_json(model.to_json())
    model_copied.set_weights(model.get_weights())
    return model_copied


def train_model_DQN(model, env, learning_rate=0.001, max_episodes=300, max_memory=2000, train_start=1000, batch_size=64, render=False):
    #
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    state_size = model.input_shape[1]
    action_size = model.output_shape[1]
    model_target = copy_model(model)

    # memory 할당
    memory = deque(maxlen=max_memory)
    states = np.zeros((batch_size, state_size))
    states_next = np.zeros((batch_size, state_size))
    actions = np.zeros(batch_size, dtype=int)
    rewards = np.zeros(batch_size)
    dones = np.zeros(batch_size, dtype=bool)

    epsilon = 1.0
    scores, episodes = [], []
    for i in range(max_episodes):
        # env 초기화
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        done = False
        score = 0
        while not done:

            if render:
                env.render()

            # epsilon 탐욕 정책으로 action 선택
            epsilon = get_epsilon(epsilon)
            action = get_action(model, state, epsilon)

            # 선택한 action으로 step 진행
            state_next, reward, done, _ = env.step(action)
            state_next = np.reshape(state_next, [1, state_size])
            score += reward
            if done:
                reward = -100

            # 메모리에 경험(s, a, s', r) 저장
            memory.append((state, action, state_next, reward, done))

            if len(memory) > train_start:
                # memory에서 batch 크기만큼 무작위로 샘플 추출
                mini_batch = random.sample(memory, batch_size)
                for j in range(batch_size):
                    states[j] = mini_batch[j][0]
                    actions[j] = mini_batch[j][1]
                    states_next[j] = mini_batch[j][2]
                    rewards[j] = mini_batch[j][3]
                    dones[j] = mini_batch[j][4]

                # 벨만 방정식을 통한 q 함수 근사 및 model 업데이트
                q = bellmann_eq(model, model_target, states, actions, states_next, rewards, dones, batch_size)
                model.train_on_batch(states, q)

            state = state_next

        # model_target 업데이트
        model_target.set_weights(model.get_weights())

        print('Episode: {:3d}, Score: {:3.0f}, epsilon: {:.3f}'.format(i, score, epsilon))

        episodes.append(i)
        scores.append(score)

        # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
        if np.mean(scores[-min(10, len(scores)):]) > 490:
            break

    save_file_id = '{}(DQN)'.format(env.spec.id)
    model.save('{}.h5'.format(save_file_id))

    plt.figure()
    plt.plot(episodes, scores)
    plt.savefig('{}.svg'.format(save_file_id))

    return save_file_id


def play_model(model_file, env_id):
    model = load_model(model_file)

    env = gym.make(env_id)
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()

        state = np.reshape(state, [1, state.size])
        action = get_action(model, state, 0.0)
        state_next, reward, done, _ = env.step(action)
        score += reward
        state = state_next

    print('Score: {:3.0f}'.format(score))
