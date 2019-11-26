import tensorflow as tf
import numpy as np
import random
from collections import deque

class DQN:
    REPLAY_MEMORY = 10000  # 학습에 사용할 플레이 결고를 얼마나 저장해서 사용할지 정함
    BATCH_SIZE = 32  # 한 번 학습할 때 몇 개의 기억을 사용할지 지정. 미니배치의 크기
    GAMMA = 0.99  # 오래된 상태의 가중치를 위한 하이퍼파라미터
    STATE_LEN = 4  # 한 번에 볼 프레임의 총 수

    def __init__(self, session, width, height, n_action):
        self.session = session
        self.n_action = n_action
        self.width = width
        self.height = height
        self.memory = deque()
        self.state = None
        # placeholder
        self.input_X = tf.placeholder(tf.float32, [None, width, height, self.STATE_LEN])  # 게임의 상태를 입력받음
        self.input_A = tf.placeholder(tf.int64, [None])  # 각 상태를 만들어낸 액션 값을 받음
        self.input_Y = tf.placeholder(tf.float32, [None])  # 손실값 계산에 사용할 값을 입력받음

        # 학습신경망, 목표신경망 구성
        self.Q = self._build_network('main')  # 게임 진행 시, 행동 예측하는데 사용하는 주 신경망
        self.cost, self.train_op = self._build_op()  # 학습시에만 보조적으로 사용하는 목표신경망

        self.target_Q = self._build_network('target')

    def _build_network(self, name):
        """
        CNN으로 구성. pooling 계층이 존재하지 않음.(이미지의 세세한 부분까지 판단하도록 하기 위함)
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            model = tf.layers.conv2d(self.input_X, 32, [4, 4], padding="same", activation=tf.nn.relu)
            model = tf.layers.conv2d(model, 64, [2, 2], padding="same", activation=tf.nn.relu)
            model = tf.contrib.layers.flatten(model)
            model = tf.layers.dense(model, 512, activation=tf.nn.relu)

            Q = tf.layers.dense(model, self.n_action, activation=None)

        return Q

    def _build_op(self):
        """
        DQN의 손실함수
         - 현재 상태를 이용해 학습 신경망으로 구한 Q_value와 다른 상태를 이용해 목표 신경망으로 구한 Q_value(input_Y)를 이용
        :return:
        """
        one_hot = tf.one_hot(self.input_A, self.n_action, 1.0, 0.0)
        Q_value = tf.reduce_sum(tf.multiply(self.Q, one_hot), axis=1)
        cost = tf.reduce_mean(tf.square(self.input_Y - Q_value))
        train_op = tf.train.AdamOptimizer(1e-6).minimize(cost)

        return cost, train_op

    def update_target_network(self):
        """
        목표신경망 갱신
         - 학습 신경망의 변수들의 값을 목표 신경망으로 복사해서 목표 신경망들의 변수들을 갱신
        :return:
        """
        copy_op = []

        main_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='main')

        target_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target')

        for main_var, target_var in zip(main_vars, target_vars):
            copy_op.append(target_var.assign(main_var.value()))

        self.session.run(copy_op)

    def get_action(self):
        """
        현재 상태를 이용해 다음에 취해야 할 행동을 찾는 함수
        :return:
        """
        Q_value = self.session.run(self.Q, feed_dict={self.input_X: [self.state]})

        action = np.argmax(Q_value[0])

        return action

    def train(self):
        state, next_state, action, reward, terminal = self._sample_memory()

        target_Q_value = self.session.run(self.target_Q, feed_dict={self.input_X: next_state})

        Y = []
        for i in range(self.BATCH_SIZE):
            if terminal[i]:
                Y.append(reward[i])
            else:
                Y.append(reward[i] + self.GAMMA * np.max(target_Q_value[i]))

        self.session.run(self.train_op, feed_dict={self.input_X: state, self.input_A: action, self.input_Y: Y})

    def init_state(self, state):
        state = [state for _ in range(self.STATE_LEN)]
        self.state = np.stack(state, axis=2)

    def remember(self, state, action, reward, terminal):
        next_state = np.reshape(state, (self.width, self.height, 1))
        next_state = np.append(self.state[:, :, 1:], next_state, axis=2)

        self.memory.append((self.state, next_state, action, reward, terminal))

        if len(self.memory) > self.REPLAY_MEMORY:
            self.memory.popleft()

        self.state = next_state

    def _sample_memory(self):
        sample_memory = random.sample(self.memory, self.BATCH_SIZE)

        state = [memory[0] for memory in sample_memory]
        next_state = [memory[1] for memory in sample_memory]
        action = [memory[2] for memory in sample_memory]
        reward = [memory[3] for memory in sample_memory]
        terminal = [memory[4] for memory in sample_memory]

        return state, next_state, action, reward, terminal