
import tensorflow as tf
import tensorflow.contrib.layers as layers
import numpy as np
import random
from scr_.schedules import LinearSchedule
from common.replay_buffer import ReplayBuffer
from scr_.config_ship4000 import Config


class DQNAgent(object):
    """
    refs: https://github.com/skumar9876/Hierarchical-DQN/blob/master/dqn.py
    """

    def __init__(self, states_n, actions_n, scope_name: str, sess=None, seed=1, logdir='logs', savedir='save'):
        """

        :param states_n: tuple
        :param actions_n: int
        :param hidden_layers: list
        :param scope_name: str
        :param sess: tf.Session
        :param learning_rate: float
        :param discount: float
        :param replay_memory_size: int
        :param batch_size: int
        :param begin_train: int
        :param targetnet_update_freq: int
        :param epsilon_start: float
        :param epsilon_end: float
        :param epsilon_decay_step: int
        :param seed: int
        :param logdir: str
        """
        self.states_n = states_n
        self.actions_n = actions_n
        self._hidden_layers = Config.hidden_layers
        self._scope_name = scope_name
        self.lr = Config.learning_rate
        self._target_net_update_freq = Config.targetnet_update_freq
        self._current_time_step = 0
        self._epsilon_schedule = LinearSchedule(Config.epsilon_decay_step, Config.epsilon_end, Config.epsilon_start)
        self._train_batch_size = Config.batch_size
        self._begin_train = Config.begin_train
        self._gamma = Config.discount

        self._use_tau = Config.use_tau
        self._tau = Config.tau

        self.savedir = savedir
        self.save_freq = Config.save_freq

        self.qnet_optimizer = tf.train.AdamOptimizer(self.lr)

        self._replay_buffer = ReplayBuffer(Config.replary_memory_size)

        self._seed(seed)

        with tf.Graph().as_default():
            self._build_graph()
            self._merged_summary = tf.summary.merge_all()

            if sess is None:
                self.sess = tf.Session()
            else:
                self.sess = sess
            self.sess.run(tf.global_variables_initializer())

            self._saver = tf.train.Saver()

            self._summary_writer = tf.summary.FileWriter(logdir=logdir)
            self._summary_writer.add_graph(tf.get_default_graph())

    def show_memory(self):
        print(self._replay_buffer.show())

    def _q_network(self, state, hidden_layers, outputs, scope_name, trainable):

        with tf.variable_scope(scope_name):
            out = state
            for ly in hidden_layers:
                out = layers.fully_connected(out, ly, activation_fn=tf.nn.relu, trainable=trainable)
            out = layers.fully_connected(out, outputs, activation_fn=None, trainable=trainable)
        return out

    def _build_graph(self):
        #self._state = tf.placeholder(dtype=tf.float32, shape=(None, ) + self.states_n, name='state_input')
        self._state = tf.placeholder(dtype=tf.float32, shape=(None, ) + (self.states_n,), name='state_input')


        with tf.variable_scope(self._scope_name):
            self._q_values = self._q_network(self._state, self._hidden_layers, self.actions_n, 'q_network', True)
            self._target_q_values = self._q_network(self._state, self._hidden_layers, self.actions_n, 'target_q_network', False)

        with tf.variable_scope('q_network_update'):
            self._actions_onehot = tf.placeholder(dtype=tf.float32, shape=(None, self.actions_n), name='actions_onehot_input')
            self._td_targets = tf.placeholder(dtype=tf.float32, shape=(None, ), name='td_targets')
            self._q_values_pred = tf.reduce_sum(self._q_values * self._actions_onehot, axis=1)

            self._error = tf.abs(self._q_values_pred - self._td_targets)
            quadratic_part = tf.clip_by_value(self._error, 0.0, 1.0)
            linear_part = self._error - quadratic_part
            self._loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

            qnet_gradients = self.qnet_optimizer.compute_gradients(self._loss, tf.trainable_variables())
            for i, (grad, var) in enumerate(qnet_gradients):
                if grad is not None:
                    qnet_gradients[i] = (tf.clip_by_norm(grad, 10), var)
            self.train_op = self.qnet_optimizer.apply_gradients(qnet_gradients)

            tf.summary.scalar('loss', self._loss)


            with tf.name_scope('target_network_update'):
                q_network_params = [t for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                                 scope=self._scope_name + '/q_network')
                                    if t.name.startswith(self._scope_name + '/q_network/')]
                target_q_network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                            scope=self._scope_name + '/target_q_network')

                self.target_update_ops = []
                for var, var_target in zip(sorted(q_network_params, key=lambda v: v.name),
                                           sorted(target_q_network_params, key=lambda v: v.name)):
                    # self.target_update_ops.append(var_target.assign(var))

                    # soft target update
                    self.target_update_ops.append(var_target.assign(tf.multiply(var_target, 1 - self._tau) +
                                                                    tf.multiply(var, self._tau)))
                self.target_update_ops = tf.group(*self.target_update_ops)

    def choose_action(self, state, epsilon=None):
        """
        for one agent
        :param state:
        :param epsilon:
        :return:
        """
        if epsilon is not None:
            epsilon_used = epsilon
        else:
            epsilon_used = self._epsilon_schedule.value(self._current_time_step)
        if np.random.random() < epsilon_used:
            return np.random.randint(0, self.actions_n)
        else:
            q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})

            return np.argmax(q_values[0])

    def choose_actions(self, states, epsilons=None):
        """
        for multi-agent
        :param states:
        :param epsilon:
        :return:
        """
        if epsilons is not None:
            epsilons_used = epsilons
        else:
            epsilons_used = self._epsilon_schedule.value(self._current_time_step)

        actions = []
        for i, state in enumerate(states):
            if np.random.random() < epsilons_used[i]:
                actions.append(np.random.randint(0, self.actions_n))
            else:
                q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})

                actions.append(np.argmax(q_values[0]))

        return actions

    def check_network_output(self, state):
        q_values = self.sess.run(self._q_values, feed_dict={self._state: state[None]})
        print(q_values[0])

    def store(self, state, action, reward, next_state, terminate):
        self._replay_buffer.add(state, action, reward, next_state, terminate)

    # DQN：动作是eval中的最大q值动作a，计算q现实是用target中的最大q值（不是前面的a对应的q值）来更新；测试时只用eval网络选动作
    # DDQN：动作是eval中的最大q值动作a，计算q现实是用前面的a在target对应的q值来更新计算。
    def get_max_target_Q_s_a(self, next_states):
        next_state_q_values = self.sess.run(self._q_values, feed_dict={self._state: next_states})
        next_state_target_q_values = self.sess.run(self._target_q_values, feed_dict={self._state: next_states})

        # 估计网络最大q值的a
        next_select_actions = np.argmax(next_state_q_values, axis=1)
        bt_sz = len(next_states)
        next_select_actions_onehot = np.zeros((bt_sz, self.actions_n))
        for i in range(bt_sz):
            next_select_actions_onehot[i, next_select_actions[i]] = 1.

        # target网络中eval_max动作的q值
        next_state_max_q_values = np.sum(next_state_target_q_values * next_select_actions_onehot, axis=1)
        return next_state_max_q_values

    def train(self):

        self._current_time_step += 1
        self.epsilon = self._epsilon_schedule.value(self._current_time_step)


        if self._current_time_step == 1:
            print('Training starts.')
            self.sess.run(self.target_update_ops)

        if self._current_time_step > self._begin_train:
            states, actions, rewards, next_states, terminates = self._replay_buffer.sample(batch_size=
                                                                                           self._train_batch_size)

            actions_onehot = np.zeros((self._train_batch_size, self.actions_n))
            for i in range(self._train_batch_size):
                # target网络s'的动作
                actions_onehot[i, actions[i]] = 1.

            # 输入s 得到估计q
            next_state_q_values = self.sess.run(self._q_values, feed_dict={self._state: next_states})
            # 输入s' 得到现实q
            next_state_target_q_values = self.sess.run(self._target_q_values, feed_dict={self._state: next_states})

            # 下一状态 估计q的动作
            next_select_actions = np.argmax(next_state_q_values, axis=1)
            next_select_actions_onehot = np.zeros((self._train_batch_size, self.actions_n))
            for i in range(self._train_batch_size):
                next_select_actions_onehot[i, next_select_actions[i]] = 1.

            # 下一状态 最大现实q值；用target中的最大q值去更新eval中的最大q值（target计算），然后和原eval的q相减求误差
            next_state_max_q_values = np.sum(next_state_target_q_values * next_select_actions_onehot, axis=1)

            # 下一状态 q现实
            td_targets = rewards + self._gamma * next_state_max_q_values * (1 - terminates)

            _, str_ = self.sess.run([self.train_op, self._merged_summary], feed_dict={self._state: states,
                                                    self._actions_onehot: actions_onehot,
                                                    self._td_targets: td_targets})

            self._summary_writer.add_summary(str_, self._current_time_step)

        # update target_net
        if self._use_tau:
            self.sess.run(self.target_update_ops)
        else:
            if self._current_time_step % self._target_net_update_freq == 0:
                self.sess.run(self.target_update_ops)

        # save model
        if self._current_time_step % self.save_freq == 0:

            # TODO save the model with highest performance
            self._saver.save(sess=self.sess, save_path=self.savedir + '/my-model',
                             global_step=self._current_time_step)

    def train_without_replaybuffer(self, states, actions, target_values):

        self._current_time_step += 1

        if self._current_time_step == 1:
            print('Training starts.')
            self.sess.run(self.target_update_ops)

        bt_sz = len(states)
        actions_onehot = np.zeros((bt_sz, self.actions_n))
        for i in range(bt_sz):
            actions_onehot[i, actions[i]] = 1.

        _, str_ = self.sess.run([self.train_op, self._merged_summary], feed_dict={self._state: states,
                                                self._actions_onehot: actions_onehot,
                                                self._td_targets: target_values})

        self._summary_writer.add_summary(str_, self._current_time_step)

        # update target_net
        if self._use_tau:
            self.sess.run(self.target_update_ops)
        else:
            if self._current_time_step % self._target_net_update_freq == 0:
                self.sess.run(self.target_update_ops)

        # save model
        if self._current_time_step % self.save_freq == 0:

            # TODO save the model with highest performance
            self._saver.save(sess=self.sess, save_path=self.savedir + '/my-model',
                             global_step=self._current_time_step)

    def load_model(self):
        self._saver.restore(self.sess, tf.train.latest_checkpoint(self.savedir))

    def _seed(self, lucky_number):
        tf.set_random_seed(lucky_number)
        np.random.seed(lucky_number)
        random.seed(lucky_number)

