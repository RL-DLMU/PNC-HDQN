#交通专属   -2000， 贪婪一万

from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

class Config(object):
    #交通的数据
    batch_size = 50
    hidden_layers = [100, 100]     #[100, 100] [64, 64]

    hidden_size = 100
    trace_length = 10

    epsilon_start = 1.0#0.5#
    epsilon_end = 0.001
    epsilon_decay_step = 20000 #  15000

    max_steps = 100
    replary_memory_size = 200000


    savedir = 'save'
    auto_save = True
    save_freq = 10000
    use_tau = True
    tau = 0.001
    discount = 0.99

    qr_atoms = 64

    targetnet_update_freq = 300   #1000
    begin_train = 1000

    #delta_z = (v_max - v_min) / (atoms - 1.)
    #z = np.asarray([v_min + i * delta_z for i in range(atoms)]).astype(np.float32)

    learning_rate_minimum = 1e-4

    lr_method = "rmsprop"
    learning_rate = 1e-4
    lr_decay = 0.97
    keep_prob = 0.8

    num_lstm_layers = 1
    lstm_size = 512
    min_history = 4
    states_to_update = 4

    #oedc
    optimistic_start = 0.2
    optimistic_end = 1
    optimistic_decay_step = 80000#200000  # 160000, 18000, 200000, 220000, 240000. default 200000
    atoms = 51
    v_max = 0
    v_min = -2000
    d_u = 0.0002

    #lenient
    ts_greedy_coeff = 1.

    #lenient simple
    leniency_decay_step = 200000
    leniency_start = 1.
    leniency_end = 0.1

    #hysteretic
    her = 0.5


    #nui-ddqn
    sigma = 3
    min_decay_rate = [0.01, 0.99]  # 0.001 for negative reward and 0.995 for positive reward
    info_id = False  # true or false, ture: classify trajectories use info; false: don't classify trajectories
    nui_memory_size = 100000
    nui_begin_train = 1000
    delayed_return = True # true or false, ture: games with delayed return (the ship game); false: others
    nui_batch_size = 50

class GymConfig(Config):
    # ship
    agents = 1
    madrl = 'oe-cdqn' # options: dqn, lenient, hysteretic, lenient_simple, nui_ddqn, oe-cdqn



