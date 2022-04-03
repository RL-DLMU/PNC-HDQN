import argparse
import shutil
from collections import deque

import config

from agent_atsc.hdqn_atsc import HystereticDQNAgent

from anon_env_new import AnonEnv

import time
import pandas as pd
import os
import copy
import json
import numpy as np
from random import sample
import matplotlib.pyplot as plt
import math



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--road_net", type=str, default='3_4')   # which road net you are going to run
    parser.add_argument("--volume", type=str, default='jinan')
    parser.add_argument("--suffix", type=str, default="real")

    global hangzhou_archive
    hangzhou_archive = False
    global TOP_K_ADJACENCY
    TOP_K_ADJACENCY = 5
    global TOP_K_ADJACENCY_LANE
    TOP_K_ADJACENCY_LANE = 5
    global NUM_ROUNDS
    NUM_ROUNDS = 100
    global EARLY_STOP
    EARLY_STOP = False
    global NEIGHBOR
    # TAKE CARE
    # **********************是否使用
    NEIGHBOR = True
    global SAVEREPLAY  # if you want to relay your simulation, set it to be True
    SAVEREPLAY = False
    global ADJACENCY_BY_CONNECTION_OR_GEO
    # TAKE CARE
    ADJACENCY_BY_CONNECTION_OR_GEO = False

    # modify:TOP_K_ADJACENCY in line 154
    global PRETRAIN
    PRETRAIN = False
    # parser.add_argument("--mod", type=str, default='CoLight')  # SimpleDQN,SimpleDQNOne,GCN,CoLight,Lit
    parser.add_argument("--cnt", type=int, default=3600)  # 3600
    # parser.add_argument("-all", action="store_true", default=False)
    # parser.add_argument("--workers", type=int, default=7)
    # parser.add_argument("--onemodel", type=bool, default=False)

    global ANON_PHASE_REPRE
    # tt = parser.parse_args()
    # if 'CoLight_Signal' in tt.mod:
    # 12dim
    # ANON_PHASE_REPRE = {
    #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
    #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],  # 'WSES',
    #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],  # 'NSSS',
    #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],  # 'WLEL',
    #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]  # 'NLSL',
    # }
    # else:
    #     # 12dim
    ANON_PHASE_REPRE = {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
    }
    # print('ANON_PHASE_REPRE:', ANON_PHASE_REPRE)

    return parser.parse_args()


def _init_env(road_net, suffix, volume):
    # main(args.memo, args.env, args.road_net, args.gui, args.volume, args.ratio, args.mod, args.cnt, args.gen)
    # Jinan_3_4
    NUM_COL = int(road_net.split('_')[0])
    NUM_ROW = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)

    ENVIRONMENT = "anon"

    traffic_file_list = ["{0}_{1}_{2}_{3}".format(ENVIRONMENT, road_net, volume, suffix)]
    traffic_file_list = [i + ".json" for i in traffic_file_list]

    process_list = []

    global PRETRAIN
    global NUM_ROUNDS
    global EARLY_STOP
    for traffic_file in traffic_file_list:
        # 会覆盖config文件里的数据
        dic_exp_conf_extra = {
            #

            "TRAFFIC_FILE": [traffic_file],  # here: change to multi_traffic

            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            "NUM_ROUNDS": NUM_ROUNDS,

            "MODEL_POOL": False,
            "NUM_BEST_MODEL": 3,

            "PRETRAIN": PRETRAIN,  #

            "AGGREGATE": False,
            "DEBUG": False,
            "EARLY_STOP": EARLY_STOP,
        }

        global TOP_K_ADJACENCY
        global TOP_K_ADJACENCY_LANE
        global NEIGHBOR
        global SAVEREPLAY
        global ADJACENCY_BY_CONNECTION_OR_GEO
        global ANON_PHASE_REPRE
        dic_traffic_env_conf_extra = {
            "USE_LANE_ADJACENCY": True,
            # "ONE_MODEL": onemodel,
            "NUM_AGENTS": num_intersections,
            "NUM_INTERSECTIONS": num_intersections,
            "ACTION_PATTERN": "set",
            "MEASURE_TIME": 10,
            # "IF_GUI": gui,
            "DEBUG": False,
            "TOP_K_ADJACENCY": TOP_K_ADJACENCY,
            "ADJACENCY_BY_CONNECTION_OR_GEO": ADJACENCY_BY_CONNECTION_OR_GEO,
            "TOP_K_ADJACENCY_LANE": TOP_K_ADJACENCY_LANE,
            "SIMULATOR_TYPE": ENVIRONMENT,
            "BINARY_PHASE_EXPANSION": True,
            "FAST_COMPUTE": True,

            "NEIGHBOR": NEIGHBOR,
            # "MODEL_NAME": mod,

            "SAVEREPLAY": SAVEREPLAY,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,

            "TRAFFIC_FILE": traffic_file,
            "VOLUME": volume,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

            # 定义动作
            "phase_expansion": {
                1: [0, 1, 0, 1, 0, 0, 0, 0],
                2: [0, 0, 0, 0, 0, 1, 0, 1],
                3: [1, 0, 1, 0, 0, 0, 0, 0],
                4: [0, 0, 0, 0, 1, 0, 1, 0],
                5: [1, 1, 0, 0, 0, 0, 0, 0],
                6: [0, 0, 1, 1, 0, 0, 0, 0],
                7: [0, 0, 0, 0, 0, 0, 1, 1],
                8: [0, 0, 0, 0, 1, 1, 0, 0]
            },

            "phase_expansion_4_lane": {
                1: [1, 1, 0, 0],
                2: [0, 0, 1, 1],
            },

            # 要选择的状态特征在这里选择
            "LIST_STATE_FEATURE": [
                "cur_phase",
                # "time_this_phase",
                # "vehicle_position_img",
                # "vehicle_speed_img",
                # "vehicle_acceleration_img",
                # "vehicle_waiting_time_img",
                # 每个车道上的车辆数量（正在行驶+正在等待）
                "lane_num_vehicle",
                # "lane_num_vehicle_been_stopped_thres01",
                # 正在等待的车辆数量
                "lane_num_vehicle_been_stopped_thres1",
                # "lane_queue_length",
                # "lane_num_vehicle_left",
                # "lane_sum_duration_vehicle_left",
                # "lane_sum_waiting_time",
                # "terminal",
                # "coming_vehicle",
                # "leaving_vehicle",
                # "pressure"

                # "adjacency_matrix",
                # "lane_queue_length",
                # "connectivity",

                # adjacency_matrix_lane
            ],

            "DIC_FEATURE_DIM": dict(
                D_LANE_QUEUE_LENGTH=(4,),
                D_LANE_NUM_VEHICLE=(4,),

                D_COMING_VEHICLE=(12,),
                D_LEAVING_VEHICLE=(12,),

                D_LANE_NUM_VEHICLE_BEEN_STOPPED_THRES1=(4,),
                D_CUR_PHASE=(1,),
                D_NEXT_PHASE=(1,),
                D_TIME_THIS_PHASE=(1,),
                D_TERMINAL=(1,),
                D_LANE_SUM_WAITING_TIME=(4,),
                D_VEHICLE_POSITION_IMG=(4, 60,),
                D_VEHICLE_SPEED_IMG=(4, 60,),
                D_VEHICLE_WAITING_TIME_IMG=(4, 60,),

                D_PRESSURE=(1,),

                D_ADJACENCY_MATRIX=(2,),

                D_ADJACENCY_MATRIX_LANE=(6,), ),

            # 定义奖励，不要这个奖励就写0，需要就写系数
            "DIC_REWARD_INFO": {
                "flickering": 0,  # -5,#
                "sum_lane_queue_length": 0,
                "sum_lane_wait_time": 0,
                "sum_lane_num_vehicle_left": 0,  # -1,#
                "sum_duration_vehicle_left": 0,
                "sum_num_vehicle_been_stopped_thres01": 0,
                "sum_num_vehicle_been_stopped_thres1": -0.25,
                "pressure": 0
            },

            "LANE_NUM": {
                "LEFT": 1,
                "RIGHT": 1,
                "STRAIGHT": 1
            },

            "PHASE": {
                # "sumo": {
                #     0: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     1: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     2: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                #     3: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                # },

                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],# 'WSES',
                #     2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],# 'NSSS',
                #     3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],# 'WLEL',
                #     4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
                "anon": ANON_PHASE_REPRE,
                # "anon": {
                #     # 0: [0, 0, 0, 0, 0, 0, 0, 0],
                #     1: [0, 1, 0, 1, 0, 0, 0, 0],# 'WSES',
                #     2: [0, 0, 0, 0, 0, 1, 0, 1],# 'NSSS',
                #     3: [1, 0, 1, 0, 0, 0, 0, 0],# 'WLEL',
                #     4: [0, 0, 0, 0, 1, 0, 1, 0]# 'NLSL',
                #     # 'WSWL',
                #     # 'ESEL',
                #     # 'WSES',
                #     # 'NSSS',
                #     # 'NSNL',
                #     # 'SSSL',
                # },
            }
        }

        ## ==================== multi_phase ====================
        global hangzhou_archive
        if hangzhou_archive:
            template = 'Archive+2'
        elif volume == "mydata":
            template = "mydata"
        elif volume == 'jinan':
            template = "Jinan"
        elif volume == 'hangzhou':
            template = 'Hangzhou'
        elif volume == 'newyork':
            template = 'NewYork'
        elif volume == 'chacha':
            template = 'Chacha'
        elif volume == 'dynamic_attention':
            template = 'dynamic_attention'
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LS:
            template = "template_ls"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._S:
            template = "template_s"
        elif dic_traffic_env_conf_extra["LANE_NUM"] == config._LSR:
            template = "template_lsr"
        else:
            raise ValueError

        if dic_traffic_env_conf_extra['NEIGHBOR']:
            list_feature = dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].copy()
            list_feature.remove("lane_num_vehicle_been_stopped_thres1")
            for feature in list_feature:
                for i in range(4):
                    dic_traffic_env_conf_extra["LIST_STATE_FEATURE"].append(feature + "_" + str(i))

        dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]

        if dic_traffic_env_conf_extra['BINARY_PHASE_EXPANSION']:
            dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE'] = (8,)
            if dic_traffic_env_conf_extra['NEIGHBOR']:
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (8,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)
            else:

                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_0'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_0'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_1'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_1'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_2'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_2'] = (4,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_CUR_PHASE_3'] = (1,)
                dic_traffic_env_conf_extra['DIC_FEATURE_DIM']['D_LANE_NUM_VEHICLE_3'] = (4,)

        print(traffic_file)
        prefix_intersections = str(road_net)
        dic_path_extra = {
            # 模型保存的位置
            "PATH_TO_MODEL": os.path.join("model", traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                      time.localtime(time.time()))),
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", traffic_file + "_" + time.strftime('%m_%d_%H_%M_%S',
                                                                                                 time.localtime(
                                                                                                     time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, prefix_intersections),
            # "PATH_TO_PRETRAIN_MODEL": os.path.join("model", "initial", traffic_file),
            # "PATH_TO_PRETRAIN_WORK_DIRECTORY": os.path.join("records", "initial", traffic_file),
            # "PATH_TO_ERROR": os.path.join("errors")
        }
        # merge(left, right, how='inner', on=None, left_on=None, right_on=None, left_index=False, right_index=False, sort=False,
        # suffixes=('_x', '_y'), copy=True, indicator=False, validate=None)
        # 数据集连接合并  如果left,right有相同的key，则链接呢，不相同则内连接，全连接，左连接几种形式
        deploy_dic_exp_conf = merge(config.DIC_EXP_CONF, dic_exp_conf_extra)
        # deploy_dic_agent_conf = merge(getattr(config, "DIC_{0}_AGENT_CONF".format(mod.upper())),
        #                               dic_agent_conf_extra)
        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)
        # getattr() 函数用于返回一个对象属性值。
        # TODO add agent_conf for different agents
        # deploy_dic_agent_conf_all = [deploy_dic_agent_conf for i in range(deploy_dic_traffic_env_conf["NUM_AGENTS"])]

        deploy_dic_path = merge(config.DIC_PATH, dic_path_extra)

        path_check(deploy_dic_path)
        copy_conf_file(deploy_dic_path, deploy_dic_exp_conf, deploy_dic_traffic_env_conf)
        copy_anon_file(deploy_dic_path, deploy_dic_exp_conf)

        path_to_log = os.path.join(deploy_dic_path["PATH_TO_WORK_DIRECTORY"], "train_round")
        if not os.path.exists(path_to_log):
            os.makedirs(path_to_log)

        env = AnonEnv(
            path_to_log=path_to_log,
            path_to_work_directory=deploy_dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=deploy_dic_traffic_env_conf)

        return env, deploy_dic_exp_conf, deploy_dic_path


def copy_conf_file(dic_path, dic_exp_conf, dic_traffic_env_conf):
    # write conf files

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_exp_conf, open(os.path.join(path, "exp.conf"), "w"),
              indent=4)
    json.dump(dic_traffic_env_conf,
              open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def path_check(deploy_dic_path):
    for path in deploy_dic_path.values():
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)


def copy_anon_file(dic_path, dic_exp_conf):
    # hard code !!!

    path = dic_path["PATH_TO_WORK_DIRECTORY"]
    # copy sumo files

    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["TRAFFIC_FILE"][0]),
                os.path.join(path, dic_exp_conf["TRAFFIC_FILE"][0]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_exp_conf["ROADNET_FILE"]),
                os.path.join(path, dic_exp_conf["ROADNET_FILE"]))


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)

    return dic_result

def plot_leniency_sample(x,y):
    plt.plot(x, y)
    plt.ylabel('leniency_sample')
    plt.xlabel('train times')
    plt.show()

def plot_leniencyn(x,y):
    plt.plot(x, y)
    plt.ylabel('leniencyn')
    plt.xlabel('train times')
    plt.show()

# 转换形式
def tra_state(state):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']

    s = np.concatenate((cur_phase, lane_num_vehicle))

    return s


def state2state(state, current_step):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    step = float(current_step / 360)
    return np.concatenate(([step], cur_phase, lane_num_vehicle))


# 状态加邻居
def neighbor_state(state):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    nei_state = np.concatenate((cur_phase, lane_num_vehicle))
    for i in range(4):
        nei_cur_phase = state["cur_phase_{0}".format(i)]
        nei_lane_num_vehicle = state["lane_num_vehicle_{0}".format(i)]
        nei_state = np.concatenate((nei_state, nei_cur_phase, nei_lane_num_vehicle))
    return nei_state


def neistate_class(state):
    cur_phase = state['cur_phase']
    lane_num_vehicle = state['lane_num_vehicle']
    nei_state = np.concatenate((cur_phase, lane_num_vehicle))
    for i in range(4):
        nei_cur_phase = state["cur_phase_{0}".format(i)]
        nei_lane_num_vehicle = state["lane_num_vehicle_{0}".format(i)]

        a = 0
        for t in range(len(nei_lane_num_vehicle)):
            a += nei_lane_num_vehicle[t]
        if a >= 0 and a < 100 :
            for t in range(len(nei_lane_num_vehicle)):
                nei_lane_num_vehicle[t] = 0
        elif a >= 100 and a < 200 :
            for t in range(len(nei_lane_num_vehicle)):
                nei_lane_num_vehicle[t] = nei_lane_num_vehicle[t] * 0.5
        elif a >= 200 :
            for t in range(len(nei_lane_num_vehicle)):
                nei_lane_num_vehicle[t] = nei_lane_num_vehicle[t]

        nei_state = np.concatenate((nei_state, nei_cur_phase, nei_lane_num_vehicle))
    return nei_state



def ttest(agent, render=False, load_model=False):
    args = parse_args()
    test_env, test_dic_exp_conf, test_deploy_dic_path = _init_env(args.road_net, args.suffix, args.volume)
    agent_num = len(agent)
    episode_len = int(test_dic_exp_conf["RUN_COUNTS"] / test_env.dic_traffic_env_conf["MIN_ACTION_TIME"])

    ep_agent_reward = []
    ep_reward = []
    agent_reward = np.zeros(agent_num)
    for iii in range(2):
        state = test_env.reset()

        ep_len = 0
        final_reward = 0
        while ep_len < episode_len:
            action = []
            for i in range(agent_num):

                s = tra_state(state[i])
                action.append(agent[i].choose_action_optimistic(s, 0.0))
            next_state_, reward_, done_, ind_reward_ = test_env.step(action)
            state = next_state_
            final_reward += np.mean(reward_)
            for i in range(agent_num):
                agent_reward[i] += reward_[i]
            ep_len += 1
        ep_agent_reward.append(agent_reward)
        ep_reward.append(final_reward)

    return np.mean(ep_agent_reward, axis=0), np.mean(ep_reward)


def train(env, dic_exp_conf, deploy_dic_path):
    env.reset()
    Agent_ls = []
    leniency1 = []
    episode1 = []
    erm_size = 200000
    agent_num = env.dic_traffic_env_conf["NUM_INTERSECTIONS"]
    episode_len = int(dic_exp_conf["RUN_COUNTS"] / env.dic_traffic_env_conf["MIN_ACTION_TIME"])

    for i in range(agent_num):
        Agent_ls.append(HystereticDQNAgent(65, 4, 'rnd_hdqnAgent' + str(i), logdir='ologs1_o12_1', savedir='osave1_o12_1'))



    print('after init')
    train_log = []
    test_log = []
    global_step = 0
    episode = 0

    while episode <= 2000 :
        state = env.reset()

        current_step = 0
        episode += 1
        train_agent_reward = np.zeros(agent_num)
        train_reward_ind = 0
        train_reward = 0
        episode1 = []
        episoden = []

        log_model = False

        for i in range(env.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
            episoden.append(episode1)

        while current_step < episode_len:
            global_step = global_step + 1
            actions = []

            for i in range(agent_num):

                s = neighbor_state(state[i])

                actions.append(Agent_ls[i].choose_action(s))



            if episode == 2000:
                log_model = True

            # 下一状态，自己定义的奖励（智能体和上下左右四个智能体（一共五个智能体）的奖励），交通没有终止状态一直false，智能体独立的奖励
            next_state, reward, done, ind_reward = env.step(actions, log_model)

            avg_reward_ind = np.mean(ind_reward)
            train_reward_ind += avg_reward_ind

            avg_reward = np.mean(reward)
            train_reward += avg_reward


            # store experience
            for i in range(agent_num):
                train_agent_reward[i] += ind_reward[i]

                s = neighbor_state(state[i])

                next_s = neighbor_state(next_state[i])

                Agent_ls[i].store(s, actions[i], reward[i], next_s, done)

                episoden[i].append((s, actions[i]))

            # train
            if global_step % 10 == 0 :
                for i in range(agent_num):
                    Agent_ls[i].train_hdqn()


            state = next_state
            current_step += 1


        std_reward = np.std(train_agent_reward)
        log = {"global_step": global_step,
               "episode:": episode,
               'agent_reward': train_agent_reward,
               'eposode_reward_ind': train_reward_ind,
               'eposode_reward': train_reward,
               'std_reward': std_reward,
               'epsilon:': Agent_ls[1].epsilon,

               }
        train_log.append(log)
        print("log", log)


        if episode > 1000 and episode % 20 == 0:
            env.bulk_log_multi_process()




        if episode % 50 == 0:
            train_log_name = deploy_dic_path["PATH_TO_WORK_DIRECTORY"] + "/train_" + str(
                dic_exp_conf["TRAFFIC_FILE"][0]) + ".csv"
            dd = pd.DataFrame(train_log)
            dd.to_csv(train_log_name)





    train_log_name = deploy_dic_path["PATH_TO_WORK_DIRECTORY"] + "/train_" + str(
        dic_exp_conf["TRAFFIC_FILE"][0]) + ".csv"
    dd = pd.DataFrame(train_log)
    dd.to_csv(train_log_name)



if __name__ == '__main__':
    args = parse_args()
    env, dic_exp_conf, deploy_dic_path = _init_env(args.road_net, args.suffix, args.volume)
    train(env, dic_exp_conf, deploy_dic_path)
