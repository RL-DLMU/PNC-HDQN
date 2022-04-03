# PNC-HDQN
codes for paper 《Neighborhood Cooperative Multiagent Reinforcement Learning for Adaptive Traffic Signal Control in Epidemic Regions》

This is the code for using reinforcement learning(RL) algorithm to solve adaptive traffic signal control(ATSC).
This code models the ATSC problem as a networked Markov game (NMG), in which agents take into account information, including traffic conditions of it and its connected neighbors. A cooperative MARL framework named neighborhood cooperative hysteretic DQN (NC-HDQN) is proposed.
In the code, there are two RL algorithms, both using HDQN algorithm as the base algorithm.The first is called ENC-HDQN that maps the correlation degree between two connected agents according to vehicle numbers on roads between the two agents. The second method is named PNC-HDQN using the Pearson correlation coefficient to calculate the correlation degree adaptively.

If you want to train the PNC-HDQN algorithm, you can run PNC-HDQN.py.
If you want to train the ENC-HDQN algorithm, you can run ENC-HDQN.py.
Similarly, you also can run HDQN.py and DQN.py to train HDQN algorithm and DQN algorithm.

The hdqn_atsc.py in the agent_atsc folder is the code for agent algorithm framework construction.
The replay_buffer.py in the common folder is the code for experience replay.
The config_ship4000.py in the scr_ folder is the parameter setting for RL algorithm , and schedules.py is the code for ε-greedy.
The env_o.py builds a traffic control environment . The config.py is the parameter setting of the traffic environment.

Before you run the run.py, you need to create two new folders called model and records to store the model and data results, otherwise an error will be reported.
