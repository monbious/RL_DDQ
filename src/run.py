"""
This should be a simple minimalist run file. It's only responsibility should be to parse the arguments 
(which agent, user simulator to use) and launch a dialog simulation.
"""
import sys
import argparse, json, copy, os
import pickle
import numpy
import random
import torch

from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import (AgentCmd, InformAgent, RequestAllAgent, RandomAgent, 
                                EchoAgent, RequestBasicsAgent, AgentDQN, AgentSAC, AgentTD3, AgentDQNZ)
from deep_dialog.usersims import RuleSimulator, ModelBasedSimulator

from deep_dialog import dialog_config
from deep_dialog.dialog_config import *

from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

from deep_dialog.data.win2unix import win2unix

torch.set_num_threads(24)

seed = 5
numpy.random.seed(seed)
random.seed(seed)

""" 
Launch a dialog simulation per the command line arguments
This function instantiates a user_simulator, an agent, and a dialog system.
Next, it triggers the simulator to run for the specified number of episodes.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dict_path', dest='dict_path', type=str, default='./deep_dialog/data/dicts.v3.p',
                        help='path to the .json dictionary file')
    parser.add_argument('--movie_kb_path', dest='movie_kb_path', type=str, 
                        default='./deep_dialog/data/movie_kb.1k.p', help='path to the movie kb .json file')
    parser.add_argument('--act_set', dest='act_set', type=str, default='./deep_dialog/data/dia_acts.txt',
                        help='path to dia act set; none for loading from labeled file')
    parser.add_argument('--slot_set', dest='slot_set', type=str, default='./deep_dialog/data/slot_set.txt',
                        help='path to slot set; none for loading from labeled file')
    parser.add_argument('--goal_file_path', dest='goal_file_path', type=str,
                        default='./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p',
                        help='a list of user goals')
    parser.add_argument('--diaact_nl_pairs', dest='diaact_nl_pairs', type=str,
                        default='./deep_dialog/data/dia_act_nl_pairs.v6.json',
                        help='path to the pre-defined dia_act&NL pairs')

    parser.add_argument('--max_turn', dest='max_turn', default=20, type=int,
                        help='maximum length of each dialog (default=20, 0=no maximum length)')
    parser.add_argument('--episodes', dest='episodes', default=1, type=int,
                        help='Total number of episodes to run (default=1)')
    parser.add_argument('--slot_err_prob', dest='slot_err_prob', default=0.05, type=float,
                        help='the slot err probability')
    parser.add_argument('--slot_err_mode', dest='slot_err_mode', default=0, type=int,
                        help='slot_err_mode: 0 for slot_val only; 1 for three errs')
    parser.add_argument('--intent_err_prob', dest='intent_err_prob', default=0.05, type=float,
                        help='the intent err probability')

    parser.add_argument('--agt', dest='agt', default=0, type=int,
                        help='Select an agent: 0 for a command line input, 1-6 for rule based agents')
    parser.add_argument('--usr', dest='usr', default=0, type=int,
                        help='Select a user simulator. 0 is a Frozen user simulator.')

    parser.add_argument('--epsilon', dest='epsilon', type=float, default=0.3,
                        help='Epsilon to determine stochasticity of epsilon-greedy agent policies')

    # load NLG & NLU model
    parser.add_argument('--nlg_model_path', dest='nlg_model_path', type=str,
                        default='./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
                        help='path to model file')
    parser.add_argument('--nlu_model_path', dest='nlu_model_path', type=str,
                        default='./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
                        help='path to the NLU model file')

    parser.add_argument('--act_level', dest='act_level', type=int, default=0,
                        help='0 for dia_act level; 1 for NL level')
    parser.add_argument('--run_mode', dest='run_mode', type=int, default=0,
                        help='run_mode: 0 for default NL; 1 for dia_act; 2 for both')
    parser.add_argument('--auto_suggest', dest='auto_suggest', type=int, default=0,
                        help='0 for no auto_suggest; 1 for auto_suggest')
    parser.add_argument('--cmd_input_mode', dest='cmd_input_mode', type=int, default=0,
                        help='run_mode: 0 for NL; 1 for dia_act')

    # RL agent parameters
    parser.add_argument('--experience_replay_pool_size', dest='experience_replay_pool_size', 
                        type=int, default=5000, help='the size for experience replay')
    parser.add_argument('--dqn_hidden_size', dest='dqn_hidden_size', type=int, default=60,
                        help='the hidden size for DQN')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.9, help='gamma for DQN')
    parser.add_argument('--predict_mode', dest='predict_mode', type=bool, default=False,
                        help='predict model for DQN')
    parser.add_argument('--simulation_epoch_size', dest='simulation_epoch_size', type=int, default=50,
                        help='the size of validation set')
    parser.add_argument('--warm_start', dest='warm_start', type=int, default=1,
                        help='0: no warm start; 1: warm start for training')
    parser.add_argument('--warm_start_epochs', dest='warm_start_epochs', type=int, default=100,
                        help='the number of epochs for warm start')
    parser.add_argument('--planning_steps', dest='planning_steps', type=int, default=4,
                        help='the number of planning steps')

    parser.add_argument('--trained_model_path', dest='trained_model_path', type=str, default=None,
                        help='the path for trained model')
    parser.add_argument('-o', '--write_model_dir', dest='write_model_dir', type=str,
                        default='./deep_dialog/checkpoints/', help='write model to disk')
    parser.add_argument('--save_check_point', dest='save_check_point', type=int, default=100,
                        help='number of epochs for saving model')
    parser.add_argument('--print_interval', dest='print_interval', type=int, default=20,
                        help='interval of printing the process')

    # We changed to have a queue to hold experences. So this threshold will not be used to flush the buffer.
    parser.add_argument('--success_rate_threshold', dest='success_rate_threshold', type=float, default=0.6,
                        help='the threshold for success rate')

    parser.add_argument('--split_fold', dest='split_fold', default=5, type=int,
                        help='the number of folders to split the user goal')
    parser.add_argument('--learning_phase', dest='learning_phase', default='all', type=str,
                        help='train/test/all; default is all')

    parser.add_argument('--grounded', dest='grounded', type=int, default=0,
                        help='planning k steps with environment rather than world model')
    parser.add_argument('--boosted', dest='boosted', type=int, default=1, help='Boost the world model')
    parser.add_argument('--train_world_model', dest='train_world_model', type=int, default=1,
                        help='Whether train world model on the fly or not')
    parser.add_argument('--torch_seed', dest='torch_seed', type=int, default=100,
                        help='random seed for troch')

    args = parser.parse_args()
    params = vars(args)

    print('Dialog Parameters: ')
    print(json.dumps(params, indent=2))

max_turn = params['max_turn']
num_episodes = params['episodes']

agt = params['agt']
usr = params['usr']

if not os.path.exists(params['write_model_dir']):
    os.mkdir(params['write_model_dir'])

# load the user goals from .p file
goal_file_path = params['goal_file_path']
try:
    all_goal_set = pickle.load(open(goal_file_path, 'rb'))
except:
    all_goal_set = pickle.load(open(win2unix(goal_file_path), 'rb'))

# split goal set
split_fold = params.get('split_fold', 5)
goal_set = {'train': [], 'valid': [], 'test': [], 'all': []}
for u_goal_id, u_goal in enumerate(all_goal_set):
    if u_goal_id % split_fold == 1:
        goal_set['test'].append(u_goal)
    else:
        goal_set['train'].append(u_goal)
    goal_set['all'].append(u_goal)
# end split goal set

movie_kb_path = params['movie_kb_path']
try:
    movie_kb = pickle.load(open(movie_kb_path, 'rb'))
except:
    movie_kb = pickle.load(open(win2unix(movie_kb_path), 'rb'), encoding='iso-8859-1')

act_set = text_to_dict(params['act_set'])
slot_set = text_to_dict(params['slot_set'])

################################################################################
# a movie dictionary for user simulator - slot:possible values
################################################################################
dict_path = params['dict_path']
try:
    movie_dictionary = pickle.load(open(dict_path, 'rb'), encoding='iso-8859-1')
except:
    movie_dictionary = pickle.load(open(win2unix(dict_path), 'rb'))

dialog_config.run_mode = params['run_mode']
dialog_config.auto_suggest = params['auto_suggest']

################################################################################
#   Parameters for Agents
################################################################################
agent_params = {}
agent_params['max_turn'] = max_turn
agent_params['epsilon'] = params['epsilon']
agent_params['agent_run_mode'] = params['run_mode']
agent_params['agent_act_level'] = params['act_level']

agent_params['experience_replay_pool_size'] = params['experience_replay_pool_size']
agent_params['dqn_hidden_size'] = params['dqn_hidden_size']
agent_params['batch_size'] = params['batch_size']
agent_params['gamma'] = params['gamma']
agent_params['predict_mode'] = params['predict_mode']
agent_params['trained_model_path'] = params['trained_model_path']
agent_params['warm_start'] = params['warm_start']
agent_params['cmd_input_mode'] = params['cmd_input_mode']

################################################################################
#   Parameters for User Simulators
################################################################################
usersim_params = {}
usersim_params['max_turn'] = max_turn
usersim_params['slot_err_probability'] = params['slot_err_prob']
usersim_params['slot_err_mode'] = params['slot_err_mode']
usersim_params['intent_err_probability'] = params['intent_err_prob']
usersim_params['simulator_run_mode'] = params['run_mode']
usersim_params['simulator_act_level'] = params['act_level']
usersim_params['learning_phase'] = params['learning_phase']
usersim_params['hidden_size'] = params['dqn_hidden_size']

# Manually set torch seed to ensure fail comparison.
torch.manual_seed(params['torch_seed'])

if agt == 0:
    agent = AgentCmd(movie_kb, act_set, slot_set, agent_params)
elif agt == 1:
    agent = InformAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 2:
    agent = RequestAllAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 3:
    agent = RandomAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 4:
    agent = EchoAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 5:
    agent = RequestBasicsAgent(movie_kb, act_set, slot_set, agent_params)
elif agt == 6:
    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
elif agt == 7:
    agent = AgentSAC(movie_kb, act_set, slot_set, agent_params)
elif agt == 9:
    agent = AgentTD3(movie_kb, act_set, slot_set, agent_params)
elif agt == 10:
    merged_params = dict(agent_params, **usersim_params)
    agent = AgentDQNZ(movie_kb, movie_dictionary, act_set, slot_set, goal_set, merged_params)

################################################################################
#    Add your agent here
################################################################################
else:
    pass



if usr == 0:  # real user
    user_sim = RealUser(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
elif usr == 1:
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = ModelBasedSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    agent.set_user_planning(world_model)
elif usr == 2:
    user_sim = RuleSimulator(movie_dictionary, act_set, slot_set, goal_set, usersim_params)
    world_model = agent
################################################################################
#    Add your user simulator here
################################################################################
else:
    pass

################################################################################
# load trained NLG model
################################################################################
nlg_model_path = params['nlg_model_path']
diaact_nl_pairs = params['diaact_nl_pairs']
nlg_model = nlg()
nlg_model.load_nlg_model(nlg_model_path)
nlg_model.load_predefine_act_nl_pairs(diaact_nl_pairs)

agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)
world_model.set_nlg_model(nlg_model)

################################################################################
# load trained NLU model
################################################################################
nlu_model_path = params['nlu_model_path']
nlu_model = nlu()
nlu_model.load_nlu_model(nlu_model_path)

agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)
world_model.set_nlu_model(nlu_model)

################################################################################
# Dialog Manager
################################################################################
dialog_manager = DialogManager(agent, user_sim, world_model, act_set, slot_set, movie_kb, params)

################################################################################
#   Run num_episodes Conversation Simulations
################################################################################
status = {'successes': 0, 'count': 0, 'cumulative_reward': 0}

simulation_epoch_size = params['simulation_epoch_size']
batch_size = params['batch_size']  # default = 16
warm_start = params['warm_start']
warm_start_epochs = params['warm_start_epochs']
planning_steps = params['planning_steps'] - 1

agent.planning_steps = planning_steps

success_rate_threshold = params['success_rate_threshold']
save_check_point = params['save_check_point']
print_interval = params['print_interval']

""" Best Model and Performance Records """
best_model = {}
best_res = {'success_rate': 0, 'ave_reward': float('-inf'), 'ave_turns': float('inf'), 'epoch': 0}
best_model['model'] = copy.deepcopy(agent)
best_res['success_rate'] = 0

performance_records = {}
performance_records['success_rate'] = {}
performance_records['ave_turns'] = {}
performance_records['ave_reward'] = {}

""" Save model """
def save_model(path, agt, success_rate, agent, best_epoch, cur_epoch):
    filename = 'agt_%s_%s_%s_%.5f.pkl' % (agt, best_epoch, cur_epoch, success_rate)
    filepath = os.path.join(path, filename)
    try:
        agent.save(filepath)
        print('saved model in %s' % (filepath,))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath,))
        print(e)


""" save performance numbers """
def save_performance_records(path, agt, records):
    filename = 'agt_%s_performance_records.json' % (agt)
    filepath = os.path.join(path, filename)
    try:
        json.dump(records, open(filepath, "w", encoding="utf8"))
        print('saved model in %s' % (filepath,))
    except Exception as e:
        print('Error: Writing model fails: %s' % (filepath,))
        print(e)


""" Run N simulation Dialogues """
def simulation_epoch(simulation_epoch_size):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(simulation_epoch_size):
        dialog_manager.memory_actions['m_agent_actions'] = []
        dialog_manager.memory_actions['m_user_actions'] = []
        dialog_manager.initialize_episode(use_environment=True)
        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn(record_training_data_for_user=False)
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    # print("simulation episode %s: Success" % (episode))
                # else:
                    # print("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    # print("simulation success rate %s, ave reward %s, ave turns %s" % (
    #     res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


def simulation_epoch_for_training(simulation_epoch_size, grounded_for_model=False):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    for episode in range(simulation_epoch_size):

        if episode % simulation_epoch_size == 0:
            use_environment = True
        else:
            use_environment = False

        dialog_manager.memory_actions['m_agent_actions'] = []
        dialog_manager.memory_actions['m_user_actions'] = []
        dialog_manager.initialize_episode(use_environment or grounded_for_model)

        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                    # print("simulation episode %s: Success" % (episode))
                # else:
                    # print("simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

    res['success_rate'] = float(successes) / simulation_epoch_size
    res['ave_reward'] = float(cumulative_reward) / simulation_epoch_size
    res['ave_turns'] = float(cumulative_turns) / simulation_epoch_size
    # print("simulation success rate %s, ave reward %s, ave turns %s" % (
    #     res['success_rate'], res['ave_reward'], res['ave_turns']))
    return res


""" Warm_Start Simulation (by Rule Policy) """
def warm_start_simulation():
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    res = {}
    warm_start_run_epochs = 0
    for episode in range(warm_start_epochs):
        dialog_manager.memory_actions['m_agent_actions']=[]
        dialog_manager.memory_actions['m_user_actions']=[]
        dialog_manager.initialize_episode(use_environment=True)

        episode_over = False
        while (not episode_over):
            episode_over, reward = dialog_manager.next_turn()
            cumulative_reward += reward
            if episode_over:
                if reward > 0:
                    successes += 1
                #     print("warm_start simulation episode %s: Success" % (episode))
                # else:
                #     print("warm_start simulation episode %s: Fail" % (episode))
                cumulative_turns += dialog_manager.state_tracker.turn_count

        warm_start_run_epochs += 1

        # if len(agent.experience_replay_pool) >= agent.experience_replay_pool_size:
        #     break

    if params['boosted']:
        world_model.train(batch_size, num_batches=5)
    else:
        world_model.training_examples = []

    agent.warm_start = 2
    res['success_rate'] = float(successes) / warm_start_run_epochs
    res['ave_reward'] = float(cumulative_reward) / warm_start_run_epochs
    res['ave_turns'] = float(cumulative_turns) / warm_start_run_epochs
    print("Warm_Start %s epochs, success rate %s, ave reward %s, ave turns %s" % (
        episode + 1, res['success_rate'], res['ave_reward'], res['ave_turns']))
    print("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))


def warm_start_simulation_preload():

    agent.experience_replay_pool = pickle.load(open('warm_up_experience_pool_seed3081_r5.pkl', 'rb'))
    world_model.training_examples = pickle.load(open('warm_up_experience_pool_seed3081_r5_user.pkl', 'rb'))
    world_model.train(batch_size, 5)

    agent.warm_start = 2

    print("Current experience replay buffer size %s" % (len(agent.experience_replay_pool)))


def run_episodes(count, status):
    successes = 0
    cumulative_reward = 0
    cumulative_turns = 0

    grounded_for_model = params['grounded']
    simulation_epoch_size = planning_steps + 1

    if (agt == 6 or agt == 9 or agt == 10) and params['trained_model_path'] == None and warm_start == 1:
        print('warm_start starting ...')
        warm_start_simulation()
        print('warm_start finished, start RL training ...')

    for episode in range(count):

        # print("Episode: %s" % (episode))

        agent.predict_mode = False
        dialog_manager.memory_actions['m_agent_actions'] = []
        dialog_manager.memory_actions['m_user_actions'] = []
        dialog_manager.initialize_episode(True)
        episode_over = False

        while not episode_over:
            episode_over, reward = dialog_manager.next_turn(record_training_data_for_user=False)
            cumulative_reward += reward

            if episode_over:
                if reward > 0:
                    # print("Successful Dialog!")
                    successes += 1
                # else:
                    # print("Failed Dialog!")
                cumulative_turns += dialog_manager.state_tracker.turn_count

        # simulation
        if (agt == 6 or agt == 9 or agt == 10) and params['trained_model_path'] == None:
            agent.predict_mode = True
            world_model.predict_mode = True
            # 用世界模型训练策略模型
            simulation_epoch_for_training(simulation_epoch_size, grounded_for_model)

            agent.predict_mode = False
            world_model.predict_mode = False
            # 用基于规则的真实数据训练策略模型
            simulation_res = simulation_epoch(50)

            performance_records['success_rate'][episode] = simulation_res['success_rate']
            performance_records['ave_turns'][episode] = simulation_res['ave_turns']
            performance_records['ave_reward'][episode] = simulation_res['ave_reward']

            if simulation_res['success_rate'] >= best_res['success_rate']:
                if simulation_res['success_rate'] >= success_rate_threshold:  # threshold = 0.30
                    agent.predict_mode = True
                    world_model.predict_mode = True
                    # 再一次用世界模型训练策略网络
                    simulation_epoch_for_training(simulation_epoch_size, grounded_for_model)

            if simulation_res['success_rate'] > best_res['success_rate']:
                best_model['model'] = copy.deepcopy(agent)
                best_res['success_rate'] = simulation_res['success_rate']
                best_res['ave_reward'] = simulation_res['ave_reward']
                best_res['ave_turns'] = simulation_res['ave_turns']
                best_res['epoch'] = episode

            # 回放经验训练策略模型
            agent.train(batch_size, 1, episode, print_interval)
            # agent.train(batch_size, 1)
            agent.reset_dqn_target()

            if params['train_world_model'] and params['planning_steps']:
                # 回放经验训练世界模型
                world_model.train(batch_size, 1, episode, print_interval)
                # world_model.train(batch_size, 1)

            # save the model every 10 episodes
            if episode % save_check_point == 0:  # and params['trained_model_path'] == None: 
                save_model(params['write_model_dir'], agt, best_res['success_rate'], best_model['model'],
                           best_res['epoch'], episode)
                save_performance_records(params['write_model_dir'], agt, performance_records)

            if episode % print_interval == 0:
                print("-"*100, "\n", "Progress: %s / %s, Success rate: %s / %s Avg reward: %.2f Avg turns: %.2f" % (
                    episode + 1, count, successes, episode + 1, float(cumulative_reward) / (episode + 1),
                    float(cumulative_turns) / (episode + 1)))
                print("Simulation success rate %s, Ave reward %s, Ave turns %s, Best success rate %s" % (
                    performance_records['success_rate'][episode], performance_records['ave_reward'][episode],
                    performance_records['ave_turns'][episode], best_res['success_rate']))
        
    print("_"*100, "\n","Success rate: %s / %s (%.2f), Avg reward: %.2f, Avg turns: %.2f" % (
        successes, count, float(successes / count), float(cumulative_reward) / count, float(cumulative_turns) / count))
    
    status['successes'] += successes
    status['count'] += count

    if (agt == 6 or agt == 9 or agt == 10) and params['trained_model_path'] == None:
        save_model(params['write_model_dir'], agt, float(successes) / count, 
                   best_model['model'], best_res['epoch'], count)
        save_performance_records(params['write_model_dir'], agt, performance_records)

run_episodes(num_episodes, status)
