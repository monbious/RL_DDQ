'''
An DQN Agent modified for DDQ Agent

Some methods are not consistent with super class Agent.
'''

import random, copy, json
import pickle
import numpy as np
from collections import namedtuple, deque

from deep_dialog import dialog_config

from .agent import Agent
from deep_dialog.qlearning import DQNZ

import torch
import torch.optim as optim
import torch.nn.functional as F

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'term'))


class AgentDQNZ(Agent):
    def __init__(self, movie_dict=None, movie_dictionary=None, act_set=None, slot_set=None, start_set=None, params=None):
        self.movie_dict = movie_dict
        self.movie_dictionary = movie_dictionary
        self.act_set = act_set
        self.slot_set = slot_set
        self.start_set = start_set
        self.act_cardinality = len(act_set.keys())
        self.slot_cardinality = len(slot_set.keys())

        self.feasible_actions_users = dialog_config.feasible_actions_users
        self.feasible_actions = dialog_config.feasible_actions
        self.num_actions = len(self.feasible_actions)

        self.epsilon = params['epsilon']
        self.agent_run_mode = params['agent_run_mode']
        self.agent_act_level = params['agent_act_level']

        self.experience_replay_pool_size = params.get('experience_replay_pool_size', 5000)
        self.experience_replay_pool = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.experience_replay_pool_from_model = deque(
            maxlen=self.experience_replay_pool_size)  # experience replay pool <s_t, a_t, r_t, s_t+1>
        self.running_expereince_pool = None # hold experience from both user and world model

        self.hidden_size = params.get('dqn_hidden_size', 60)
        self.gamma = params.get('gamma', 0.9)
        self.predict_mode = params.get('predict_mode', False)
        self.warm_start = params.get('warm_start', 0)

        self.max_turn = params['max_turn'] + 5
        self.state_dimension = 2 * self.act_cardinality + 7 * self.slot_cardinality + 3 + self.max_turn

        self.slot_err_probability = params['slot_err_probability']
        self.slot_err_mode = params['slot_err_mode']
        self.intent_err_probability = params['intent_err_probability']
        self.simulator_run_mode = params['simulator_run_mode']
        self.simulator_act_level = params['simulator_act_level']
        self.learning_phase = params['learning_phase']

        self.dqnz = DQNZ(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        self.target_dqnz = DQNZ(self.state_dimension, self.hidden_size, self.num_actions).to(DEVICE)
        self.target_dqnz.load_state_dict(self.dqnz.state_dict())
        self.target_dqnz.eval()

        self.optimizer = optim.RMSprop(self.dqnz.parameters(), lr=1e-3)

        self.cur_bellman_err = 0

        # Prediction Mode: load trained DQN model
        if params['trained_model_path'] != None:
            self.load(params['trained_model_path'])
            self.predict_mode = True
            self.warm_start = 2

    def initialize_episode(self):
        """ Initialize a new episode. This function is called every time a new episode is run. """

        self.current_slot_id = 0
        self.phase = 0
        self.request_set = ['moviename', 'starttime', 'city', 'date', 'theater', 'numberofpeople']

        self.state = {}
        self.state['history_slots'] = {}
        self.state['inform_slots'] = {}
        self.state['request_slots'] = {}
        self.state['rest_slots'] = []
        self.state['turn'] = 0

        self.episode_over = False
        self.dialog_status = dialog_config.NO_OUTCOME_YET

        self.goal = self._sample_goal(self.start_set)
        self.goal['request_slots']['ticket'] = 'UNK'
        self.constraint_check = dialog_config.CONSTRAINT_CHECK_FAILURE

        # sample first action
        user_action = self._sample_action()
        assert (self.episode_over != 1), ' but we just started'
        return user_action

    def mcts_state_to_action(self, mcts_state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.mcts_representation = self.prepare_state_representation(mcts_state)
        self.mcts_action, mcts_value, term = self.mcts_run_policy(self.mcts_representation)

        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.mcts_action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.mcts_action[0]])
            self.mcts_action = self.mcts_action[0]

        mcts_action = {'act_slot_response': act_slot_response, 'act_slot_value_response': None}
        if mcts_state['turn'] >= self.max_turn-1:
            term = 1

        return mcts_action, mcts_value, term, self.mcts_action


    def state_to_action(self, state):
        """ DQN: Input state, output action """
        # self.state['turn'] += 2
        self.representation = self.prepare_state_representation(state)
        self.action = self.run_policy(self.representation)

        if self.warm_start == 1:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action])
        else:
            act_slot_response = copy.deepcopy(self.feasible_actions[self.action[0]])

        return {'act_slot_response': act_slot_response, 'act_slot_value_response': None}

    def prepare_state_representation(self, state):
        """ Create the representation for each state """

        user_action = state['user_action']
        current_slots = state['current_slots']
        kb_results_dict = state['kb_results_dict']
        agent_last = state['agent_action']

        ########################################################################
        #   Create one-hot of acts to represent the current user action
        ########################################################################
        user_act_rep = np.zeros((1, self.act_cardinality))
        user_act_rep[0, self.act_set[user_action['diaact']]] = 1.0

        ########################################################################
        #     Create bag of inform slots representation to represent the current user action
        ########################################################################
        user_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['inform_slots'].keys():
            user_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Create bag of request slots representation to represent the current user action
        ########################################################################
        user_request_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in user_action['request_slots'].keys():
            user_request_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Creat bag of filled_in slots based on the current_slots
        ########################################################################
        current_slots_rep = np.zeros((1, self.slot_cardinality))
        for slot in current_slots['inform_slots']:
            current_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent act
        ########################################################################
        agent_act_rep = np.zeros((1, self.act_cardinality))
        if agent_last:
            agent_act_rep[0, self.act_set[agent_last['diaact']]] = 1.0

        ########################################################################
        #   Encode last agent inform slots
        ########################################################################
        agent_inform_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['inform_slots'].keys():
                agent_inform_slots_rep[0, self.slot_set[slot]] = 1.0

        ########################################################################
        #   Encode last agent request slots
        ########################################################################
        agent_request_slots_rep = np.zeros((1, self.slot_cardinality))
        if agent_last:
            for slot in agent_last['request_slots'].keys():
                agent_request_slots_rep[0, self.slot_set[slot]] = 1.0

        # turn_rep = np.zeros((1,1)) + state['turn'] / 10.
        turn_rep = np.zeros((1, 1))

        ########################################################################
        #  One-hot representation of the turn count?
        ########################################################################
        turn_onehot_rep = np.zeros((1, self.max_turn))
        turn_onehot_rep[0, state['turn']] = 1.0

        # ########################################################################
        # #   Representation of KB results (scaled counts)
        # ########################################################################
        # kb_count_rep = np.zeros((1, self.slot_cardinality + 1)) + kb_results_dict['matching_all_constraints'] / 100.
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_count_rep[0, self.slot_set[slot]] = kb_results_dict[slot] / 100.
        #
        # ########################################################################
        # #   Representation of KB results (binary)
        # ########################################################################
        # kb_binary_rep = np.zeros((1, self.slot_cardinality + 1)) + np.sum( kb_results_dict['matching_all_constraints'] > 0.)
        # for slot in kb_results_dict:
        #     if slot in self.slot_set:
        #         kb_binary_rep[0, self.slot_set[slot]] = np.sum( kb_results_dict[slot] > 0.)

        kb_count_rep = np.zeros((1, self.slot_cardinality + 1))

        ########################################################################
        #   Representation of KB results (binary)
        ########################################################################
        kb_binary_rep = np.zeros((1, self.slot_cardinality + 1))

        self.final_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep, kb_count_rep])
        return self.final_representation

    def mcts_run_policy(self, mcts_representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return torch.IntTensor([random.randint(0, self.num_actions - 1)]), 0, 0.5
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy(), 0.5, 0.5
            else:
                return self.mcts_DQN_policy(mcts_representation)

    def run_policy(self, representation):
        """ epsilon-greedy policy """

        if random.random() < self.epsilon:
            return torch.IntTensor([random.randint(0, self.num_actions - 1)])
        else:
            if self.warm_start == 1:
                if len(self.experience_replay_pool) > self.experience_replay_pool_size:
                    self.warm_start = 2
                return self.rule_policy()
            else:
                return self.DQN_policy(representation)

    def rule_policy(self):
        """ Rule Policy """

        act_slot_response = {}

        if self.current_slot_id < len(self.request_set):
            slot = self.request_set[self.current_slot_id]
            self.current_slot_id += 1

            act_slot_response = {}
            act_slot_response['diaact'] = "request"
            act_slot_response['inform_slots'] = {}
            act_slot_response['request_slots'] = {slot: "UNK"}
        elif self.phase == 0:
            act_slot_response = {'diaact': "inform", 'inform_slots': {'taskcomplete': "PLACEHOLDER"},
                                 'request_slots': {}}
            self.phase += 1
        elif self.phase == 1:
            act_slot_response = {'diaact': "thanks", 'inform_slots': {}, 'request_slots': {}}

        return self.action_index(act_slot_response)

    def mcts_DQN_policy(self, mcts_state_representation):
        """ Return action from DQN"""

        with torch.no_grad():
            action, reward, qvalue, term = self.dqnz.predict(torch.FloatTensor(mcts_state_representation))
        return action, reward, term

    def DQN_policy(self, state_representation):
        """ Return action from DQN"""

        with torch.no_grad():
            action, _, _, _ = self.dqnz.predict(torch.FloatTensor(state_representation))
        return action

    def action_index(self, act_slot_response):
        """ Return the index of action """

        for (i, action) in enumerate(self.feasible_actions):
            if act_slot_response == action:
                return i
        print(act_slot_response)
        raise Exception("action index not found")
        return None

    def register_experience_replay_tuple(self, s_t, a_t, reward, s_tplus1, episode_over, st_user, from_model=False):
        """ Register feedback from either environment or world model, to be stored as future training data """

        state_t_rep = self.prepare_state_representation(s_t)
        action_t = self.action
        reward_t = reward
        state_tplus1_rep = self.prepare_state_representation(s_tplus1)
        st_user = self.prepare_state_representation(s_tplus1)
        training_example = (state_t_rep, action_t, reward_t, state_tplus1_rep, episode_over, st_user)

        if self.predict_mode == False:  # Training Mode
            if self.warm_start == 1:
                self.experience_replay_pool.append(training_example)
        else:  # Prediction Mode
            if not from_model:
                self.experience_replay_pool.append(training_example)
            else:
                self.experience_replay_pool_from_model.append(training_example)

    def sample_from_buffer(self, batch_size):
        """Sample batch size examples from experience buffer and convert it to torch readable format"""
        # type: (int, ) -> Transition

        batch = [random.choice(self.running_expereince_pool) for i in range(batch_size)]
        np_batch = []
        for x in range(len(Transition._fields)):
            v = []
            for i in range(batch_size):
                v.append(batch[i][x])
            np_batch.append(np.vstack(v))

        return Transition(*np_batch)

    def train(self, batch_size=1, num_batches=100, episode=1, print_interval=1):
        """ Train DQN with experience buffer that comes from both user and world model interaction."""
        if episode > 100: self.epsilon = 0

        self.cur_bellman_err = 0.
        self.cur_bellman_err_planning = 0.
        self.running_expereince_pool = list(self.experience_replay_pool) + list(self.experience_replay_pool_from_model)

        for iter_batch in range(num_batches):
            for iter in range(int(len(self.running_expereince_pool) / batch_size)):
                self.optimizer.zero_grad()
                batch = self.sample_from_buffer(batch_size)

                # TODO
                action, _, _, _ = self.dqnz(torch.FloatTensor(batch.state))
                state_value = action.gather(1, torch.tensor(batch.action, dtype=torch.int64))
                next_action, _, _, _ = self.target_dqnz(torch.FloatTensor(batch.next_state))
                next_state_value, _ = next_action.max(1)
                next_state_value = next_state_value.unsqueeze(1)

                term = np.asarray(batch.term, dtype=np.float32)
                expected_value = torch.FloatTensor(batch.reward) + self.gamma * next_state_value * (
                    1 - torch.FloatTensor(term))

                loss = F.mse_loss(state_value, expected_value)
                loss.backward()
                self.optimizer.step()
                self.cur_bellman_err += loss.item()

            if len(self.experience_replay_pool) != 0 and (episode % print_interval == 0):
                print(
                    "cur bellman err %.4f, experience replay pool %s, model replay pool %s, cur bellman err for planning %.4f" % (
                        float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
                        len(self.experience_replay_pool), len(self.experience_replay_pool_from_model),
                        self.cur_bellman_err_planning))

    # def train_one_iter(self, batch_size=1, num_batches=100, planning=False):
    #     """ Train DQN with experience replay """
    #     self.cur_bellman_err = 0
    #     self.cur_bellman_err_planning = 0
    #     running_expereince_pool = self.experience_replay_pool + self.experience_replay_pool_from_model
    #     for iter_batch in range(num_batches):
    #         batch = [random.choice(self.experience_replay_pool) for i in range(batch_size)]
    #         np_batch = []
    #         for x in range(5):
    #             v = []
    #             for i in range(len(batch)):
    #                 v.append(batch[i][x])
    #             np_batch.append(np.vstack(v))
    #
    #         batch_struct = self.dqn.singleBatch(np_batch)
    #         self.cur_bellman_err += batch_struct['cost']['total_cost']
    #         if planning:
    #             plan_step = 3
    #             for _ in range(plan_step):
    #                 batch_planning = [random.choice(self.experience_replay_pool) for i in
    #                                   range(batch_size)]
    #                 np_batch_planning = []
    #                 for x in range(5):
    #                     v = []
    #                     for i in range(len(batch_planning)):
    #                         v.append(batch_planning[i][x])
    #                     np_batch_planning.append(np.vstack(v))
    #
    #                 s_tp1, r, t = self.user_planning.predict(np_batch_planning[0], np_batch_planning[1])
    #                 s_tp1[np.where(s_tp1 >= 0.5)] = 1
    #                 s_tp1[np.where(s_tp1 <= 0.5)] = 0
    #
    #                 t[np.where(t >= 0.5)] = 1
    #
    #                 np_batch_planning[2] = r
    #                 np_batch_planning[3] = s_tp1
    #                 np_batch_planning[4] = t
    #
    #                 batch_struct = self.dqn.singleBatch(np_batch_planning)
    #                 self.cur_bellman_err_planning += batch_struct['cost']['total_cost']
    #
    #     if len(self.experience_replay_pool) != 0:
    #         print("cur bellman err %.4f, experience replay pool %s, cur bellman err for planning %.4f" % (
    #             float(self.cur_bellman_err) / (len(self.experience_replay_pool) / (float(batch_size))),
    #             len(self.experience_replay_pool), self.cur_bellman_err_planning))

    ################################################################################
    #    Debug Functions
    ################################################################################
    def save_experience_replay_to_file(self, path):
        """ Save the experience replay pool to a file """

        try:
            pickle.dump(self.experience_replay_pool, open(path, "wb"))
            print('saved model in %s' % (path,))
        except Exception as e:
            print('Error: Writing model fails: %s' % (path,))
            print(e)

    def load_experience_replay_from_file(self, path):
        """ Load the experience replay pool from a file"""

        self.experience_replay_pool = pickle.load(open(path, 'rb'))

    def load_trained_DQN(self, path):
        """ Load the trained DQN from a file """

        trained_file = pickle.load(open(path, 'rb'))
        model = trained_file['model']
        print("Trained DQN Parameters:", json.dumps(trained_file['params'], indent=2))
        return model

    def set_user_planning(self, user_planning):
        self.user_planning = user_planning

    def save(self, filename):
        torch.save(self.dqnz.state_dict(), filename)

    def load(self, filename):
        self.dqnz.load_state_dict(torch.load(filename))

    def reset_dqn_target(self):
        self.target_dqnz.load_state_dict(self.dqnz.state_dict())

    def next(self, s, a):
        """
        Provide
        :param s: state representation from tracker
        :param a: last action from agent
        :return: next user action, termination and reward predicted by world model
        """

        self.state['turn'] += 2
        if (self.max_turn > 0 and self.state['turn'] >= self.max_turn-1):
            reward = - self.max_turn
            term = True
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()
            self.state['diaact'] = "closing"
            response_action = {}
            response_action['diaact'] = self.state['diaact']
            response_action['inform_slots'] = self.state['inform_slots']
            response_action['request_slots'] = self.state['request_slots']
            response_action['turn'] = self.state['turn']
            return response_action, term, reward

        s = self.prepare_state_representation(s)

        # g = self.prepare_user_goal_representation(self.sample_goal)
        # s = np.hstack([s, g])
        try:
            action, reward, qvalue, term = self.predict(torch.FloatTensor(s), torch.LongTensor(np.asarray(a)[:, None]))
        except Exception as e:
            # print(e)
            action, reward, qvalue, term = self.predict(torch.FloatTensor(s), torch.LongTensor(np.asarray([a])[:, None]))
        action = action.item()
        reward = reward.item()
        term = term.item()
        action = copy.deepcopy(self.feasible_actions_users[action])

        if action['diaact'] == 'inform':
            if len(action['inform_slots'].keys()) > 0:
                slots = list(action['inform_slots'].keys())[0]
                if slots in self.sample_goal['inform_slots'].keys():
                    action['inform_slots'][slots] = self.sample_goal['inform_slots'][slots]
                else:
                    action['inform_slots'][slots] = dialog_config.I_DO_NOT_CARE

        response_action = action

        term = term > 0.5

        if reward > 1:
            reward = 2 * self.max_turn
        elif reward < -1:
            reward = -self.max_turn
        else:
            reward = -1

        return response_action, term, reward

    def predict(self, s, a):
        return self.dqnz.predict(s)

    def _sample_action(self):
        """ randomly sample a start action based on user goal """

        self.state['diaact'] = random.choice(list(dialog_config.start_dia_acts.keys()))

        # "sample" informed slots
        if len(self.goal['inform_slots']) > 0:
            known_slot = random.choice(list(self.goal['inform_slots'].keys()))
            self.state['inform_slots'][known_slot] = self.goal['inform_slots'][known_slot]

            if 'moviename' in self.goal['inform_slots'].keys():  # 'moviename' must appear in the first user turn
                self.state['inform_slots']['moviename'] = self.goal['inform_slots']['moviename']

            for slot in self.goal['inform_slots'].keys():
                if known_slot == slot or slot == 'moviename': continue
                self.state['rest_slots'].append(slot)

        self.state['rest_slots'].extend(self.goal['request_slots'].keys())

        # "sample" a requested slot
        request_slot_set = list(self.goal['request_slots'].keys())
        request_slot_set.remove('ticket')
        if len(request_slot_set) > 0:
            request_slot = random.choice(request_slot_set)
        else:
            request_slot = 'ticket'
        self.state['request_slots'][request_slot] = 'UNK'

        if len(self.state['request_slots']) == 0:
            self.state['diaact'] = 'inform'

        if (self.state['diaact'] in ['thanks', 'closing']):
            self.episode_over = True  # episode_over = True
        else:
            self.episode_over = False  # episode_over = False

        sample_action = {}
        sample_action['diaact'] = self.state['diaact']
        sample_action['inform_slots'] = self.state['inform_slots']
        sample_action['request_slots'] = self.state['request_slots']
        sample_action['turn'] = self.state['turn']

        self.add_nl_to_action(sample_action)
        return sample_action

    def _sample_goal(self, goal_set):
        """ sample a user goal  """

        self.sample_goal = random.choice(self.start_set[self.learning_phase])
        return self.sample_goal

    def prepare_user_goal_representation(self, user_goal):
        """"""

        request_slots_rep = np.zeros((1, self.slot_cardinality))
        inform_slots_rep = np.zeros((1, self.slot_cardinality))
        for s in user_goal['request_slots']:
            s = s.strip()
            request_slots_rep[0, self.slot_set[s]] = 1
        for s in user_goal['inform_slots']:
            s = s.strip()
            inform_slots_rep[0, self.slot_set[s]] = 1
        self.user_goal_representation = np.hstack([request_slots_rep, inform_slots_rep])

        return self.user_goal_representation
