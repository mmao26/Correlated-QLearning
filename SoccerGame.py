### Author Name: Manqing Mao
### GTID: mmao33
### Soccer Game

import copy
import random
from enum import Enum
from collections import defaultdict

from cvxopt.modeling import op
from cvxopt.modeling import variable
from cvxopt.solvers import options


class Actions(Enum):
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4
    STICK = 5

# Soccer players
class Player:
    def __init__(self, idNum, cords, has_ball):
        # ID number, coordinate, has ball or not
        self.id = idNum
        self.cords = cords
        self.has_ball = has_ball

    def __eq__(self, other):
        return self.id == other.id and self.cords == other.cords and self.has_ball == other.has_ball

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self.id, self.cords, self.has_ball))

    def __str__(self):
        return str((self.id, self.cords, self.has_ball))

class SoccerGame:
    def __init__(self, state):
        self.state = copy.deepcopy(state)

    # Both players choose actions simultaneously, however actions are not executed simultaneously. 
    # There is a 50% chance player 1 will go before player 2.
    # a - active player
    # o - opponent, same below
    def apply_actions(self, a, o):
        self.state = random.sample(self.state.get_reachable_states(a, o), 1)[0]

class State:
    def __init__(self, player1, player2):
        self.player1 = player1
        self.player2 = player2

    def get_reachable_states(self, a, o):
        reachable_states = set()

        # A moves first
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)

        if new_cords_a == self.player2.cords:
            if self.player1.has_ball:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
            else:
                tmp_a = Player(1, self.player1.cords, False)
                tmp_b = Player(2, self.player2.cords, True)
        else:
            if self.player1.has_ball:
                tmp_a = Player(1, new_cords_a, True)
                if new_cords_a == new_cords_b:
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
            else:
                if new_cords_a == new_cords_b:
                    tmp_a = Player(1, new_cords_a, True)
                    tmp_b = Player(2, self.player2.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
                    tmp_b = Player(2, new_cords_b, True)

        reachable_states.add(State(tmp_a, tmp_b))

        # B moves first - same as above
        new_cords_a = self.new_player_cords(self.player1, a)
        new_cords_b = self.new_player_cords(self.player2, o)
        if new_cords_b == self.player1.cords:
            if self.player2.has_ball:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
            else:
                tmp_b = Player(2, self.player2.cords, False)
                tmp_a = Player(1, self.player1.cords, True)
        else:
            if self.player2.has_ball:
                tmp_b = Player(2, new_cords_b, True)
                if new_cords_b == new_cords_a:
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_a = Player(1, new_cords_a, False)
            else:
                if new_cords_b == new_cords_a:
                    tmp_b = Player(2, new_cords_b, True)
                    tmp_a = Player(1, self.player1.cords, False)
                else:
                    tmp_b = Player(2, new_cords_b, False)
                    tmp_a = Player(1, new_cords_a, True)

        reachable_states.add(State(tmp_a, tmp_b))

        return reachable_states

    def __eq__(self, other):
        return self.player1 == other.player1 and self.player2 == other.player2

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return hash((self.player1, self.player2))

    def __str__(self):
        return str((str(self.player1), str(self.player2)))

    def reward_value(self):
        # Dimensions coming from paper
        if self.player1.has_ball:
            x, y = self.player1.cords
            if x == 0:
                return 100
            elif x == 3:
                return -100
            else:
                return 0
        elif self.player2.has_ball:
            x, y = self.player2.cords
            if x == 0:
                return 100
            elif x == 3:
                return -100
            else:
                return 0

    @staticmethod
    def new_player_cords(player, action):
        x, y = player.cords
        if action == Actions.NORTH:
            y = max(0, y - 1)
        elif action == Actions.SOUTH:
            y = min(1, y + 1)
        elif action == Actions.WEST:
            x = max(0, x - 1)
        elif action == Actions.EAST:
            x = min(3, x + 1)
        return x, y

# Four learning algorithms
class Solver:
    def __init__(self):
        # Actions players can take
        self.actions = [Actions.NORTH, Actions.SOUTH, Actions.EAST, Actions.WEST, Actions.STICK]

        # Initial state
        self.init_state = State(Player(1, (3, 0), False),
                                Player(2, (1, 0), True))

        # State for gathering Q-value differences
        self.q_stat_state = State(Player(1, (2, 0), False),
                                  Player(2, (1, 0), True))

        # V tables for players A and B initialized to 1
        self.V1 = defaultdict(lambda: 1)
        self.V2 = defaultdict(lambda: 1)

    # Normal Q-learning algorithm
    def q_learning(self, num_steps, alpha, gamma):
        # State action pair to gather q-value differences from
        q_stat = (self.q_stat_state, Actions.SOUTH)

        # List of gathered statistics
        statistics = list()

        # Q-table
        Q = defaultdict(lambda: 1)

        # Time step counter
        time_step_counter = 0

        # Soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < num_steps:
            # Restart game if ended already.
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # Current state
            cur_state = copy.deepcopy(game.state)

            # Select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # Apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # Get reward
            current_reward = game.state.reward_value()

            # If reward not zero, game is now over.
            if current_reward != 0:
                game_over = True

            # Set value of new state
            self.V1[game.state] = max(Q[(game.state, Actions.NORTH)],
                                      Q[(game.state, Actions.SOUTH)],
                                      Q[(game.state, Actions.EAST)],
                                      Q[(game.state, Actions.WEST)],
                                      Q[(game.state, Actions.STICK)])

            # previous q-value
            pre_q = Q[(cur_state, a)]

            # q-update
            Q[(cur_state, a)] = (1 - alpha) * Q[(cur_state, a)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post update q-value
            post_q = Q[(cur_state, a)]

            # record stats if in correct state action pair
            if (cur_state, a) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format is (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    # Assumes the other player is an ally and will always attempt to help.
    def friend_q_learning(self, num_steps, alpha, gamma):
        # state joint action pair to record q-diff's
        q_stat = (self.q_stat_state, Actions.SOUTH, Actions.STICK)

        # List of gathered statistics
        statistics = list()

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start soccer game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < num_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # take actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # get current reward
            current_reward = game.state.reward_value()

            # if not 0, game is over
            if current_reward != 0:
                game_over = True

            # get max q-value
            max_q_value = Q[(game.state, Actions.NORTH, Actions.NORTH)]
            for p1_a in self.actions:
                for p2_o in self.actions:
                    max_q_value = max(max_q_value, Q[(game.state, p1_a, p2_o)])

            # update value of state
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # q-value update
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # record q-diff's
            if (cur_state, a, o) == q_stat:
                statistics.append(
                    (time_step_counter, abs(post_q - pre_q), pre_q, post_q))

            # decay alpha
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val)
        return statistics

    # Assumes the other player will always attempt to minimize its reward
    def foe_q_learning(self, num_steps, alpha, gamma):
        # turn lp solver logging off
        options['show_progress'] = False

        # state joint action pair to record q-diffs in
        q_stat = (self.q_stat_state, Actions.SOUTH, Actions.STICK)

        # statistics for recording
        statistics = list()

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < num_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # action probabilities
            probs = list()
            for i in range(len(self.actions)):
                probs.append(1./len(self.actions))

            # for i in range(len(self.actions)):
            #     constrs.append((probs[i] >= 0))

            # # sum of probabilities = 1 constraint
            # total_prob = sum(probs)
            # constrs.append((total_prob == 1))

            # objective
            v = variable()
            # constraints
            constrs = list()

            # set mini-max constraints
            for j in range(5):
                c = 0
                for i in range(5):
                    c += Q[(game.state, self.actions[i], self.actions[j])] * probs[i]
                constrs.append((c >= v))

            # maximize objective
            # op will minimize v, so we use -v
            lp = op(-v, constrs)
            lp.solve()

            # set value of state
            max_q_value = v.value[0]
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # update q-value
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # gather statistics
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for i in range(len(probs)):
                    prob_list.append(probs[i])
                # print prob_list
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha from Littman's paper
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics

    # Identical to foe-q learning
    def ce_q_learning(self, num_steps, alpha, gamma):
        # turn lp solver logging off
        options['show_progress'] = False

        # state joint action pair to record q-diffs in
        q_stat = (self.q_stat_state, Actions.SOUTH, Actions.STICK)

        # statistics for recording
        statistics = list()

        # Q-table
        Q = defaultdict(lambda: 1)

        # time-step counter
        time_step_counter = 0

        # Start game
        game = SoccerGame(self.init_state)
        game_over = False

        while time_step_counter < num_steps:
            # restart game if needed
            if game_over:
                game = SoccerGame(self.init_state)
                game_over = False

            # current state
            cur_state = copy.deepcopy(game.state)

            # select actions
            a = random.choice(self.actions)
            o = random.choice(self.actions)

            # apply actions
            game.apply_actions(a, o)
            time_step_counter += 1

            # get current reward
            current_reward = game.state.reward_value()
            if current_reward != 0:
                game_over = True

            # action probabilities
            probs = list()
            for i in range(len(self.actions)):
                probs.append(1./len(self.actions))

            # for i in range(len(self.actions)):
            #     constrs.append((probs[i] >= 0))

            # # sum of probabilities = 1 constraint
            # total_prob = sum(probs)
            # constrs.append((total_prob == 1))

            # objective
            v = variable()
            # constraints
            constrs = list()

            # set mini-max constraints
            for j in range(5):
                c = 0
                for i in range(5):
                    c += Q[(game.state, self.actions[i], self.actions[j])] * probs[i]
                constrs.append((c >= v))

            # maximize objective
            # op will minimize v, so we use -v
            lp = op(-v, constrs)
            lp.solve()

            # set value of state
            max_q_value = v.value[0]
            self.V1[game.state] = max_q_value

            # pre q-value
            pre_q = Q[(cur_state, a, o)]

            # update q-value
            Q[(cur_state, a, o)] = (1 - alpha) * Q[
                (cur_state, a, o)] + alpha * (
                current_reward + gamma * self.V1[game.state])

            # post q-value
            post_q = Q[(cur_state, a, o)]

            # gather statistics
            if (cur_state, a, o) == q_stat:
                prob_list = list()
                for i in range(len(probs)):
                    prob_list.append(probs[i])
                # print prob_list
                statistics.append((
                                  time_step_counter, abs(post_q - pre_q), pre_q,
                                  post_q, prob_list))

            # decay alpha from Littman's paper
            alpha = max(0.001, alpha * 0.999995)

        # format (time-step, q-diff, pre-q-val, post-q-val, probabilities)
        return statistics
        
if __name__ == "__main__":		
	# Parameters recommended by paper and discussions from piazza
	gamma = 0.9
	alpha = 0.2
	num_steps = 10

	# Normal Q-learning test
	solver = Solver()
	stats = solver.q_learning(num_steps, alpha, gamma)

	file = open('q-learning.csv', 'w')
	for ts, q_diff, pre_q, post_q in stats:
	    file.write('{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q))

	# Friend-Q test
	solver = Solver()
	stats = solver.friend_q_learning(num_steps, alpha, gamma)

	file = open('friend-q.csv', 'w')
	for ts, q_diff, pre_q, post_q in stats:
	    file.write('{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q))

	# Foe-Q test
	solver = Solver()
	stats = solver.foe_q_learning(num_steps, alpha, gamma)

	file = open('foe-q.csv', 'w')
	for ts, q_diff, pre_q, post_q, probs in stats:
	    file.write('{},{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q, probs))

	# CE-Q test
	solver = Solver()
	stats = solver.ce_q_learning(num_steps, alpha, gamma)

	file = open('ce-q.csv', 'w')
	for ts, q_diff, pre_q, post_q, probs in stats:
	    file.write('{},{},{},{},{}\n'.format(ts, q_diff, pre_q, post_q, probs))
