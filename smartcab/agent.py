import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import math
import operator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        self.Q_TABLE = {}
        self.alpha = 0.4
        self.gamma = 0.2
        self.eps = 0.3
        self.reached_destination = 0
        self.positive_reward = 0
        self.negative_reward = 0
        self.trial = 0
        self.trial_reach = {}
        self.trial_not_reach = {}
        self.action_to_waypoint = []


    def reset(self, destination=None):
        if self.trial > 89:
            if self.trial not in self.trial_reach.keys():
                self.trial_not_reach[self.trial] = False
        self.planner.route_to(destination)
        self.trial += 1


    def get_Action(self):
        # set Actions to 0 if there is no entry in Q_TABLE
        if self.state not in self.Q_TABLE.keys():

            self.Q_TABLE[self.state] = {'forward': 1, 'left': 1, 'right': 1, None: 1}

            choice = random.choice(self.Q_TABLE[self.state].keys())
        # choose Action with highest value
        else:
            state = self.Q_TABLE[self.state]
            choice = max(state.iteritems(), key=operator.itemgetter(1))[0]
            p = self.boltzman(choice)
            if p < 1 - self.eps:
                return random.choice(self.Q_TABLE[self.state].keys())
        return choice

    def choose_max_Q_Value(self, state):
        # return max Q value of a state
        if state not in self.Q_TABLE.keys():
            self.Q_TABLE[state] = {'forward': 1, 'left': 1, 'right': 1, None:1}
        return max(self.Q_TABLE[state].values())


    def boltzman(self, action):
        if self.state in self.Q_TABLE.keys():
            sum = 0
            for i in self.Q_TABLE[self.state].values():
                sum += math.exp(i)

            val = math.exp(self.Q_TABLE[self.state][action])

            p = (val/0.5)/(sum/0.5)
            return p

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        s = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)
        self.state = s


        actions = ['left', 'right', 'forward']
        #a = random.choice(actions)
        a = self.get_Action()

        if a != self.next_waypoint:
            if inputs['light'] != None or inputs['oncoming'] != None or inputs['right'] != None or inputs['left'] != None:
                self.action_to_waypoint.append(str(self.trial) + ";" +str(inputs['light']) + ";"+ str(inputs['oncoming']) + ";"+ str(inputs['right']) + ";" + str(inputs['left'])+ ";"+str(self.next_waypoint)+ ";" +str(a))
        #a = self.next_waypoint
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        s_next = (inputs['light'], inputs['oncoming'], inputs['right'], inputs['left'], self.next_waypoint)

        q_max = self.choose_max_Q_Value(s_next)
        r = self.env.act(self, a)

        if r > 0:
            self.positive_reward += r
        else:
            self.negative_reward += r
        if r == 12:
            self.reached_destination += 1
            self.trial_reach[self.trial] = True

        self.Q_TABLE[s][a] = (1- self.alpha) * self.Q_TABLE[s][a] + self.alpha * (r + self.gamma * q_max)

        #print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, a, r)  # [debug]

    def sumUp(self):
        print "Positive Reward: " + str(self.positive_reward)
        print "Negative Reward: " + str(self.negative_reward)
        print "Total Reward: " + str((self.positive_reward + abs(self.negative_reward)))
        print "Reached destination: " + str(self.reached_destination)
        print "Alpha: " + str(self.alpha)
        print "Gamma: " + str(self.gamma)
        print "Epsilon: " + str(self.eps)
        #print "Trial: " + str(self.trial)
        #print "Trials not reached: " + str(self.trial_not_reach)
        #print "\n".join(self.action_to_waypoint)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent

    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified num ber of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    a.sumUp()

if __name__ == '__main__':
    run()
