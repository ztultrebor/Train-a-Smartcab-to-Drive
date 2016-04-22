import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

# =============================================================================

class StochasticAgent(Agent):
    """
    An agent that makes no attempt to learn. It just executes some random
    action at every intersection. Satisfies the first part of the assignment.
    """

    def __init__(self, env):
        super(StochasticAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route
        # planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        # Execute random action and get reward
        action = (None, 'forward', 'right', 'left')[random.randrange(4)]
        reward = self.env.act(self, action)
        print "LearningAgent.update():deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'black'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state

        # TODO: Select action according to your policy
        x = random.randrange(4)
        #action = (None, 'forward', 'right', 'left')[x]
        action = self.next_waypoint
        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update():deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run(agent, enforce, delay):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(agent)  # create agent
    e.set_primary_agent(a, enforce_deadline=enforce)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=delay)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit



if __name__ == '__main__':
    run(StochasticAgent, enforce=False, delay=0.2)
