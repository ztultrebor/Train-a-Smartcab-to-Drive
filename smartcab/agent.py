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
        action = self.env.valid_actions[random.randrange(4)]
        reward = self.env.act(self, action)
        print "LearningAgent.update():deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

#==============================================================================

class DimAwarenessAgent(StochasticAgent):
    """
    An agent that is aware of the state its in, but is not able to reason and
    plan based upon this knowledge. Satisfies the second part of the assignment.
    """

    def __init__(self, env):
        super(DimAwarenessAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)
        action = self.env.valid_actions[random.randrange(4)]
        # Execute action and get reward
        reward = self.env.act(self, action)
        print "LearningAgent.update():deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

#==============================================================================

class QLearningAgent(StochasticAgent):
    """
    An agent that attempts to learn what actions are best under various circumstances. Satisfies the third part of the assignment.
    """

    def __init__(self, env):
        super(QLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.states = [ (light, oncoming, left, waypt)
                for light in ('red', 'green')
                for oncoming in ('forward','left', 'right', None)
                for left in ('forward','left', 'right', None)
                for waypt in ('forward','left', 'right') ]
        self.Q = { (state, action) : 0
                for state in self.states
                for action in self.env.valid_actions }
        self.policy = { state : 'forward'
                for state in self.states }
        self.gamma = 0.9
        self.alpha = 0.1
        self.policy_needs_updated = False

    def __update_Q__(self):
        # Determine the highest attainable Q-value for the current state
        Q_prime = max([self.Q[self.state, a]
                        for a in ('forward','left', 'right', None)])
        # Update the Q-value for the previous state and action
        self.Q[self.previous_state, self.action] = (
                    (1 - self.alpha) * self.Q[self.previous_state, self.action]
                    + self.alpha * (self.reward + self.gamma * Q_prime)
                    )

    def __update_policy__(self):
        self.__update_Q__()
        self.policy[self.previous_state] = max(
                                [(a, self.Q[self.previous_state, a])
                                for a in ('forward','left', 'right', None)],
                                key= lambda x: x[1]
                                )[0]

    def __choose_action__(self):
        return self.policy[self.state]

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'],
                        self.next_waypoint)
        # Update the policy for the last round's state and action
        if self.policy_needs_updated:
            self.__update_policy__()
        # Execute action and get reward
        self.action = self.__choose_action__()
        self.reward = self.env.act(self, self.action)
        self.previous_state = self.state
        self.policy_needs_updated = True
        print "LearningAgent.update():deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, self.action, self.reward)  # [debug]

#==============================================================================

class TrueLearningAgent(QLearningAgent):
    """
    An agent that does an admirable job of learning what actions are best under
    various circumstances. This is done by introducing a penalty factor applied
    to each Q-Learning iteration, that gives most actions an effective negative
    reward. Satisfies the fourth part of the assignment.
    """

    def __init__(self, env):
        super(TrueLearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color

    def __update_Q__(self):
        p = penalty = 1.25
        Q_prime = max([self.Q[self.state, ax]
                        for ax in ('forward','left', 'right', None)])
        self.Q[self.previous_state, self.action] = (
                    (1 - self.alpha) * self.Q[self.previous_state, self.action]
                    + self.alpha * (self.reward - p + self.gamma * Q_prime)
                    )

#==============================================================================

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
    run(StochasticAgent, enforce=False, delay=0.5)
    run(DimAwarenessAgent, enforce=False, delay=0.5)
    run(QLearningAgent, enforce=True, delay=0.5)
    run(TrueLearningAgent, enforce=True, delay=0.5)
