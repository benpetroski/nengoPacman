# pacmanAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman import Directions
from game import Agent
import random
import game
import util


# NENGO & NUMPY STUFF
import nengo                                # neural network simulation library
from nengo.utils.functions import piecewise # useful for defining a piecewise function of time
import numpy as np                          # a scientific computing library
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
# from matplotlib import pyplot as plt        # a plotting library
# from mpl_toolkits.mplot3d import Axes3D     # necessary for generating 3D plots
# plt.rcParams['axes.labelsize'] = 20         # set default plot axes label font size
# plt.rcParams['axes.titlesize'] = 24         # set default plot title font size
# plt.rcParams['legend.fontsize'] = 18        # set default plot legend font size
# make plots inline in the notebook instead of creating a new window


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[current] in legal: return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal: return Directions.LEFT[left]
        return Directions.STOP

class GreedyAgent(Agent):
    def __init__(self, evalFn="scoreEvaluation"):
        self.evaluationFunction = util.lookup(evalFn, globals())
        assert self.evaluationFunction != None

    def getAction(self, state):
        # Generate candidate actions
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal: legal.remove(Directions.STOP)

        successors = [(state.generateSuccessor(0, action), action) for action in legal]
        scored = [(self.evaluationFunction(state), action) for state, action in successors]
        bestScore = max(scored)[0]
        bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
        return random.choice(bestActions)

class NengoAgent(Agent):
    def __init__(self):
        # [0,1,2,3] => [north,east,south,west]
        N = 50  # number of neurons per ensemble
        tau = 0.01  # synapse time constant for probe
        self.T = 1  # length of nengo simulation

        self.model = nengo.Network()
        input_state = np.zeros([4, 4])
        with self.model:
            # [north,east,south,west] by [ghosts,food,power,wall]
			ghosts= nengo.Node(lambda t: input_state[:,0])
			food= nengo.Node(lambda t: input_state[:,1])
			power= nengo.Node(lambda t: input_state[:,2])
			wall= nengo.Node(lambda t: input_state[:,3])
			
            powered_up = nengo.Node(1)  # -1 if powered up, 1 if not
            output = nengo.Node(size_out=4)

            cost_north = nengo.Ensemble(N, dimensions=5)
            cost_east = nengo.Ensemble(N, dimensions=5)
            cost_south = nengo.Ensemble(N, dimensions=5)
            cost_west = nengo.Ensemble(N, dimensions=5)

            nengo.Connection(ghosts[0], cost_north[0])
			nengo.Connection(food[0], cost_north[1])
			nengo.Connection(power[0], cost_north[2])
			nengo.Connection(wall[0], cost_north[3])
            nengo.Connection(powered_up, cost_north[4])
			
			nengo.Connection(ghosts[1], cost_east[0])
			nengo.Connection(food[1], cost_east[1])
			nengo.Connection(power[1], cost_east[2])
			nengo.Connection(wall[1], cost_east[3])
            nengo.Connection(powered_up, cost_east[4])
			
			nengo.Connection(ghosts[2], cost_south[0])
			nengo.Connection(food[2], cost_south[1])
			nengo.Connection(power[2], cost_south[2])
			nengo.Connection(wall[2], cost_south[3])
            nengo.Connection(powered_up, cost_south[4])
			
			nengo.Connection(ghosts[3], cost_west[0])
			nengo.Connection(food[3], cost_west[1])
			nengo.Connection(power[3], cost_west[2])
			nengo.Connection(wall[3], cost_west[3])
            nengo.Connection(powered_up, cost_west[4])

            def cost_fun(state, powered):
                # state= [ghosts,food,power,wall]
                return 100*state[3] + powered*10*state[0] - 2*state[1] - 5*state[2]

            nengo.Connection(cost_north, output[0], function=cost_fun)
            nengo.Connection(cost_east, output[1], function=cost_fun)
            nengo.Connection(cost_south, output[2], function=cost_fun)
            nengo.Connection(cost_west, output[3], function=cost_fun)

            self.output_cost = nengo.Probe(output, synapse=tau)

        self.sim = nengo.Simulator(self.model)

        # should only need to set stim.output= lambda t: input_state for each new board state and then sim.run(T,progress_bar= False) in the getAction method
        # pretty sure we only need one call to sim= nengo.Simulator(model). Will need to pass model (for model.stim) and sim to getAction method somehow

    def getAction(self, state):
        self.sim.run(self.T, progress_bar=False)

        self.t = self.sim.trange()
        plt.ion()
        plt.plot(self.t, self.sim.data[self.probe])
        plt.plot(self.t, self.sim.data[self.probe_in])
        plt.plot(self.t, np.square(self.sim.data[self.probe_in]))
        plt.draw()
        plt.show()

        print self.sim.data[self.output_cost]


        # legal = state.getLegalPacmanActions()
        # current = state.getPacmanState().configuration.direction
        # if current == Directions.STOP: current = Directions.NORTH
        # left = Directions.RIGHT[current]
        # if left in legal: return left
        # if current in legal: return current
        # if Directions.RIGHT[left] in legal: return Directions.RIGHT[left]
        # if Directions.LEFT[current] in legal: return Directions.LEFT[current]
        # return Directions.STOP


def scoreEvaluation(state):
    return state.getScore()
