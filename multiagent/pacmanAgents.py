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
from scipy.signal import butter, lfilter
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
        N = 400  # number of neurons per ensemble
        tau = 0.1  # synapse time constant for probe
        self.T = 1  # length of nengo simulation
        self.powered_up_var = 1
        self.dir_opt = [Directions.NORTH, Directions.EAST, Directions.SOUTH, Directions.WEST]

        self.model = nengo.Network()
        self.input_state = np.zeros([4, 3])

        with self.model:
            # [north,east,south,west] by [ghosts,food,power]
            ghosts = nengo.Node(lambda t, d: self.input_state[:, 0], size_in=4)
            food = nengo.Node(lambda t, d: self.input_state[:, 1], size_in=4)
            power = nengo.Node(lambda t, d: self.input_state[:, 2], size_in=4)

            powered_up = nengo.Node(lambda t: self.powered_up_var)  # -1 if powered up, 1 if not
            output = nengo.Ensemble(N, dimensions=4,radius=10)

            cost_north = nengo.Ensemble(N, dimensions=4,radius=10)
            cost_east = nengo.Ensemble(N, dimensions=4,radius=10)
            cost_south = nengo.Ensemble(N, dimensions=4,radius=10)
            cost_west = nengo.Ensemble(N, dimensions=4,radius=10)

            nengo.Connection(ghosts[0], cost_north[0])
            nengo.Connection(food[0], cost_north[1])
            nengo.Connection(power[0], cost_north[2])
            nengo.Connection(powered_up, cost_north[3])

            nengo.Connection(ghosts[1], cost_east[0])
            nengo.Connection(food[1], cost_east[1])
            nengo.Connection(power[1], cost_east[2])
            nengo.Connection(powered_up, cost_east[3])

            nengo.Connection(ghosts[2], cost_south[0])
            nengo.Connection(food[2], cost_south[1])
            nengo.Connection(power[2], cost_south[2])
            nengo.Connection(powered_up, cost_south[3])

            nengo.Connection(ghosts[3], cost_west[0])
            nengo.Connection(food[3], cost_west[1])
            nengo.Connection(power[3], cost_west[2])
            nengo.Connection(powered_up, cost_west[3])

            def cost_fun(x):
                # state= [ghosts,food,power]
                ghosts, food, power, powered = x
                return powered*40*ghosts - 4*food - 10*power
                # return powered + ghosts + food + power

            nengo.Connection(cost_north, output[0], function=cost_fun)
            nengo.Connection(cost_east, output[1], function=cost_fun)
            nengo.Connection(cost_south, output[2], function=cost_fun)
            nengo.Connection(cost_west, output[3], function=cost_fun)

            self.output_cost = nengo.Probe(output, synapse=tau)

        self.sim = nengo.Simulator(self.model)
        self.sim.run(3, progress_bar=False)
        # should only need to set stim.output= lambda t: input_state for each new board state and then sim.run(T,progress_bar= False) in the getAction method
        # pretty sure we only need one call to sim= nengo.Simulator(model). Will need to pass model (for model.stim) and sim to getAction method somehow

    def butter_bandpass(self, highcut, fs, order=4):
        nyq = 0.5 * fs
        high = highcut / nyq
        b, a = butter(order, high, btype='high')
        return b, a

    def butter_bandpass_filter(self, data, highcut, fs, order=4):
        b, a = self.butter_bandpass(highcut, fs, order=order)
        y = lfilter(b, a, data, axis=0)
        return y

    def getAction(self, state):
        self.state = state
        self.input_state = np.zeros([4, 3])
        above = (self.state.getPacmanPosition()[0]+0.0, self.state.getPacmanPosition()[1]+1.0)
        aboveL = (self.state.getPacmanPosition()[0]-1.0, self.state.getPacmanPosition()[1]+1.0)
        aboveA = (self.state.getPacmanPosition()[0]+0.0, self.state.getPacmanPosition()[1]+2.0)
        aboveR = (self.state.getPacmanPosition()[0]+1.0, self.state.getPacmanPosition()[1]+1.0)
        right = (self.state.getPacmanPosition()[0]+1.0, self.state.getPacmanPosition()[1]+0.0)
        rightA = (self.state.getPacmanPosition()[0]+1.0, self.state.getPacmanPosition()[1]+1.0)
        rightR = (self.state.getPacmanPosition()[0]+2.0, self.state.getPacmanPosition()[1]+0.0)
        rightB = (self.state.getPacmanPosition()[0]+1.0, self.state.getPacmanPosition()[1]-1.0)
        below = (self.state.getPacmanPosition()[0]+0.0, self.state.getPacmanPosition()[1]-1.0)
        belowR = (self.state.getPacmanPosition()[0]+1.0, self.state.getPacmanPosition()[1]-1.0)
        belowB = (self.state.getPacmanPosition()[0]+0.0, self.state.getPacmanPosition()[1]-2.0)
        belowL = (self.state.getPacmanPosition()[0]-1.0, self.state.getPacmanPosition()[1]-1.0)
        left = (self.state.getPacmanPosition()[0]-1.0, self.state.getPacmanPosition()[1]+0.0)
        leftB = (self.state.getPacmanPosition()[0]-1.0, self.state.getPacmanPosition()[1]-1.0)
        leftL = (self.state.getPacmanPosition()[0]-2.0, self.state.getPacmanPosition()[1]+0.0)
        leftA = (self.state.getPacmanPosition()[0]-1.0, self.state.getPacmanPosition()[1]+1.0)
        aboveGroup = [above, aboveL, aboveA, aboveR]
        rightGroup = [right, rightA, rightR, rightB]
        belowGroup = [below, belowR, belowB, belowL]
        leftGroup = [left, leftB, leftL, leftA]
        surroundingGroups = [aboveGroup, rightGroup, belowGroup, leftGroup]
        # print self.state.getPacmanPosition(), ", ", above, ", ", right, ", ", below, ", ", left
        # print self.state.getGhostPositions()
        # print self.state.hasFood(x, y)

        # print state.data.agentStates[1].scaredTimer
        # if state.data.agentStates[1].scaredTimer > 0:
        #     self.powered_up_var = -1
        # else:
        #     self.powered_up_var = 1

        print self.state.getPacmanPosition()
        for i in xrange(4):
            for j in xrange(4):
                if surroundingGroups[i][j] in self.state.getGhostPositions():
                    self.input_state[i, 0] += 1
                    print i, j, 'poop there is a ghost'
                try:
                    if self.state.hasFood(int(surroundingGroups[i][j][0]), int(surroundingGroups[i][j][1])):
                        self.input_state[i, 1] += 1
                        print i, j, 'yum food'
                except IndexError as e:
                    continue

        print self.input_state

        self.sim.run(self.T, progress_bar=False)

        # self.output_new = self.butter_bandpass_filter(self.sim.data[self.output_cost], 1, 1000)
        # self.output_new = [x - self.sim.data[self.output_cost][100,:] for x in self.sim.data[self.output_cost]]

        self.output_new = np.array(self.sim.data[self.output_cost]) - np.mean(np.array(self.sim.data[self.output_cost][self.T*500:self.T*1000, :]), axis=0)

        plt.figure(0)
        self.t = self.sim.trange()
        plt.ion()
        plt.clf()
        plt.plot(self.t, self.output_new[:,0], 'r', label='north')
        plt.plot(self.t, self.output_new[:,1], 'g', label='east')
        plt.plot(self.t, self.output_new[:,2], 'b', label='south')
        plt.plot(self.t, self.output_new[:,3], 'y', label='west')
        plt.legend(loc=2)
        plt.draw()
        plt.show()

        legal = state.getLegalPacmanActions()
        endSlice = self.output_new[-1,:].tolist()
        while(True):
            m = min(i for i in endSlice)
            pos = endSlice.index(m)
            # print("Position:", pos)
            # print("Value:", m)
            if self.dir_opt[pos] in legal:
                return self.dir_opt[pos]
            else:
                endSlice[pos] = float('Inf')


def scoreEvaluation(state):
    return state.getScore()
