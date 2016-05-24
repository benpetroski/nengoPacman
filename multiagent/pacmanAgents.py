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
        T = 1.               # duration of simulation
        tau_ens_probe = .01  # Synapse param when creating Probes of Ensembles
        in_fun = lambda t: np.sin(2*np.pi*t)  # input function to your network
        N = 100  # Number of neurons in each Ensemble

        self.network = nengo.Network()
        with self.network:
            ensemble = nengo.Ensemble(N, dimensions=1)
            stimulator = nengo.Node(in_fun)
            nengo.Connection(stimulator, ensemble)
            ensemble_two = nengo.Ensemble(N, dimensions=1)
            def square(x):
                return np.square(x)
            nengo.Connection(ensemble, ensemble_two, function=square)
            self.probe = nengo.Probe(ensemble_two, synapse=tau_ens_probe)
            self.probe_in = nengo.Probe(ensemble, synapse=tau_ens_probe)

        self.sim = nengo.Simulator(self.network)
        self.sim.run(T)
        input_ens, A = nengo.utils.ensemble.tuning_curves(ensemble, self.sim)


    def getAction(self, state):
        self.t = self.sim.trange()
        plt.ion()
        plt.plot(self.t, self.sim.data[self.probe])
        plt.plot(self.t, self.sim.data[self.probe_in])
        plt.plot(self.t, np.square(self.sim.data[self.probe_in]))
        plt.draw()
        plt.show()

        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP: current = Directions.NORTH
        left = Directions.RIGHT[current]
        if left in legal: return left
        if current in legal: return current
        if Directions.RIGHT[left] in legal: return Directions.RIGHT[left]
        if Directions.LEFT[current] in legal: return Directions.LEFT[current]
        return Directions.STOP


def scoreEvaluation(state):
    return state.getScore()
