# baselineTeam.py
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.capsule_time = 0
    self.distancer = distanceCalculator.Distancer(gameState.data.layout)
    self.distancer.getMazeDistances()
    self.weights = None

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)
    if self.getPreviousObservation()!= None:
      if len(self.getCapsules(self.getPreviousObservation())) - len(self.getCapsules(gameState)):
        self.capsule_time = 38
      else:
        self.capsule_time = max(0, self.capsule_time - 1)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    if bestActions == []:
      print actions, values
      print gameState
      print [self.getFeatures(gameState, a) for a in actions]

    random_action = random.choice(bestActions)
    self.update_weight(gameState, random_action)
    return random_action

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def update_weight(self, gameState, action):
    return self.getWeights(gameState, action)

  def evaluate_minimax(self, gameState):
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    return maxValue

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    myPos = gameState.getAgentPosition(self.index)
    score = self.getScore(gameState)
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    myAgent = successor.getAgentState(self.index)
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    

    # Successor Score: How much score higher than the other team
    features['successorScore'] = self.getScore(successor)

    # Food To Eat: how many food we need to eat to beat the other team
    foodList = self.getFood(successor).asList()
    features['foodToEat'] = len(self.getFoodYouAreDefending(successor).asList()) - len(foodList)

    # Min Score Dist: Minimum
    min_dist_to_score = min(
      [self.getMazeDistance(myPos, (16, float(i))) for i in range(0, 16) if not successor.hasWall(16, i)])
    if self.red:
      min_dist_to_score = min(
        [self.getMazeDistance(myPos, (15, float(i))) for i in range(0, 16) if not successor.hasWall(15, i)])
    features['minScoreDist'] = -min_dist_to_score

    # Distance to Food: distance to the nearest food
    minDistance = 0
    if len(foodList) > 0:
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
    features['distanceToFood'] = -minDistance


    # Distance to Ghost: how far are we from the closest ghost
    min_dist_to_ghost = 5
    for oppo in self.getOpponents(successor):
      if not successor.getAgentState(oppo).isPacman:
        if successor.getAgentPosition(oppo) is None:
          continue
        dist = self.getMazeDistance(myPos, successor.getAgentPosition(oppo))
        if dist < min_dist_to_ghost or not min_dist_to_ghost:
          min_dist_to_ghost = dist
    if self.capsule_time > 0: # Opponent is white ghost
        min_dist_to_ghost = 5
    if myPos == successor.getInitialAgentPosition(self.index):
        min_dist_to_ghost = -5 # We got eaten
    features['distanceToGhost'] = min_dist_to_ghost


    # Distance to Capsules
    dist_to_capsules = 0
    capPos = self.getCapsules(successor)
    if capPos:
        capPos = capPos[0]
        dist_to_capsules = self.getMazeDistance(myPos, capPos)
    features['distanceToCapsules'] = -dist_to_capsules

    # ifDeadEnd: 3 walls around
    deadEnd = 0
    if len(successor.getLegalActions(self.index)) == 1:
        deadEnd = -1
    elif len(successor.getLegalActions(self.index)) == 2:
        deadEnd = -0.5
    features['isDeadEnd'] = deadEnd

    # Distance to Pacman: Attack opponent pacman when we haven't get over yet
    dist_to_pacman = 0
    if not myAgent.isPacman:
        for oppo in self.getOpponents(successor):
            if successor.getAgentState(oppo).isPacman \
              and successor.getAgentPosition(oppo) is not None:
                dist = self.getMazeDistance(myPos, successor.getAgentPosition(oppo))
                if dist_to_pacman and dist < dist_to_pacman:
                  dist_to_pacman = dist
    features['distanceToPacman'] = -dist_to_pacman

    # Distance to partner
    # dist_to_partner = 0
    # myTeam = self.getTeam(gameState)
    # partner_index = [a for a in myTeam if a != self.index][0]
    # partner = gameState.getAgentState(partner_index)
    # partnerPos = partner.getPosition()
    # features['distanceToPartner'] = self.getMazeDistance(myPos, partnerPos)
    return features

  def getWeights(self, gameState, action):
    # How many food on us we haven't delivered back
    food_bearing = 20 - len(self.getFood(gameState).asList()) - self.getScore(gameState)
    if self.weights is None:
      self.weights = {'successorScore': 100000,
                      'distanceToFood': 1,
                      'foodToEat': 10000,
                      'minScoreDist': 2 * food_bearing,
                      'distanceToGhost': 5 * (food_bearing + 1),
                      'distanceToCapsules': 10,
                      'isDeadEnd': 1,
                      'distanceToPacman': 10}
    return self.weights

  def update_weight(self, gameState, action):
    # state after I take the action
    next_state = self.getSuccessor(gameState, action)
    # next state after enemy take action
    # enemy aim to minimize feature
    oppoTeam = self.getOpponents(gameState)
    state = next_state
    for oppo in oppoTeam:
      if next_state.getAgentPosition(oppo) is not None:
        for a in next_state.getLegalActions(oppo):
          s = next_state.generateSuccessor(oppo, a)
          if state == None or self.evaluate_minimax(s) < self.evaluate_minimax(state):
            state = s
    # find out reward: 2 parts
    # 1. positive reward: score gain
    # 2. negative reward: food loss
    reward = 0
    # score gain
    reward += (self.getScore(state) - self.getScore(gameState))
    reward -= (len(self.getFood(state).asList()) - len(self.getFood(gameState).asList()))
    """difference = (reward + gamma * max Q(s', a')) - Q(s, a)"""
    # temporary gamma(discount factor) 0.8
    diff = (reward + 0.8 * self.evaluate_minimax(state)) - self.evaluate(gameState, action)
    """update: weight[i] = weight[i] + alpha * difference * feature[i]"""
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    for key in weights.keys():
      weights[key] += 0.5 * diff * features[key]
    # print self.getWeights(gameState, action)

class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # # Computes whether we're on defense (1) or offense (0)
    # features['onDefense'] = 1
    # if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    if self.weights is None:
      self.weights = {'numInvaders': -1, 'invaderDistance': -0.5, 'stop': -0.5, 'reverse': -0.1}
    return self.weights

  def update_weight(self, gameState, action):
    # state after I take the action
    next_state = self.getSuccessor(gameState, action)
    # next state after enemy take action
    # enemy aim to minimize feature
    oppoTeam = self.getOpponents(gameState)
    state = next_state
    for oppo in oppoTeam:
      if next_state.getAgentPosition(oppo) is not None:
        for a in next_state.getLegalActions(oppo):
          s = next_state.generateSuccessor(oppo, a)
          if state == None or self.evaluate_minimax(s) < self.evaluate_minimax(state):
            state = s
    # find out reward: 2 parts
    # 1. positive reward: score gain
    # 2. negative reward: food loss
    reward = 0
    # score opponent gain
    reward -= (self.getScore(gameState) - self.getScore(state))
    # potential opponent score loss
    reward += (len(self.getFoodYouAreDefending(state).asList()) - len(self.getFoodYouAreDefending(gameState).asList()))
    """difference = (reward + gamma * max Q(s', a')) - Q(s, a)"""
    # temporary gamma(discount factor) 0.8
    diff = (reward + 0.8 * self.evaluate_minimax(state)) - self.evaluate(gameState, action)
    """update: weight[i] = weight[i] + alpha * difference * feature[i]"""
    features = self.getFeatures(gameState, action)
    for key in self.getWeights(gameState, action).keys():
      self.getWeights(gameState, action)[key] = min(0.5 * diff * features[key] + self.getWeights(gameState, action)[key], -0.1)
    print self.getWeights(gameState, action)
