# myTeam.py
# ---------
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

from captureAgents import CaptureAgent
import random
import util
from game import Directions
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveAgent', second='DefensiveAgent'):
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

class ReflexAgent(CaptureAgent):
  """
  Shared helper code for the two agents.
  """

  def registerInitialState(self, gameState):
    CaptureAgent.registerInitialState(self, gameState)
    self.start = gameState.getAgentPosition(self.index)
    self.midPoints = self.getMidPoints(gameState)

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    return successor

  def getMidPoints(self, gameState):
    x = (gameState.data.layout.width - 2) // 2
    if not self.red:
      x += 1

    points = []
    for y in range(1, gameState.data.layout.height - 1):
      if not gameState.hasWall(x, y):
        points.append((x, y))
    return points

  def closestDistance(self, position, targets):
    if not targets:
      return 0
    return min(self.getMazeDistance(position, target) for target in targets)

  def closestTarget(self, position, targets):
    if not targets:
      return None
    return min(targets, key=lambda target: self.getMazeDistance(position, target))

  def visibleEnemyGhosts(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    return [enemy for enemy in enemies
            if enemy.getPosition() is not None
            and not enemy.isPacman
            and enemy.scaredTimer == 0]

  def visibleInvaders(self, gameState):
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    return [enemy for enemy in enemies
            if enemy.getPosition() is not None
            and enemy.isPacman]

  def actionsWithoutStop(self, gameState):
    actions = gameState.getLegalActions(self.index)
    if Directions.STOP in actions:
      actions.remove(Directions.STOP)
    return actions

  def chooseActionByTarget(self, gameState, target, actions=None):
    if target is None:
      actions = actions or self.actionsWithoutStop(gameState)
      return random.choice(actions)

    actions = actions or self.actionsWithoutStop(gameState)
    distances = []
    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      position = successor.getAgentPosition(self.index)
      distances.append(self.getMazeDistance(position, target))

    bestDistance = min(distances)
    bestActions = [action for action, distance in zip(actions, distances)
                   if distance == bestDistance]
    return random.choice(bestActions)


class OffensiveAgent(ReflexAgent):
  """
  Offense first, but returns home after collecting food or when danger is close.
  """

  def registerInitialState(self, gameState):
    ReflexAgent.registerInitialState(self, gameState)
    self.openingTarget = self.chooseOpeningTarget()
    self.lastFood = self.getFood(gameState).asList()
    self.lastCapsuleCount = len(self.getCapsules(gameState))
    self.foodCarried = 0
    self.powerMode = False
    self.stuckCounter = 0
    self.lastFoodCount = len(self.lastFood)

  def chooseOpeningTarget(self):
    if not self.midPoints:
      return self.start
    middle = len(self.midPoints) // 2
    return self.midPoints[middle]

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    position = successor.getAgentPosition(self.index)
    food = self.getFood(successor).asList()
    ghosts = self.visibleEnemyGhosts(successor)

    features['score'] = self.getScore(successor)
    features['foodLeft'] = len(food)
    features['foodDistance'] = self.closestDistance(position, food)
    features['homeDistance'] = self.closestDistance(position, self.midPoints)

    if successor.getAgentState(self.index).isPacman:
      features['isPacman'] = 1

    if ghosts:
      ghostDistances = [self.getMazeDistance(position, ghost.getPosition())
                        for ghost in ghosts]
      nearestGhost = min(ghostDistances)
      if nearestGhost <= 5:
        features['ghostDanger'] = 6 - nearestGhost

    if action == Directions.STOP:
      features['stop'] = 1

    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == reverse:
      features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    position = gameState.getAgentPosition(self.index)
    foodRemaining = len(self.getFood(gameState).asList())
    ghosts = self.visibleEnemyGhosts(gameState)
    nearestGhost = 999
    if ghosts:
      nearestGhost = min(self.getMazeDistance(position, ghost.getPosition())
                         for ghost in ghosts)

    shouldReturn = self.foodCarried >= 3 or foodRemaining <= 2
    if nearestGhost <= 4 and gameState.getAgentState(self.index).isPacman:
      shouldReturn = True

    if shouldReturn:
      return {
          'score': 200,
          'foodLeft': -50,
          'foodDistance': -1,
          'homeDistance': -25,
          'ghostDanger': -500,
          'stop': -100,
          'reverse': -4,
      }

    return {
        'score': 200,
        'foodLeft': -100,
        'foodDistance': -8,
        'homeDistance': -1,
        'ghostDanger': -350,
        'stop': -100,
        'reverse': -3,
    }

  def updateFoodMemory(self, gameState):
    currentFood = self.getFood(gameState).asList()
    if len(currentFood) < len(self.lastFood):
      self.foodCarried += len(self.lastFood) - len(currentFood)
    if not gameState.getAgentState(self.index).isPacman:
      self.foodCarried = 0
      self.powerMode = False

    capsuleCount = len(self.getCapsules(gameState))
    if capsuleCount < self.lastCapsuleCount:
      self.powerMode = True
      self.foodCarried = 0

    self.lastFood = currentFood
    self.lastCapsuleCount = capsuleCount

  def chooseAction(self, gameState):
    self.updateFoodMemory(gameState)
    position = gameState.getAgentPosition(self.index)

    if position == self.start and self.openingTarget is not None:
      return self.chooseActionByTarget(gameState, self.openingTarget)

    food = self.getFood(gameState).asList()
    capsules = self.getCapsules(gameState)
    ghosts = self.visibleEnemyGhosts(gameState)
    nearestGhostDistance = 999
    if ghosts:
      nearestGhostDistance = min(self.getMazeDistance(position, ghost.getPosition())
                                 for ghost in ghosts)

    if self.powerMode and gameState.getAgentState(self.index).isPacman:
      if self.foodCarried >= 5 or not food:
        return self.chooseActionByTarget(gameState, self.closestTarget(position, self.midPoints))
      return self.chooseActionByTarget(gameState, self.closestTarget(position, food))

    if capsules and nearestGhostDistance <= 6:
      return self.chooseActionByTarget(gameState, self.closestTarget(position, capsules))

    if self.foodCarried >= 3 or (nearestGhostDistance <= 4 and gameState.getAgentState(self.index).isPacman):
      return self.chooseActionByTarget(gameState, self.closestTarget(position, self.midPoints))

    currentFoodCount = len(food)
    if currentFoodCount == self.lastFoodCount:
      self.stuckCounter += 1
    else:
      self.stuckCounter = 0
      self.lastFoodCount = currentFoodCount

    actions = self.actionsWithoutStop(gameState)
    if self.stuckCounter > 15:
      reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      nonReverse = [action for action in actions if action != reverse]
      if nonReverse:
        actions = nonReverse

    values = [self.evaluate(gameState, action) for action in actions]
    bestValue = max(values)
    bestActions = [action for action, value in zip(actions, values) if value == bestValue]
    return random.choice(bestActions)


class DefensiveAgent(ReflexAgent):
  """
  Patrols the border, chases visible invaders, and checks recently eaten food.
  """

  def registerInitialState(self, gameState):
    ReflexAgent.registerInitialState(self, gameState)
    self.target = None
    self.lastDefendedFood = self.getFoodYouAreDefending(gameState).asList()
    self.patrolPoints = self.trimPatrolPoints(self.midPoints)

  def trimPatrolPoints(self, points):
    if len(points) <= 4:
      return points[:]
    return points[1:-1]

  def legalDefensiveActions(self, gameState):
    actions = self.actionsWithoutStop(gameState)
    reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if reverse in actions and len(actions) > 1:
      actions.remove(reverse)

    defensiveActions = []
    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      if not successor.getAgentState(self.index).isPacman:
        defensiveActions.append(action)

    if defensiveActions:
      return defensiveActions

    actions = self.actionsWithoutStop(gameState)
    return actions

  def chooseTarget(self, gameState):
    position = gameState.getAgentPosition(self.index)
    invaders = self.visibleInvaders(gameState)
    if invaders:
      invaderPositions = [invader.getPosition() for invader in invaders]
      return self.closestTarget(position, invaderPositions)

    defendedFood = self.getFoodYouAreDefending(gameState).asList()
    if self.lastDefendedFood and len(defendedFood) < len(self.lastDefendedFood):
      eaten = list(set(self.lastDefendedFood) - set(defendedFood))
      if eaten:
        return self.closestTarget(position, eaten)

    if len(defendedFood) <= 4:
      importantTargets = defendedFood + self.getCapsulesYouAreDefending(gameState)
      if importantTargets:
        return self.closestTarget(position, importantTargets)

    if not self.target or position == self.target:
      if self.patrolPoints:
        return random.choice(self.patrolPoints)
      return self.closestTarget(position, self.midPoints)

    return self.target

  def chooseAction(self, gameState):
    self.target = self.chooseTarget(gameState)
    self.lastDefendedFood = self.getFoodYouAreDefending(gameState).asList()

    actions = self.legalDefensiveActions(gameState)
    distances = []
    for action in actions:
      successor = gameState.generateSuccessor(self.index, action)
      position = successor.getAgentPosition(self.index)
      distances.append(self.getMazeDistance(position, self.target))

    bestDistance = min(distances)
    bestActions = [action for action, distance in zip(actions, distances)
                   if distance == bestDistance]
    return random.choice(bestActions)
