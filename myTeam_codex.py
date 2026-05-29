from captureAgents import CaptureAgent
from game import Directions
from util import Counter, nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


ACTION_ORDER = [
    Directions.NORTH,
    Directions.EAST,
    Directions.SOUTH,
    Directions.WEST,
    Directions.STOP,
]


class ReflexCaptureAgent(CaptureAgent):
    """Small, deterministic reflex agent built for the capture time limit."""

    defaultRole = 'offense'

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.previousDefendingFood = self.getFoodYouAreDefending(gameState).asList()
        self.missingFoodTarget = None
        self.patrolPoints = self._computeBoundaryPoints(gameState)
        if not self.patrolPoints:
            self.patrolPoints = [self.start]

    def chooseAction(self, gameState):
        legal = gameState.getLegalActions(self.index)
        if not legal:
            return Directions.STOP

        try:
            self._updateMissingFoodTarget(gameState)
            role = self.selectRole(gameState)
            actions = self._candidateActions(gameState, legal, role)
            values = [(self.evaluate(gameState, action, role), action) for action in actions]
            bestValue = max(value for value, action in values)
            bestActions = [action for value, action in values if value == bestValue]
            choice = self._firstByActionOrder(bestActions)
        except Exception:
            choice = self.safeFallbackAction(gameState, legal)

        if choice not in legal:
            choice = self.safeFallbackAction(gameState, legal)
        self.previousDefendingFood = self.getFoodYouAreDefending(gameState).asList()
        return choice

    def selectRole(self, gameState):
        myState = gameState.getAgentState(self.index)
        visibleInvaders = self.getVisibleInvaders(gameState)
        foodLeft = len(self.getFood(gameState).asList())
        carriedFood = getattr(myState, 'numCarrying', 0)
        score = self.getScore(gameState)
        closeGhost = self.closestVisibleGhostDistance(gameState)
        timeLeft = getattr(gameState.data, 'timeleft', None)
        homeDistance = self.distanceToHome(gameState, myState.getPosition())

        if visibleInvaders and self.shouldDefendInvaders(gameState, visibleInvaders):
            return 'defense'
        if foodLeft <= 2:
            return 'return' if myState.isPacman or carriedFood > 0 else 'defense'
        if carriedFood >= 4:
            return 'return'
        if score > 0 and carriedFood >= 2:
            return 'return'
        if carriedFood > 0 and closeGhost is not None and closeGhost <= 4:
            return 'return'
        if timeLeft is not None and myState.isPacman and homeDistance + 8 >= timeLeft:
            return 'return'
        return self.defaultRole

    def shouldDefendInvaders(self, gameState, visibleInvaders):
        if self.defaultRole == 'defense':
            return True
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return False
        myDist = min(self.getMazeDistance(myPos, inv.getPosition()) for inv in visibleInvaders)
        teammateDists = []
        for teammate in self.getTeam(gameState):
            if teammate == self.index:
                continue
            teammatePos = gameState.getAgentPosition(teammate)
            if teammatePos is not None:
                teammateDists.append(min(self.getMazeDistance(teammatePos, inv.getPosition())
                                         for inv in visibleInvaders))
        return len(visibleInvaders) > 1 or not teammateDists or myDist + 2 < min(teammateDists)

    def evaluate(self, gameState, action, role):
        if role == 'defense':
            features = self.getDefenseFeatures(gameState, action)
            weights = self.getDefenseWeights(gameState, action)
        else:
            features = self.getOffenseFeatures(gameState, action)
            weights = self.getOffenseWeights(gameState, action, role)
        return features * weights

    def getSuccessor(self, gameState, action):
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            return successor.generateSuccessor(self.index, action)
        return successor

    def getOffenseFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        foodList = self.getFood(successor).asList()
        capsules = self.getCapsules(successor)

        features['successorScore'] = self.getScore(successor)
        features['carriedFood'] = getattr(myState, 'numCarrying', 0)
        features['distanceToHome'] = self.distanceToHome(successor, myPos)

        if foodList:
            features['distanceToFood'] = min(self.getMazeDistance(myPos, food) for food in foodList)
        if capsules:
            features['capsuleDistance'] = min(self.getMazeDistance(myPos, capsule) for capsule in capsules)

        ghostDistances = self.visibleGhostDistances(successor, myPos)
        features['ghostDistance'] = min(min(ghostDistances), 6) if ghostDistances else 6
        if ghostDistances and min(ghostDistances) <= 1:
            features['ghostDistance'] = -10

        self.addMovementFeatures(gameState, action, features)
        return features

    def getOffenseWeights(self, gameState, action, role):
        carriedFood = getattr(gameState.getAgentState(self.index), 'numCarrying', 0)
        if role == 'return':
            return {
                'successorScore': 120,
                'distanceToFood': -1,
                'distanceToHome': -24,
                'carriedFood': 70,
                'ghostDistance': 45,
                'capsuleDistance': -3,
                'stop': -300,
                'reverse': -10,
            }
        return {
            'successorScore': 120,
            'distanceToFood': -9,
            'distanceToHome': -3 * min(carriedFood, 4),
            'carriedFood': 25,
            'ghostDistance': 35,
            'capsuleDistance': -6,
            'stop': -300,
            'reverse': -6,
        }

    def getDefenseFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 0 if myState.isPacman else 1
        invaders = self.getVisibleInvaders(successor)
        if invaders:
            features['visibleInvaderDistance'] = min(
                self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders)
        elif self.missingFoodTarget is not None:
            features['missingFoodTargetDistance'] = self.getMazeDistance(myPos, self.missingFoodTarget)
        else:
            patrolTarget = self.getBoundaryPatrolTarget(gameState, myPos)
            features['boundaryPatrolDistance'] = self.getMazeDistance(myPos, patrolTarget)

        self.addMovementFeatures(gameState, action, features)
        return features

    def getDefenseWeights(self, gameState, action):
        return {
            'onDefense': 260,
            'visibleInvaderDistance': -18,
            'missingFoodTargetDistance': -9,
            'boundaryPatrolDistance': -5,
            'stop': -220,
            'reverse': -8,
        }

    def addMovementFeatures(self, gameState, action, features):
        if action == Directions.STOP:
            features['stop'] = 1
        currentDirection = gameState.getAgentState(self.index).configuration.direction
        reverse = Directions.REVERSE[currentDirection]
        if action == reverse:
            features['reverse'] = 1

    def getVisibleInvaders(self, gameState):
        enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        return [enemy for enemy in enemies if enemy.isPacman and enemy.getPosition() is not None]

    def visibleGhostDistances(self, gameState, myPos):
        distances = []
        for enemyIndex in self.getOpponents(gameState):
            enemy = gameState.getAgentState(enemyIndex)
            enemyPos = enemy.getPosition()
            if enemyPos is not None and not enemy.isPacman and enemy.scaredTimer <= 2:
                distances.append(self.getMazeDistance(myPos, enemyPos))
        return distances

    def closestVisibleGhostDistance(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return None
        distances = self.visibleGhostDistances(gameState, myPos)
        return min(distances) if distances else None

    def distanceToHome(self, gameState, pos):
        if pos is None or not self.patrolPoints:
            return 0
        return min(self.getMazeDistance(pos, point) for point in self.patrolPoints)

    def getBoundaryPatrolTarget(self, gameState, myPos):
        if self.missingFoodTarget is not None:
            return self.missingFoodTarget
        centerY = (gameState.data.layout.height - 1) / 2.0
        team = sorted(self.getTeam(gameState))
        offset = -2 if team.index(self.index) == 0 else 2
        targetY = centerY + offset
        return min(self.patrolPoints,
                   key=lambda point: (abs(point[1] - targetY),
                                      self.getMazeDistance(myPos, point),
                                      point[1]))

    def _computeBoundaryPoints(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        x = width // 2 - 1 if self.red else width // 2
        points = [(x, y) for y in range(1, height - 1) if not gameState.hasWall(x, y)]
        if len(points) > 4:
            points = points[1:-1]
        return points

    def _updateMissingFoodTarget(self, gameState):
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        if self.previousDefendingFood:
            missing = list(set(self.previousDefendingFood) - set(currentFood))
            myPos = gameState.getAgentPosition(self.index)
            if missing and myPos is not None:
                self.missingFoodTarget = min(missing,
                                             key=lambda food: self.getMazeDistance(myPos, food))
        if self.missingFoodTarget == myPosOrNone(gameState, self.index):
            self.missingFoodTarget = None

    def _candidateActions(self, gameState, legal, role):
        actions = [action for action in legal if action != Directions.STOP]
        if not actions:
            return legal
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        noReverse = [action for action in actions if action != reverse]
        if noReverse and role != 'defense':
            return noReverse
        return actions

    def safeFallbackAction(self, gameState, legal):
        actions = [action for action in legal if action != Directions.STOP]
        if not actions:
            return Directions.STOP if Directions.STOP in legal else legal[0]

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myState.isPacman or getattr(myState, 'numCarrying', 0) > 0:
            target = min(self.patrolPoints, key=lambda point: self.getMazeDistance(myPos, point))
        else:
            target = self.getBoundaryPatrolTarget(gameState, myPos)

        scored = []
        for action in actions:
            try:
                successor = gameState.generateSuccessor(self.index, action)
                nextPos = successor.getAgentPosition(self.index)
                distance = self.getMazeDistance(nextPos, target)
            except Exception:
                distance = 999999
            scored.append((distance, ACTION_ORDER.index(action), action))
        return min(scored)[2]

    def _firstByActionOrder(self, actions):
        return min(actions, key=lambda action: ACTION_ORDER.index(action))


def myPosOrNone(gameState, index):
    try:
        return gameState.getAgentPosition(index)
    except Exception:
        return None


class OffensiveReflexAgent(ReflexCaptureAgent):
    defaultRole = 'offense'


class DefensiveReflexAgent(ReflexCaptureAgent):
    defaultRole = 'defense'
