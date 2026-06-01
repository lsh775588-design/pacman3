from captureAgents import CaptureAgent
from game import Actions, Directions
from util import Counter, manhattanDistance, nearestPoint
import time


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='OffensiveReflexAgent', second='DefensiveReflexAgent',
               **kwargs):
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

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        self.weights = Counter()
        self.currentRole = None
        self.roleTurns = 0
        self.moveDeadline = 0.0
        self.opponentBeliefs = {}
        self.legalPositions = []
        self.deadEndDepth = {}

    def registerInitialState(self, gameState):
        CaptureAgent.registerInitialState(self, gameState)
        self.start = gameState.getAgentPosition(self.index)
        self.walls = gameState.getWalls()
        self.legalPositions = self._computeLegalPositions(gameState)
        self.deadEndDepth = self._computeDeadEndDepths(gameState)
        for opponent in self.getOpponents(gameState):
            initialPos = gameState.getInitialAgentPosition(opponent)
            belief = Counter()
            belief[initialPos] = 1.0
            self.opponentBeliefs[opponent] = belief
        self.previousDefendingFood = self.getFoodYouAreDefending(gameState).asList()
        self.recentMissingFood = []
        self.missingFoodTarget = None
        self.patrolPoints = self._computeBoundaryPoints(gameState)
        if not self.patrolPoints:
            self.patrolPoints = [self.start]
        self.patrolIndex = self._initialPatrolIndex(gameState)
        self.patrolTarget = self.patrolPoints[self.patrolIndex]
        self.weights = self.initialWeights()

    def chooseAction(self, gameState):
        self.moveDeadline = time.time() + 0.85
        legal = gameState.getLegalActions(self.index)
        if not legal:
            return Directions.STOP

        try:
            self.updateOpponentBeliefs(gameState)
            self._updateMissingFoodTarget(gameState)
            role = self.applyRoleHysteresis(gameState, self.selectRole(gameState))
            if role == 'defense':
                self.updatePatrolTarget(gameState)
            actions = self._candidateActions(gameState, legal, role)
            choice = self.getPriorityAttackAction(gameState, actions, role)
            if choice is None:
                choice = self.selectActionByPolicy(gameState, actions, role)
        except Exception:
            role = self.defaultRole
            choice = self.safeFallbackAction(gameState, legal)

        if choice not in legal:
            choice = self.safeFallbackAction(gameState, legal)
        self.previousDefendingFood = self.getFoodYouAreDefending(gameState).asList()
        return choice

    def final(self, gameState):
        CaptureAgent.final(self, gameState)

    def selectActionByPolicy(self, gameState, actions, role):
        bestValue = None
        bestActions = []
        for action in actions:
            if self.outOfTime() and bestActions:
                break
            value = self.getQValue(gameState, action, role)
            if bestValue is None or value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)
        if not bestActions:
            return self.safeFallbackAction(gameState, actions)
        return self._firstByActionOrder(bestActions, role, gameState)

    def getPriorityAttackAction(self, gameState, actions, role):
        if role == 'defense':
            return None
        capsuleAction = self.getCapsuleEscapeAction(gameState, actions)
        if capsuleAction is not None:
            return capsuleAction
        scaredGhostAction = self.getScaredGhostChaseAction(gameState, actions)
        if scaredGhostAction is not None:
            return scaredGhostAction
        foodLeft = len(self.getFood(gameState).asList())
        if foodLeft <= 4:
            foodAction = self.getImmediateFoodAction(gameState, actions)
            if foodAction is not None:
                return foodAction
            endgameAction = self.getEndgameFoodAction(gameState, actions)
            if endgameAction is not None:
                return endgameAction
        if foodLeft <= 4 or self.hasScaredEnemyGhost(gameState):
            return self.getGreedyFoodAction(gameState, actions)
        return None

    def getCapsuleEscapeAction(self, gameState, actions):
        capsules = self.getCapsules(gameState)
        if not capsules:
            return None
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return None
        ghostDistance = self.closestVisibleActiveGhostDistance(gameState, myPos)
        if ghostDistance is None or ghostDistance > 5:
            return None

        scored = []
        for action in actions:
            if self.outOfTime():
                break
            successor = self.getSuccessor(gameState, action)
            if not self.isSafeAttackSuccessor(successor):
                continue
            nextPos = successor.getAgentState(self.index).getPosition()
            nextCapsules = self.getCapsules(successor)
            if len(nextCapsules) < len(capsules):
                capsuleDistance = -1
            elif nextCapsules:
                capsuleDistance = min(self.getMazeDistance(nextPos, capsule) for capsule in nextCapsules)
            else:
                capsuleDistance = -1
            scored.append((capsuleDistance, self.actionRank(action, 'offense', gameState), action))
        if not scored:
            return None
        return min(scored)[2]

    def getScaredGhostChaseAction(self, gameState, actions):
        targets = self.visibleScaredGhosts(gameState, minTimer=4)
        if not targets:
            return None

        scored = []
        currentScore = self.getScore(gameState)
        for action in actions:
            if self.outOfTime():
                break
            successor = self.getSuccessor(gameState, action)
            if not self.isSafeAttackSuccessor(successor):
                continue
            myState = successor.getAgentState(self.index)
            myPos = myState.getPosition()
            if myPos is None:
                continue

            bestTargetScore = None
            for enemyPos, scaredTimer in targets:
                distance = self.getMazeDistance(myPos, enemyPos)
                if distance > max(1, scaredTimer - 2):
                    continue
                eatNow = 1 if distance == 0 else 0
                scoreGain = self.getScore(successor) - currentScore
                deadEnd = self.deadEndDepth.get(nearestPoint(myPos), 0)
                targetScore = (scoreGain, eatNow, -distance, scaredTimer, -deadEnd)
                if bestTargetScore is None or targetScore > bestTargetScore:
                    bestTargetScore = targetScore

            if bestTargetScore is not None:
                scored.append((bestTargetScore, -self.actionRank(action, 'offense', gameState), action))

        if not scored:
            return None
        return max(scored)[2]

    def getImmediateFoodAction(self, gameState, actions):
        before = len(self.getFood(gameState).asList())
        candidates = []
        for action in actions:
            if self.outOfTime():
                break
            successor = self.getSuccessor(gameState, action)
            if len(self.getFood(successor).asList()) < before and self.isSafeAttackSuccessor(successor):
                candidates.append(action)
        if not candidates:
            return None
        return max(candidates,
                   key=lambda action: (self.getScore(self.getSuccessor(gameState, action)),
                                       -self.actionRank(action, 'offense', gameState)))

    def getEndgameFoodAction(self, gameState, actions):
        foodList = self.getFood(gameState).asList()
        if not foodList:
            return None
        scored = []
        for action in actions:
            if self.outOfTime():
                break
            successor = self.getSuccessor(gameState, action)
            if not self.isSafeAttackSuccessor(successor):
                continue
            myPos = successor.getAgentState(self.index).getPosition()
            nextFood = self.getFood(successor).asList()
            if not nextFood:
                chainDistance = -2
            else:
                firstFood = min(nextFood, key=lambda food: self.getMazeDistance(myPos, food))
                firstDistance = self.getMazeDistance(myPos, firstFood)
                remainingFood = [food for food in nextFood if food != firstFood]
                secondDistance = 0
                if remainingFood:
                    secondDistance = min(self.getMazeDistance(firstFood, food) for food in remainingFood)
                chainDistance = firstDistance + secondDistance
            scored.append((chainDistance, -self.getScore(successor),
                           self.actionRank(action, 'offense', gameState), action))
        if not scored:
            return None
        return min(scored)[3]

    def getGreedyFoodAction(self, gameState, actions):
        foodList = self.getFood(gameState).asList()
        if not foodList:
            return None
        scored = []
        for action in actions:
            if self.outOfTime():
                break
            successor = self.getSuccessor(gameState, action)
            if not self.isSafeAttackSuccessor(successor):
                continue
            myPos = successor.getAgentState(self.index).getPosition()
            nextFood = self.getFood(successor).asList()
            if len(nextFood) < len(foodList):
                distance = -1
            elif nextFood:
                distance = min(self.getMazeDistance(myPos, food) for food in nextFood)
            else:
                distance = -1
            scored.append((distance, -self.getScore(successor),
                           self.actionRank(action, 'offense', gameState), action))
        if not scored:
            return None
        return min(scored)[3]

    def isSafeAttackSuccessor(self, successor):
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos is None:
            return False
        visibleThreats = []
        for enemyIndex in self.getOpponents(successor):
            enemy = successor.getAgentState(enemyIndex)
            enemyPos = enemy.getPosition()
            if enemyPos is None or enemy.isPacman or enemy.scaredTimer > 0:
                continue
            distance = self.getMazeDistance(myPos, enemyPos)
            visibleThreats.append(distance)
            if distance <= 1:
                return False
            if myState.isPacman and distance <= 4:
                exits = [action for action in successor.getLegalActions(self.index)
                         if action != Directions.STOP]
                if len(exits) <= 2:
                    return False
        estimatedThreat = self.closestEstimatedGhostDistance(successor, myPos)
        nearestThreat = min(visibleThreats) if visibleThreats else estimatedThreat
        if estimatedThreat is not None and nearestThreat is not None:
            nearestThreat = min(nearestThreat, estimatedThreat)
        depth = self.deadEndDepth.get(nearestPoint(myPos), 0)
        if myState.isPacman and depth > 0 and nearestThreat is not None and nearestThreat <= depth + 2:
            return False
        return True

    def hasScaredEnemyGhost(self, gameState):
        return bool(self.visibleScaredGhosts(gameState, minTimer=4))

    def visibleScaredGhosts(self, gameState, minTimer=1):
        ghosts = []
        for enemyIndex in self.getOpponents(gameState):
            enemy = gameState.getAgentState(enemyIndex)
            enemyPos = enemy.getPosition()
            if enemyPos is not None and not enemy.isPacman and enemy.scaredTimer >= minTimer:
                ghosts.append((enemyPos, enemy.scaredTimer))
        return ghosts

    def closestVisibleActiveGhostDistance(self, gameState, myPos):
        distances = []
        for enemyIndex in self.getOpponents(gameState):
            enemy = gameState.getAgentState(enemyIndex)
            enemyPos = enemy.getPosition()
            if enemyPos is not None and not enemy.isPacman and enemy.scaredTimer <= 2:
                distances.append(self.getMazeDistance(myPos, enemyPos))
        return min(distances) if distances else None

    def getQValue(self, gameState, action, role):
        return self.getQFeatures(gameState, action, role) * self.weights

    def getQFeatures(self, gameState, action, role):
        if role == 'defense':
            rawFeatures = self.getDefenseFeatures(gameState, action)
        else:
            rawFeatures = self.getOffenseFeatures(gameState, action)
        return self.scaleFeatures(rawFeatures, role)

    def scaleFeatures(self, rawFeatures, role):
        features = Counter()
        for key, value in rawFeatures.items():
            name = role + ':' + key
            if key == 'successorScore':
                features[name] = value / 20.0
            elif 'Distance' in key or 'distance' in key:
                features[name] = value / 20.0
            else:
                features[name] = value
        return features

    def initialWeights(self):
        weights = Counter()
        weights.update({
            'offense:successorScore': 2400.0,
            'offense:distanceToFood': -280.0,
            'offense:distanceToHome': 0.0,
            'offense:ghostDistance': 500.0,
            'offense:scaredGhostDistance': -620.0,
            'offense:estimatedGhostDistance': 140.0,
            'offense:deadEndRisk': -90.0,
            'offense:capsuleDistance': -120.0,
            'offense:stop': -300.0,
            'offense:reverse': -6.0,
            'return:successorScore': 2400.0,
            'return:distanceToFood': -20.0,
            'return:distanceToHome': -480.0,
            'return:ghostDistance': 900.0,
            'return:estimatedGhostDistance': 240.0,
            'return:deadEndRisk': -130.0,
            'return:capsuleDistance': -60.0,
            'return:stop': -300.0,
            'return:reverse': -10.0,
            'defense:onDefense': 260.0,
            'defense:visibleInvaderDistance': -360.0,
            'defense:scaredInvaderDistance': -700.0,
            'defense:tooCloseToInvader': -160.0,
            'defense:scaredBlockDistance': -260.0,
            'defense:missingFoodTargetDistance': -180.0,
            'defense:estimatedInvaderDistance': -130.0,
            'defense:boundaryPatrolDistance': -100.0,
            'defense:stop': -220.0,
            'defense:reverse': -8.0,
        })
        return weights

    def selectRole(self, gameState):
        myState = gameState.getAgentState(self.index)
        visibleInvaders = self.getVisibleInvaders(gameState)
        foodLeft = len(self.getFood(gameState).asList())
        closeGhost = self.closestVisibleGhostDistance(gameState)

        if visibleInvaders and self.shouldDefendInvaders(gameState, visibleInvaders):
            return 'defense'
        if self.hasScaredEnemyGhost(gameState):
            return 'offense'
        if myState.isPacman and closeGhost is not None and closeGhost <= 2:
            return 'return'
        if self.shouldGuardWhenQuiet(gameState, foodLeft):
            return 'defense'
        return 'offense'

    def applyRoleHysteresis(self, gameState, desiredRole):
        visibleInvaders = self.getVisibleInvaders(gameState)
        emergency = bool(visibleInvaders) or desiredRole == 'return'
        if self.currentRole is None or desiredRole == self.currentRole or emergency:
            if desiredRole == self.currentRole:
                self.roleTurns += 1
            else:
                self.currentRole = desiredRole
                self.roleTurns = 1
            return desiredRole

        minTurns = 4 if self.currentRole == 'offense' else 3
        if self.roleTurns < minTurns:
            self.roleTurns += 1
            return self.currentRole

        self.currentRole = desiredRole
        self.roleTurns = 1
        return desiredRole

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

    def shouldGuardWhenQuiet(self, gameState, foodLeft):
        myState = gameState.getAgentState(self.index)
        if myState.isPacman:
            return False
        if self.missingFoodTarget is not None:
            return self.isClosestTeamAgent(gameState, self.missingFoodTarget)

        score = self.getScore(gameState)
        defendingFoodLeft = len(self.getFoodYouAreDefending(gameState).asList())
        if self.shouldLateAttack(gameState, score, defendingFoodLeft):
            return False
        if self.defaultRole == 'defense':
            return True
        keepOneBack = self.defaultRole == 'defense' or score >= 2 or foodLeft <= 4 or defendingFoodLeft <= 5
        if self.allTeamAgentsHome(gameState):
            return keepOneBack and not self.isClosestToHomePatrol(gameState)
        return keepOneBack and self.isClosestToHomePatrol(gameState)

    def shouldLateAttack(self, gameState, score, defendingFoodLeft):
        if self.defaultRole != 'defense' or defendingFoodLeft <= 5:
            return False
        if self.estimatedInvaderPositions(gameState):
            return False
        timeLeft = getattr(gameState.data, 'timeleft', 9999)
        if -2 <= score < 0 and timeLeft < 550:
            return True
        if score == 0 and timeLeft < 250:
            return True
        return False

    def allTeamAgentsHome(self, gameState):
        for teammate in self.getTeam(gameState):
            if gameState.getAgentState(teammate).isPacman:
                return False
        return True

    def isClosestTeamAgent(self, gameState, target):
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return False
        myKey = (self.getMazeDistance(myPos, target), self.index)
        for teammate in self.getTeam(gameState):
            if teammate == self.index:
                continue
            teammatePos = gameState.getAgentPosition(teammate)
            if teammatePos is None:
                continue
            teammateKey = (self.getMazeDistance(teammatePos, target), teammate)
            if teammateKey < myKey:
                return False
        return True

    def isClosestToHomePatrol(self, gameState):
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return False
        myKey = (self.distanceToHome(gameState, myPos), self.index)
        for teammate in self.getTeam(gameState):
            if teammate == self.index:
                continue
            teammatePos = gameState.getAgentPosition(teammate)
            if teammatePos is None:
                continue
            teammateKey = (self.distanceToHome(gameState, teammatePos), teammate)
            if teammateKey < myKey:
                return False
        return True

    def evaluate(self, gameState, action, role):
        return self.getQValue(gameState, action, role)

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
        features['distanceToHome'] = self.distanceToHome(successor, myPos)

        if foodList:
            features['distanceToFood'] = min(self.getMazeDistance(myPos, food) for food in foodList)
        if capsules:
            features['capsuleDistance'] = min(self.getMazeDistance(myPos, capsule) for capsule in capsules)

        scaredGhosts = self.visibleScaredGhosts(successor, minTimer=3)
        if scaredGhosts:
            reachableDistances = []
            for ghostPos, scaredTimer in scaredGhosts:
                distance = self.getMazeDistance(myPos, ghostPos)
                if distance <= scaredTimer:
                    reachableDistances.append(distance)
            if reachableDistances:
                features['scaredGhostDistance'] = min(reachableDistances)

        ghostDistances = self.visibleGhostDistances(successor, myPos)
        features['ghostDistance'] = min(min(ghostDistances), 6) if ghostDistances else 6
        if ghostDistances and min(ghostDistances) <= 1:
            features['ghostDistance'] = -10
        estimatedGhostDistance = self.closestEstimatedGhostDistance(successor, myPos)
        if estimatedGhostDistance is not None:
            features['estimatedGhostDistance'] = min(estimatedGhostDistance, 8)
        features['deadEndRisk'] = self.deadEndRisk(successor, myPos, ghostDistances,
                                                   estimatedGhostDistance)

        self.addMovementFeatures(gameState, action, features)
        return features

    def getDefenseFeatures(self, gameState, action):
        features = Counter()
        successor = self.getSuccessor(gameState, action)
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        features['onDefense'] = 0 if myState.isPacman else 1
        invaders = self.getVisibleInvaders(successor)
        if invaders:
            nearestInvaderDistance = min(
                self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders)
            if myState.scaredTimer > 0:
                features['scaredInvaderDistance'] = abs(nearestInvaderDistance - 3)
                if nearestInvaderDistance < 3:
                    features['tooCloseToInvader'] = 3 - nearestInvaderDistance
                blockTarget = self.getScaredBlockTarget(successor, [invader.getPosition()
                                                                    for invader in invaders])
                if blockTarget is not None:
                    features['scaredBlockDistance'] = self.getMazeDistance(myPos, blockTarget)
            else:
                features['visibleInvaderDistance'] = nearestInvaderDistance
        elif self.missingFoodTarget is not None:
            features['missingFoodTargetDistance'] = self.getMazeDistance(myPos, self.missingFoodTarget)
        elif self.estimatedInvaderPositions(successor):
            estimatedInvaders = self.estimatedInvaderPositions(successor)
            features['estimatedInvaderDistance'] = min(
                self.getMazeDistance(myPos, pos) for pos in estimatedInvaders)
        else:
            patrolTarget = self.getBoundaryPatrolTarget(gameState, myPos)
            features['boundaryPatrolDistance'] = self.getMazeDistance(myPos, patrolTarget)

        self.addMovementFeatures(gameState, action, features)
        return features

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

    def closestEstimatedGhostDistance(self, gameState, myPos):
        positions = []
        for opponent in self.getOpponents(gameState):
            enemy = gameState.getAgentState(opponent)
            enemyPos = enemy.getPosition()
            if enemyPos is not None:
                if not enemy.isPacman and enemy.scaredTimer <= 2:
                    positions.append(enemyPos)
                continue
            if enemy.scaredTimer > 2:
                continue
            likely = [pos for pos in self.mostLikelyOpponentPositions(opponent, limit=4)
                      if not self.isHomeSide(gameState, pos)]
            positions.extend(likely)
        if not positions:
            return None
        return min(self.getMazeDistance(myPos, pos) for pos in positions)

    def estimatedInvaderPositions(self, gameState):
        positions = []
        for opponent in self.getOpponents(gameState):
            enemy = gameState.getAgentState(opponent)
            enemyPos = enemy.getPosition()
            if enemyPos is not None:
                if enemy.isPacman:
                    positions.append(enemyPos)
                continue
            likely = [pos for pos in self.mostLikelyOpponentPositions(opponent, limit=4)
                      if self.isHomeSide(gameState, pos)]
            positions.extend(likely)
        return positions

    def mostLikelyOpponentPositions(self, opponent, limit=3):
        belief = self.opponentBeliefs.get(opponent, Counter())
        if not belief:
            return []
        ranked = sorted(belief.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
        return [pos for pos, value in ranked[:limit]]

    def updateOpponentBeliefs(self, gameState):
        distances = gameState.getAgentDistances() or []
        myPos = gameState.getAgentPosition(self.index)
        teamPositions = [gameState.getAgentPosition(agent) for agent in self.getTeam(gameState)]
        teamPositions = [pos for pos in teamPositions if pos is not None]

        for opponent in self.getOpponents(gameState):
            visiblePos = gameState.getAgentPosition(opponent)
            if visiblePos is not None:
                belief = Counter()
                belief[visiblePos] = 1.0
                self.opponentBeliefs[opponent] = belief
                continue

            previous = self.opponentBeliefs.get(opponent, Counter())
            if previous:
                candidates = set()
                for pos in previous:
                    candidates.update(self.legalNeighbors(pos))
            else:
                candidates = set(self.legalPositions)

            noisy = distances[opponent] if opponent < len(distances) else None
            filtered = Counter()
            for pos in candidates:
                if self.wouldBeVisible(pos, teamPositions):
                    continue
                weight = 1.0
                if noisy is not None and myPos is not None:
                    trueDistance = manhattanDistance(myPos, pos)
                    if abs(trueDistance - noisy) > 6:
                        continue
                    weight = 7 - abs(trueDistance - noisy)
                filtered[pos] = max(weight, 1.0)

            if not filtered:
                for pos in self.legalPositions:
                    if not self.wouldBeVisible(pos, teamPositions):
                        filtered[pos] = 1.0
            self.normalizeBelief(filtered)
            self.opponentBeliefs[opponent] = filtered

    def normalizeBelief(self, belief):
        total = float(sum(belief.values()))
        if total <= 0.0:
            return
        for pos in list(belief.keys()):
            belief[pos] /= total

    def wouldBeVisible(self, pos, teamPositions):
        return any(manhattanDistance(pos, teamPos) <= 5 for teamPos in teamPositions)

    def isHomeSide(self, gameState, pos):
        middle = gameState.data.layout.width // 2
        if self.red:
            return pos[0] < middle
        return pos[0] >= middle

    def deadEndRisk(self, gameState, myPos, ghostDistances, estimatedGhostDistance):
        if myPos is None:
            return 0
        depth = self.deadEndDepth.get(nearestPoint(myPos), 0)
        if depth <= 0:
            return 0
        nearestThreat = None
        if ghostDistances:
            nearestThreat = min(ghostDistances)
        if estimatedGhostDistance is not None:
            nearestThreat = min(nearestThreat, estimatedGhostDistance) if nearestThreat is not None else estimatedGhostDistance
        if nearestThreat is None or nearestThreat > depth + 4:
            return 0
        pressure = max(1, depth + 5 - nearestThreat)
        return pressure

    def distanceToHome(self, gameState, pos):
        if pos is None or not self.patrolPoints:
            return 0
        return min(self.getMazeDistance(pos, point) for point in self.patrolPoints)

    def getBoundaryPatrolTarget(self, gameState, myPos):
        if self.missingFoodTarget is not None:
            return self.missingFoodTarget
        return self.patrolTarget

    def getScaredBlockTarget(self, gameState, invaderPositions):
        if not self.patrolPoints or not invaderPositions:
            return None
        defendingFood = self.getFoodYouAreDefending(gameState).asList()
        scored = []
        for point in self.patrolPoints:
            invaderYGap = min(abs(point[1] - invader[1]) for invader in invaderPositions)
            foodYGap = 0
            if defendingFood:
                foodYGap = min(abs(point[1] - food[1]) for food in defendingFood)
            scored.append((invaderYGap, foodYGap, point[1], point))
        return min(scored)[3]

    def updatePatrolTarget(self, gameState):
        if self.missingFoodTarget is not None or not self.patrolPoints:
            return
        myPos = gameState.getAgentPosition(self.index)
        if myPos is None:
            return
        rankedTargets = self.getFoodDensityPatrolTargets(gameState, myPos)
        if not rankedTargets:
            return
        if self.patrolTarget not in rankedTargets:
            self.patrolIndex = 0
            self.patrolTarget = rankedTargets[0]
            return
        if self.getMazeDistance(myPos, self.patrolTarget) <= 1:
            currentRank = rankedTargets.index(self.patrolTarget)
            self.patrolIndex = (currentRank + 1) % min(3, len(rankedTargets))
            self.patrolTarget = rankedTargets[self.patrolIndex]

    def getFoodDensityPatrolTargets(self, gameState, myPos):
        defendingFood = self.getFoodYouAreDefending(gameState).asList()
        if not defendingFood:
            return list(self.patrolPoints)

        scored = []
        for point in self.patrolPoints:
            density = 0.0
            sameBand = 0
            for food in defendingFood:
                yGap = abs(point[1] - food[1])
                if yGap <= 3:
                    sameBand += 1
                density += 1.0 / (1.0 + yGap)
            recentPressure = 0.0
            for food, ttl in self.recentMissingFood:
                yGap = abs(point[1] - food[1])
                recentPressure += (ttl / 20.0) / (1.0 + yGap)
            scored.append((-sameBand,
                           -density,
                           -recentPressure,
                           self.getMazeDistance(myPos, point),
                           point[1],
                           point))
        scored.sort()
        return [point for _, _, _, _, _, point in scored]

    def _initialPatrolIndex(self, gameState):
        if not self.patrolPoints:
            return 0
        centerY = (gameState.data.layout.height - 1) / 2.0
        team = sorted(self.getTeam(gameState))
        offset = -2 if team.index(self.index) == 0 else 2
        targetY = centerY + offset
        target = min(self.patrolPoints,
                     key=lambda point: (abs(point[1] - targetY), point[1]))
        return self.patrolPoints.index(target)

    def _computeBoundaryPoints(self, gameState):
        width = gameState.data.layout.width
        height = gameState.data.layout.height
        x = width // 2 - 1 if self.red else width // 2
        points = [(x, y) for y in range(1, height - 1) if not gameState.hasWall(x, y)]
        if len(points) > 4:
            points = points[1:-1]
        return points

    def _computeLegalPositions(self, gameState):
        walls = gameState.getWalls()
        positions = []
        for x in range(walls.width):
            for y in range(walls.height):
                if not walls[x][y]:
                    positions.append((x, y))
        return positions

    def _computeDeadEndDepths(self, gameState):
        depths = {}
        for pos in self.legalPositions:
            neighbors = self.legalNeighbors(pos, includeStop=False)
            if len(neighbors) != 1:
                continue
            path = []
            previous = None
            current = pos
            while current not in path:
                path.append(current)
                nextOptions = [neighbor for neighbor in self.legalNeighbors(current, includeStop=False)
                               if neighbor != previous]
                if not nextOptions:
                    break
                nextPos = nextOptions[0]
                if len(self.legalNeighbors(nextPos, includeStop=False)) >= 3:
                    break
                previous = current
                current = nextPos
            length = len(path)
            for offset, pathPos in enumerate(path):
                depths[pathPos] = max(depths.get(pathPos, 0), length - offset)
        return depths

    def legalNeighbors(self, pos, includeStop=True):
        neighbors = Actions.getLegalNeighbors(pos, self.walls)
        if includeStop:
            return neighbors
        return [neighbor for neighbor in neighbors if neighbor != pos]

    def _updateMissingFoodTarget(self, gameState):
        self.recentMissingFood = [(food, ttl - 1) for food, ttl in self.recentMissingFood
                                  if ttl > 1]
        currentFood = self.getFoodYouAreDefending(gameState).asList()
        if self.previousDefendingFood:
            missing = list(set(self.previousDefendingFood) - set(currentFood))
            myPos = gameState.getAgentPosition(self.index)
            if missing and myPos is not None:
                self.missingFoodTarget = min(missing,
                                             key=lambda food: self.getMazeDistance(myPos, food))
                for food in missing:
                    self.recentMissingFood.append((food, 20))
        if self.missingFoodTarget == myPosOrNone(gameState, self.index):
            self.missingFoodTarget = None

    def _candidateActions(self, gameState, legal, role):
        actions = [action for action in legal if action != Directions.STOP]
        if not actions:
            return legal
        saferActions = self.filterDeadEndActions(gameState, actions, role)
        if saferActions:
            actions = saferActions
        reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        noReverse = [action for action in actions if action != reverse]
        if noReverse and role != 'defense':
            return noReverse
        return actions

    def filterDeadEndActions(self, gameState, actions, role):
        if role == 'defense':
            return actions
        kept = []
        for action in actions:
            if self.outOfTime():
                break
            try:
                successor = self.getSuccessor(gameState, action)
                myState = successor.getAgentState(self.index)
                myPos = myState.getPosition()
                if not myState.isPacman or self.deadEndDepth.get(nearestPoint(myPos), 0) == 0:
                    kept.append(action)
                    continue
                visibleThreats = self.visibleGhostDistances(successor, myPos)
                estimatedThreat = self.closestEstimatedGhostDistance(successor, myPos)
                nearestThreat = min(visibleThreats) if visibleThreats else estimatedThreat
                if estimatedThreat is not None and nearestThreat is not None:
                    nearestThreat = min(nearestThreat, estimatedThreat)
                depth = self.deadEndDepth.get(nearestPoint(myPos), 0)
                if nearestThreat is None or nearestThreat > depth + 3:
                    kept.append(action)
            except Exception:
                kept.append(action)
        return kept if kept else actions

    def safeFallbackAction(self, gameState, legal):
        actions = [action for action in legal if action != Directions.STOP]
        if not actions:
            return Directions.STOP if Directions.STOP in legal else legal[0]

        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myState.isPacman and self.closestVisibleGhostDistance(gameState) is not None:
            target = min(self.patrolPoints, key=lambda point: self.getMazeDistance(myPos, point))
        elif myState.isPacman and self.getFood(gameState).asList():
            target = min(self.getFood(gameState).asList(), key=lambda food: self.getMazeDistance(myPos, food))
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
            fallbackRole = 'return' if myState.isPacman and self.closestVisibleGhostDistance(gameState) is not None else 'offense'
            scored.append((distance, self.actionRank(action, fallbackRole, gameState), action))
        return min(scored)[2]

    def _firstByActionOrder(self, actions, role=None, gameState=None):
        return min(actions, key=lambda action: self.actionRank(action, role, gameState))

    def actionRank(self, action, role=None, gameState=None):
        if action == Directions.STOP:
            return 4

        attack = Directions.EAST if self.red else Directions.WEST
        home = Directions.WEST if self.red else Directions.EAST

        if role == 'defense':
            order = [Directions.NORTH, home, Directions.SOUTH, attack, Directions.STOP]
        elif role == 'return':
            order = [Directions.NORTH, home, Directions.SOUTH, attack, Directions.STOP]
        else:
            order = [Directions.NORTH, attack, Directions.SOUTH, home, Directions.STOP]
        return order.index(action) if action in order else len(order)

    def outOfTime(self):
        return self.moveDeadline > 0.0 and time.time() >= self.moveDeadline


def myPosOrNone(gameState, index):
    try:
        return gameState.getAgentPosition(index)
    except Exception:
        return None


class OffensiveReflexAgent(ReflexCaptureAgent):
    defaultRole = 'offense'

    def shouldDefendInvaders(self, gameState, visibleInvaders):
        myState = gameState.getAgentState(self.index)
        myPos = myState.getPosition()
        if myPos is None:
            return False

        if myState.isPacman and self.distanceToHome(gameState, myPos) > 4:
            return False

        if len(visibleInvaders) >= 2:
            return True

        myInvaderDistance = min(self.getMazeDistance(myPos, invader.getPosition())
                                for invader in visibleInvaders)
        if not myState.isPacman and myInvaderDistance <= 4:
            return True

        defenderInfos = []
        for teammate in self.getTeam(gameState):
            if teammate == self.index:
                continue
            teammateState = gameState.getAgentState(teammate)
            teammatePos = teammateState.getPosition()
            if teammatePos is None:
                continue
            defenderDistance = min(self.getMazeDistance(teammatePos, invader.getPosition())
                                   for invader in visibleInvaders)
            defenderInfos.append((defenderDistance, teammateState.scaredTimer))

        if not defenderInfos:
            return myInvaderDistance <= 6

        closestDefenderDistance, defenderScaredTimer = min(defenderInfos)
        if defenderScaredTimer > 0:
            return myInvaderDistance <= 7
        if closestDefenderDistance > 6 and myInvaderDistance <= 6:
            return True
        return myInvaderDistance + 3 < closestDefenderDistance


class DefensiveReflexAgent(ReflexCaptureAgent):
    defaultRole = 'defense'

    def shouldDefendInvaders(self, gameState, visibleInvaders):
        return True
