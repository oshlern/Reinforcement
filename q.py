import numpy as np
import random, copy

# def normalize(weights):
#     minW, maxW = np.amin(weights), np.amax(weights)
#     return np.divide(np.subtract(weights, minW), np.sum(weights))

from abc import ABCMeta, abstractmethod, abstractproperty


class Q:
    __metaclass__ = ABCMeta
    def __init__(self, learningRate=0.1, discountRate=0.9):
        self.learningRate, self.discountRate = learningRate, discountRate
        self.initializeQ()

    @abstractmethod
    def initializeQ(self):
        self.q = {}

    @abstractmethod
    def getQ(self, state, action):
        return 0

    @abstractmethod
    def setQ(self, state, action, val):
        pass

    # q = abstractproperty(getQ, setQ)

    def update(self, state, action, val):
        currentVal = self.get(state, action)
        newVal = np.add(currentVal, np.multiply(self.learningRate, np.subtract(val, currentVal)))
        self.setQ(state, move, newVal)


class QMatrix(Q):
    # def __init__(self, learningRate=0.1, discountRate=0.9, randomness=0.2):
    #     super(Q, self).__init__(learningRate, discountRate)

    def initializeQ(self):
        self.q = {}

    def getQ(self, state, action):
        return 0

    def setQ(self, state, action, val):
        return 0

mat = QMatrix()
print mat.q

#     def updateQ(state, nextState, possibleActions, action, reward):
#     #     Q(state, action) = R(state, action) + Gamma * Max[Q(next state, all actions)]
#         strNextState = str(nextState)
#         maxNext = np.amax([self.Q.get((strNextState, action), 0) for action in possibleActions])
#         newVal = reward + self.discountRate * maxNext
#         currentVal = self.Q.get((str(state), move), 0)
#         self.Q[(strState, move)] = currentVal + self.learningRate * (val - currentVal)

#     def action(self, env, state, actions):
#          strState = str(state)
#          if np.random() < self.randomness:
#             move = np.random.choice(actions)
#         else: # a random choice with a weighted distribution
#             strState = str(state)
#             vals = [self.Q.get((strState, action), 0) for action in actions]
#             minVal = np.amin(vals)
#             normVals = np.divide(np.subtract(vals, minVal), np.sum(vals))
#             action = np.random.choice(actions, p=normVals)
#             # move = self.bestQ(state, actions)
#         reward, nextState, possibleActions = env.act(action)
#         #  TODO: NEXT STATE
#         strNextState = str(nextState)
#         maxNext = np.amax([self.Q.get((strNextState, action), 0) for action in possibleActions])
#         newVal = reward + self.discountRate * maxNext
#         currentVal = self.Q.get((str(state), move), 0)
#         self.Q[(strState, move)] = currentVal + self.learningRate * (val - currentVal)
#         return move

#     def action(self, env, state, actions):
#         bestVal = np.amax([self.Q.get((strNextState, action), 0) for action in possibleActions])
#         currentVal = self.Q.get((str(state), move), 0)
#         self.Q[(strState, move)] = currentVal + self.learningRate * (bestVal + reward - currentVal)

#     # Look further ahead?

#     def bestQ(self, state, actions):
#         vals = [self.Q.get((str(state), action), 0) for action in actions]
#         bestVal = np.amax(vals)
#         return bestVal, actions[vals.index(bestValIndex)]

#     # def trainAction(self, state, possibleMoves, numEmpty):
#     #     move = possibleMoves.pop(random.randrange(numEmpty)) # make not always random? # possibly implement a random choice with weighted distribution?
#     #     if len(possibleMoves) <= 1:
#     #         return move
#     #     newState = np.negative(self.nextState(state, move))
#     #     newMoves = copy.deepcopy(possibleMoves)
#     #     newMoves.remove(move)

#     def train(self, state, oldState, move, possibleMoves):
#         strState, strOldState = str(state), str(oldState)
#         val = np.amax([self.Q.get((strState, possibleMove), 0) for possibleMove in possibleMoves])
#         val = np.multiply(val, np.negative(self.discountRate))
#         currentVal = self.Q.get((strOldState, move), 0)
#         self.Q[(strOldState, move)] = np.add(currentVal, np.multiply(self.learningRate, np.subtract(val, currentVal)))

    

#     def noLearnAction(self, state, actions):
#         return self.bestQ(state, actions) 


#     # def end(self, reward, winner, playerNum, state, move):
#     #     state = self.boolState(state, playerNum)
#     #     oldState = np.negative(self.nextState(state, move, change=0))
#     #     self.R[(str(oldState), move)] = reward
#     #     self.updateQVal(oldState, move, reward)
#         # if self.debugMode: print self.Q.values()

# import numpy as np
# import re, random, copy
# from board import Board
# from Agents.compileAgents import Agent, Human, RandomChoose, Minimax, compileAgents


# class Q(Agent):
#     def __init__(self, boardParams, learningRate=0.3, discountRate=0.9, randomness=0.2, debugMode=False):
#         super(Q, self).__init__(boardParams, debugMode=debugMode)
#         self.learningRate, self.discountRate, self.randomness = learningRate, discountRate, randomness
#         self.Q, self.R = {}, {}
#         self.withLearn = True
#         self.debugMode = debugMode

#     def updateQVal(self, state, move, val):
#         currentVal = self.Q.get((str(state), move), 0)
#         self.Q[(str(state), move)] = currentVal + self.learningRate * (val - currentVal)

#     def action(self, board, state, turn, playerNum, possibleMoves):
#         state = self.boolState(state, playerNum)
#         move = (0, 0)
#         if random.random() < self.randomness:
#             move = random.choice(possibleMoves)
#         else:
#             move = self.bestQ(state, possibleMoves) # possibly implement a random choice with weighted distribution?
#         if self.withLearn: self.updateQ(state, possibleMoves, move)
#         return move

#     def updateQ(self, state, possibleMoves, move):
#         if self.debugMode: print "move" + str(move)
#         if len(possibleMoves) <= 1:
#             if self.debugMode: print "moves", possibleMoves
#             return
#         newState = np.negative(self.nextState(state, move))
#         newMoves = copy.deepcopy(possibleMoves)
#         newMoves.remove(move)
#         val = self.bestQ(newState, newMoves, returnVal=True) # Opponent's best move
#         val *= -self.discountRate
#         if self.debugMode: print val
#         self.updateQVal(state, move, val)

#     def end(self, reward, winner, playerNum, state, move):
#         state = self.boolState(state, playerNum)
#         oldState = np.negative(self.nextState(state, move, change=0))
#         self.R[(str(oldState), move)] = reward
#         self.updateQVal(oldState, move, reward)
#         # if self.debugMode: print self.Q.values()

#     def train(self, iterations, withRand=True):
#         agents = [self, RandomChoose(self.boardParams)]
#         train = Board(self.boardParams)
#         train.setAgents(agents)
#         trainWins = train.runGames(numGames=iterations)
#         return trainWins  #make efficient training

#     def test(self, numGames, withRand=False, withLearn=False):
#         if withLearn:
#             wasWin = copy.deepcopy(self.withLearn)
#             self.withLearn = False
#         if withRand:
#             wasRand = copy.deepcopy(self.randomness)
#             self.randomness = 0
#         agents = [self, RandomChoose(self.boardParams)]
#         test = Board(self.boardParams)
#         test.setAgents(agents)
#         firstWins = test.runGames(numGames=numGames, shuffle=False)
#         # agents = [agents[1], agents[0]]
#         # test.setAgents(agents)
#         # secondWins = test.runGames(numGames=numGames, shuffle=False)
#         if withLearn:
#             self.withLearn = wasWin
#         if withRand:
#             self.randomness = wasRand
#         print len(firstWins)
#         return firstWins.count(1), firstWins.count(0)

#         # return firstWins.count(1) + secondWins.count(2), firstWins.count(0) + second.count(0)

#     def interactiveTest(self):
#         agents = [Human(self.boardParams), self]
#         test = Board(self.boardParams, debugMode=True)
#         test.setAgents(agents)
#         test.runGames()

#     def efficientTrain(self, numGames):
#         # Just 1st player rn
#         winners = []
#         for i in xrange(numGames):
#             if i % 1000 == 0:
#                 print "RUNNING GAME: ", i
#             # random.shuffle(agents)
#             # winners.append(self.run())
#             currentPlayer, winner, move = 1, 0, (0,0)
#             state = np.zeros(tuple([self.size]*self.dimension), dtype=np.int)
#             emptyIndices = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
#             for turn in xrange(4):
#                 move = random.choice(emptyIndices)
#                 if random.random() > 0.8: #self.randomness:
#                     bestVal = -100000000
#                     strState = str(state)
#                     for tempMove in emptyIndices:
#                         val = self.Q.get((strState, tempMove), 0)
#                         if val > bestVal:
#                             move, bestVal = tempMove, val
#                 oldStateStr = str(state)
#                 state[move] = 1
#                 emptyIndices.remove(move)
#                 if state[(move[0], 0)] == state[(move[0], 1)] == state[(move[0], 2)]:
#                     winner = 1
#                     break
#                 if state[(0, move[1])] == state[(1, move[1])] == state[(2, move[1])]:
#                     winner = 1
#                     break
#                 if move[0] == move[1]:
#                     if state[(0, 0)] == state[(1, 1)] == state[(2, 2)]:
#                         winner = 1
#                         break
#                 if move[0] == 2 - move[1]:
#                     if state[(0, 2)] == state[(1, 1)] == state[(2, 0)]:
#                         winner = 1
#                         break

#                 newStateStr = str(np.negative(state))
#                 bestVal = 0
#                 for move in emptyIndices:
#                     val = self.Q.get((newStateStr, move), 0)
#                     if val > bestVal:
#                         bestVal = val
#                 val = np.multiply(-self.discountRate, bestVal)
#                 currentVal = self.Q.get((oldStateStr, move), 0)
#                 self.Q[(oldStateStr, move)] = currentVal + self.learningRate * (val - currentVal)

#                 random.shuffle(emptyIndices)
#                 move = emptyIndices.pop()
#                 state[move] = -1
#                 if state[(move[0], 0)] == state[(move[0], 1)] == state[(move[0], 2)]:
#                     winner = -1
#                     break
#                 if state[(0, move[1])] == state[(1, move[1])] == state[(2, move[1])]:
#                     winner = -1
#                     break
#                 if move[0] == move[1]:
#                     if state[(0, 0)] == state[(1, 1)] == state[(2, 2)]:
#                         winner = -1
#                         break
#                 if move[0] == 2 - move[1]:
#                     if state[(0, 2)] == state[(1, 1)] == state[(2, 0)]:
#                         winner = -1
#                         break
#             oldState = np.negative(state)
#             oldState[move] = 0
#             strState = str(oldState)
#             self.Q[(strState, move)] = winner
#             winners.append(winner)
#         return winners




