import numpy as np
import random, copy
from sklearn.linear_model import MLPRegressor
from abc import ABCMeta, abstractmethod, abstractproperty
from Agents.abstractAgent import Agent

# def normalize(weights):
#     minW, maxW = np.amin(weights), np.amax(weights)
#     return np.divide(np.subtract(weights, minW), np.sum(weights))


class Q(Agent):
    __metaclass__ = ABCMeta
    def __init__(self, learningRate=0.1, discountRate=0.9, params=None, weighted=True, randomness = 0.3):
        self.learningRate, self.discountRate, self.params, self.weighted, self.randomness = learningRate, discountRate, params, weighted, randomness
        self.initializeQ()
        self.R = {}

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

    def update(self, state, action, val): #move into actionUpdate
        currentVal = self.getQ(state, action)
        newVal = np.add(currentVal, np.multiply(self.learningRate, np.subtract(val, currentVal)))
        self.setQ(state, action, newVal)

    def actionUpdate(self, state, nextState, nextActions, action, reward=0):
        nextQs = [self.getQ(nextState, nextAction) for nextAction in nextActions]
        maxNext = np.amax(nextQs)
        val = np.add(reward, np.multiply(self.discountRate, maxNext))
        self.update(state, action, val)

    def action(self, env, state, actions):
        if np.random.random() < self.randomness:
            action = actions[np.random.randint(len(actions))]
        elif self.weighted: # a random choice with a weighted distribution
            action = self.weightedAction(state, actions)
        else:
            action = 0 #TODO: best action
        reward, nextState, nextActions = env.act(action)
        if len(nextActions) != 0:
            self.actionUpdate(state, nextState, nextActions, action, reward=reward)
        else:
            self.update(state, action, reward)
        
    def weightedAction(self, state, actions):
        vals = [self.getQ(state, action) for action in actions]
        minVal = np.amin(vals)
        if minVal == np.amax(vals) == 0:
            action = np.random.choice(len(actions))
        else:
            normVals = np.divide(np.subtract(vals, minVal), np.sum(vals))
            print actions, type(actions), normVals
            action = np.random.choice(len(actions), p=normVals)
        return actions[action]

    def bestAction(self, state, actions):
        vals = [self.getQ(state, action) for action in actions]
        maxVal = np.amax(vals)
        action = actions[vals.index(maxVal)]
        return action

    def noLearnAction(self, state, actions):
        return self.bestAction(state, actions)

    def end(self, state, reward):
        self.R[str(state)] = reward



class QMatrix(Q):
    # def __init__(self, learningRate=0.1, discountRate=0.9, randomness=0.2):
    #     super(Q, self).__init__(learningRate, discountRate)
    # stateIsList = True
    # actionIsList = False

    def initializeQ(self):
        self.q = {}

    def getQ(self, state, action):
        if type(state) == list or type(state) == np.ndarray:
            state = str(state)
        if type(action) == list:
            action = str(action)
        return self.q.get((state, action), 0)

    def setQ(self, state, action, val):
        if type(state) == list or type(state) == np.ndarray:
            state = str(state)
        if type(action) == list:
            action = str(action)
        self.q[(state, action)] = val

class QDeep(Q):
    def __init__(self, learningRate=0.1, discountRate=0.9):
        super(QDeep, self).__init__(learningRate=learningRate, discountRate=discountRate)

    def initializeQ(self, hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        self.q = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter, shuffle=shuffle, random_state=random_state, tol=tol, verbose=verbose, warm_start=warm_start, momentum=momentum, nesterovs_momentum=nesterovs_momentum, early_stopping=early_stopping, validation_fraction=validation_fraction, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)

    def getQ(self, state, action):
        if type(state) == list:
            state = np.array(state)
        X = np.array([np.append(state.flatten(), action)]).T
        try:
            return self.q.predict(np.array([X])) #doesn't work if model hasn't been fit yet
        except:
            return 0

    def setQ(self, state, action, val):
        if type(state) == list:
            state = np.array(state)
        X = np.ravel(np.append(state.flatten(), action))
        self.q.partial_fit(np.array([X]), np.array([val]))

# np.random.seed(0)
# y = np.random.randn(n_samples)
# X = np.random.randn(n_samples, n_features)
# clf = linear_model.SGDRegressor()
# clf.fit(X, y)

# MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)[source]

# mat = QDeep()
# mat.setQ([1],2,3)
# print mat.getQ([1],2)




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




