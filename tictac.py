import numpy as np
import copy, random
from Agents.randomAgent import RandomChoose
boardParams = [2, 3, 2, 10]


class Board:
    def __init__(self, boardParams=boardParams, debugMode=False):
        if type(boardParams) == hash:
            self.numPlayers, self.size, self.dimension, self.limit = boardParams["numPlayers"], boardParams["size"], boardParams["dimension"], boardParams["limit"] # **boardParams
        elif type(boardParams) == list:
            self.numPlayers, self.size, self.dimension, self.limit = boardParams[0], boardParams[1], boardParams[2], boardParams[3] # *boardParams
        else:
            self.numPlayers, self.size, self.dimension, self.limit = boardParams
        self.debugMode = debugMode
        self.rowIndices, self.allIndices = self.findRowIndices() # A list of all the rows, where each row is represented by a list of the positions of its elements
        self.reset()

    def reset(self):
        self.currentPlayer = 1
        self.state = np.zeros(tuple([self.size]*self.dimension), dtype=np.int)
        self.emptyIndices = copy.deepcopy(self.allIndices)
        self.ended = False

    def display(self):
        print self.state

    def findRowIndices(self):
        # Compiles row indices as described in __init__
        # Currently only 2d
        rows, diag1, diag2, total = [], [], [], []
        for i in range(self.size):
            row, column = [], []
            for j in range(self.size):
                total.append((i,j))
                row.append((i,j))
                column.append((j,i))
            rows.append(row)
            rows.append(column)
            diag1.append((i,i))
            diag2.append((i,self.size-i-1))
        rows.append(diag1)
        rows.append(diag2)
        return rows, total

    def act(self, action): # return reward, next state, next moves
        if action not in self.emptyIndices: # Valid move # handle 
            return False, False, False
        self.state[action] = self.currentPlayer
        self.emptyIndices.remove(action)
        self.acted = True
        if self.checkWinSpecific(self.state, action):
            self.end(self.currentPlayer)
            return 1, self.state, self.emptyIndices
        elif len(self.emptyIndices) == 0:
            self.end(0)
            return 0, self.state, self.emptyIndices
        else:
            return False, self.state, self.emptyIndices

    def run(self):
        self.reset()
        self.currentPlayer = 1
        for turn in range(9):
            if self.debugMode: print "Player {}, make your move.".format(self.currentPlayer)
            boardInfo = (self, self.state, self.emptyIndices)
            self.acted = False
            action = self.agents[self.currentPlayer-1].action(*boardInfo)
            if not self.acted:
                self.act(action)
            if self.ended:
                break
            if self.debugMode: self.display()
            self.currentPlayer = (self.currentPlayer % self.numPlayers) + 1
        return self.winner

    def trainAct(self, action):
        if action not in self.emptyIndices: # Valid move
            return False, False, False
        self.state[action] = self.currentPlayer
        self.emptyIndices.remove(action)
        numEmpty = len(emptyIndices)
        if numEmpty == 0:
            self.end(0)
            return 0, self.state, self.emptyIndices
        elif self.checkWinSpecific(self.state, action):
            self.end(self.currentPlayer)
            return 1, self.state, self.emptyIndices
        else:
            self.currentPlayer = (self.currentPlayer % self.numPlayers) + 1
        randAction = self.emptyIndices.pop(np.random.randint(numEmpty))


    def end(self, winner):
        self.ended = True
        if winner == 0:
            if self.debugMode: print "TIE"
            for agent in self.agents:
                agent.end(self.state, 0)
        else:
            if self.debugMode: print "WINNER: {}".format(winner)
            self.agents[winner-1].end(self.state, 1)
            for agent in self.agents[:winner-1] + self.agents[winner:]:
                agent.end(self.state, -1)
        self.winner = winner

    def checkWin(self):
        # TODO: Optimize knowing the last move and player who made it, only check the appropriate rows. If all full, run checkTie()
        # Checks every row for winner
        for row in self.rowIndices:
            player = self.state[row[0]]
            if player != 0:
                if all(map(lambda i: self.state[i] == player, row[0:])):
                    return int(player)
        # Checks if there is empty room left, and if there isn't, declares a tie (represented as 'player 0' winnning)
        # TODO: Can be optimized - just do when turns run out. Also, can use info from the for loop above.
        if len(self.emptyIndices) == 0:
            return 0

        return "no winner"

    def printEnd(self, winner):
        if winner == 0:
            print "TIE"
        else:
            print winner, "WON"
        self.display()

    
    def runGames(self, numGames=100, shuffle=True):
        winners = []
        for i in xrange(numGames):
            if i % 100 == 99:
                print "RUNNING GAME: ", i+1
            if shuffle: random.shuffle(self.agents)
            winners.append(self.run())
        if self.debugMode: print winners
        return winners

    def interactiveTest(self, agent):
        from Agents.humanAgent import Human
        oldMode = copy.copy(self.debugMode)
        self.debugMode = True
        self.agents = [Human(boardParams), agent]
        self.run()
        self.debugMode = oldMode

    def test(self, agent, numGames=100, withRand=False, withLearn=False): #make efficient like train
        if withLearn:
            wasWin = copy.deepcopy(agent.withLearn)
            agent.withLearn = False
        if withRand:
            wasRand = copy.deepcopy(agent.randomness)
            agent.randomness = 0
        wasDebug = copy.deepcopy(self.debugMode)
        self.debugMode = False
        agents = [agent, RandomChoose(boardParams)]
        self.agents = agents
        firstWins = self.runGames(numGames=numGames, shuffle=False)
        agents = [agents[1], agents[0]]
        self.agents = agents
        secondWins = self.runGames(numGames=numGames, shuffle=False)
        if withLearn:
            agent.withLearn = wasWin
        if withRand:
            agent.randomness = wasRand
        self.debugMode = wasDebug
        print firstWins, secondWins
        return [("wins", firstWins.count(1), secondWins.count(2)), ("losses", firstWins.count(2), secondWins.count(1)), ("ties", firstWins.count(0), secondWins.count(0))]

    def checkWinSpecific(self, state, move):
        if state[(move[0], 0)] == state[(move[0], 1)] == state[(move[0], 2)]:
            return True
        if state[(0, move[1])] == state[(1, move[1])] == state[(2, move[1])]:
            return True
        if move[0] == move[1]:
            if state[(0, 0)] == state[(1, 1)] == state[(2, 2)]:
                return True
        if move[0] == 2 - move[1]:
            if state[(0, 2)] == state[(1, 1)] == state[(2, 0)]:
                return True
        return False

    def train(self, agent, numGames=1000):
        self.setAgent(agent)
        self.debugMode = False
        return self.runGames(numGames)

    def setAgent(self, agent):
        self.agents = [agent, RandomChoose(boardParams)]

