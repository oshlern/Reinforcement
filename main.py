import tictac, q
import numpy as np
agent = np.load('qDeep.npy').flatten()[0]
# agent = q.QDeep()
board = tictac.Board(debugMode=True)
# print agent.q
print "INITIALIZED"
board.train(agent, numGames=50000)
np.save('qDeep.npy', agent)
print board.test(agent)
agent = q.QDeep()
print board.test(agent)


# print agent.q
# board.interactiveTest(agent)


# agent = np.load('qAgent.npy').flatten()[0]
