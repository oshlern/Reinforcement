import tictac, q
import numpy as np
agent = q.QDeep()
board = tictac.Board(debugMode=True)
print agent.q
print "INITIALIZED"
board.train(agent, numGames=500)
print agent.q
board.interactiveTest(agent)


# agent = np.load('qAgent.npy').flatten()[0]
# np.save('qAgent.npy', agent)