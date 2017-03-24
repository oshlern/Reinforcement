import tictac, q
agent = q.QMatrix()
board = tictac.Board(debugMode=True)
board.train(agent, numGames=40000)
print agent.q