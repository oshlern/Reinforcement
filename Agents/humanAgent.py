from abstractAgent import Agent


class Human(Agent):

    def action(self, env, state, actions):
        # print "Current State"
        # print state
        userInput = raw_input('Your Move: ')
        # Cleaning input to standardized form
        # userInput.replace(userInput, r"$[0-9]* $[0-9]*", r"\1,\2") # fix for higher dimension
        # userInput = re.replace(userInput, r"\[\] ", "")
        position = userInput.split(',')
        for i in range(len(position)):
            position[i] = int(position[i])
        return tuple(position)
