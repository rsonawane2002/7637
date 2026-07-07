class SemanticNetsAgent:
    def __init__(self):
        #If you want to do any initial processing, add it here.
        pass

    def solve(self, initial_sheep, initial_wolves):
        #Add your code here! Your solve method should receive
        #the initial number of sheep and wolves as integers,
        #and return a list of 2-tuples that represent the moves
        #required to get all sheep and wolves from the left
        #side of the river to the right.
        #
        #If it is impossible to move the animals over according
        #to the rules of the problem, return an empty list of
        #moves.

        #initial sheep + wolves on left side, 0 = left(boat side), 1 = right (boat side)
        start_state = (initial_sheep, initial_wolves, 0)

        #goal: zero sheep + wolves on left size, and boat on right side (1)
        goal_state = (0, 0, 1)


        #start of BFS. store curr state + an array of every move we made so far
        queue = []
        queue.append((start_state, []))
        
        #keep track of visited states
        visited_states = set()
        visited_states.add(start_state)

        #begin BFS
        while len(queue) > 0:

            #get current state and moves so far, get how many
            #sheep + wolves are on the left side and where boat is
            current_state, moves_so_far = queue.pop(0)
            current_sheep_left = current_state[0]
            current_wolves_left = current_state[1]
            current_boat_side = current_state[2]

            #if we reach goal, return our set of moves
            if current_state == goal_state:
                return moves_so_far
            
            #if boat on left, the sheep and wolves that we can "manipulate"
            #is all the ones we have availaible on left side. If boat is on
            #right side however, its just the amount of sheep + wolves on right side
            #which we get by subtracting starting number of sheep + how many are 
            # on the left side

            if current_boat_side == 0:
                sheep_available = current_sheep_left
                wolves_available = current_wolves_left
        
            else:
                sheep_available = initial_sheep - current_sheep_left
                wolves_available = initial_wolves - current_wolves_left
            
            #helper function: given how many sheep + wolves are available on the boat's
            #current side, go through every way to load the boat that's physically legal
            #(1-2 animals, not more than what's available). This only checks boat capacity
            #rules, it doesn't know about states or the wolves outnumber-sheep rule.
            #That check happens separately below, once each move is turned into a new state.
            possible_moves = self.get_possible_moves(sheep_available, wolves_available)


            for move in possible_moves:

                #iterate over all possible moves, and find number of sheep + wolves in boat
                #for each move
                sheep_to_move = move[0]
                wolves_to_move = move[1]

                #if on the left side, then we just remove number of sheep and wolves in boat
                #from the current left side numbers, and flip the boat side. If its on 
                #the right side, then we just add those sheep and wolves in the boat
                #to the left side sheeps and wolves and also flip boat side 
                if current_boat_side == 0:
                    new_sheep_left = current_sheep_left - sheep_to_move
                    new_wolves_left = current_wolves_left - wolves_to_move
                    new_boat_side = 1
                else:
                    new_sheep_left = current_sheep_left + sheep_to_move
                    new_wolves_left = current_wolves_left + wolves_to_move
                    new_boat_side = 0

                #new resulting state: new sheep + wolves on left side, new boat side
                new_state = (new_sheep_left, new_wolves_left, new_boat_side)

                if new_state in visited_states:
                    continue
                
                #IMPORTANT: check validity of state. we need to check if wolves outnumber sheep, etc. If so
                #we have a violation and thus do not consider this branch of BFS and just continue to next plausible
                #state
                if not self.is_valid_state(new_sheep_left, new_wolves_left, initial_sheep, initial_wolves):
                    continue
                
                #if valid, we append the state + the move, and add to visited.
                #eventually, we explore many different paths from many different states + valid moves from there
                #branching out like a tree and pruning out all invalid points. we only need to find one 
                #to the goal state so BFS is the best appraoch here
                visited_states.add(new_state)
                new_moves_so_far = moves_so_far + [move]
                queue.append((new_state, new_moves_so_far))

        return []
    

    def get_possible_moves(self, sheep_available, wolves_available):
        # The boat can carry 1 or 2 animals total, made up of sheep and/or wolves.
        possible_moves = []


        #number of possible sheep on boat: 0, 1, 2
        for sheep_to_move in range(0, 3):
            #number of possible wolves on boat: 0, 1, 2
            for wolves_to_move in range(0, 3):
                total_animals_in_boat = sheep_to_move + wolves_to_move
                
                #cant have less than 1, so reject this. Start of loop for example,
                # 0 sheep 0 wolves = 0 animals = reject
                if total_animals_in_boat < 1:
                    continue
                #cant have more than 2 animals. End of loop for example 2 sheep, 2 wolves
                # = 4 animals - reject
                if total_animals_in_boat > 2:
                    continue
                #if we have more sheep in boat than availaible, reject. same iwth wolves
                if sheep_to_move > sheep_available:
                    continue
                if wolves_to_move > wolves_available:
                    continue
                
                possible_moves.append((sheep_to_move, wolves_to_move))

        return possible_moves
    
    #helper: check if current number of sheep + wolves on each side is a valid combo.
    #can't have less than 0 anyimals on either side, cant have more wolves than sheep on 
    #either side. 
    def is_valid_state(self, sheep_left, wolves_left, initial_sheep, initial_wolves):
        sheep_right = initial_sheep - sheep_left
        wolves_right = initial_wolves - wolves_left

        if sheep_left < 0 or wolves_left < 0:
            return False
        if sheep_right < 0 or wolves_right < 0:
            return False

        if sheep_left > 0 and wolves_left > sheep_left:
            return False
        if sheep_right > 0 and wolves_right > sheep_right:
            return False

        return True