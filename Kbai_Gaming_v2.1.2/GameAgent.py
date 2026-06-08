from typing import Tuple

from Game import Game, Type
from Token import Token


class GameAgent:
    def __init__(self, token: Token):
        """
        Initial call to create your agent.
        This only gets called 1 time and then your agent will play multiple games.
        """
        self._token = token

    def token(self):
        return self._token


    #my functions:

    def get_empty_cells(self, board):
        empty = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] is None:
                    empty.append((i,j))
        return empty

    #check rows/cols/diags for 4 in a row
    def check_winner(self, board):

        for i in range(len(board)):
            if board[i][0] is not None and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
        for i in range(len(board)):
            if board[0][i] is not None and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        
        #diagonals
        if board[0][0] is not None and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] is not None and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]


        return None



    #return opponent token
    def get_opponent(self, board):
        for row in board:
            for cell in row:
                if cell is not None and cell != self._token:
                    return cell
        return None
    
    #im going to use minimax here along with previous helper functins to find a winner
    #assuming optimal play
    def minimax(self, board, is_maximizing):
        winner = self.check_winner(board)
        if winner is not None:
            if winner == self._token:
                return 10
            else:
                return -10

        empty = self.get_empty_cells(board)
        if len(empty) == 0:
            return 0  # draw

        #our turn
        if is_maximizing:
            best = float('-inf')
            for (row, col) in empty:
                board[row][col] = self._token
                score = self.minimax(board, not is_maximizing)
                board[row][col] = None  
                best = max(best, score)
            return best
        
        #opponent
        else:
            best = float('inf')
            for (row, col) in empty:
                board[row][col] = self.get_opponent(board)
                score = self.minimax(board, not is_maximizing)
                board[row][col] = None
                best = min(best, score)
            return best

    def make_move(self, game: Game) -> Tuple[int, int] | int:
        """
        This is the main driver of the agent. The game controller will call this with an updated game object
        every time the agent is expected to make a move.

        The agent will return a Tuple(int, int) for Tic-tac-toe
        or just an int for all Connect Four games.

        Tic-tac-toe:
        Return a tuple in row, column form. (row, col)
        Returning (-1,-1) will make a random selection of available positions.

        Both Row and Column begin at index 0.
        A typical Tic-Tac-Toe board is laid out such that 0,0 is upper left corner.

        Example Board:
        0,0  0,1  0,2
        1,0  1,1  1,2
        2,0  2,1  2,2


        CONNECT FOUR:
        Return an integer representing the column to drop the token into (no row is needed only the column)
        Returning -1 will make a random selection of available positions.

        The Column begins at index 0.
        A typical Connect Four board is laid out such that columns 0 starts on the left side of the play field.
        As new pieces are added, they will start filling columns from the lowest position in the column.

        Example Board
           | 0 | 1 | 2 | 3 | 4 | 5 | 6 |

           |   |   |   |   |   |   |   |
           |   |   |   |   |   |   |   |
           |   |   |   |   |   |   |   |
           |   |   |   |   |   |   |   |
           |   |   |   | Y | Y |   |   |
           | R |   | Y | R | R |   |   |


        Remember each game type requires a different output '(int, int)' or 'int'

        Write your code starting here.
        """
        

        #my algorithm
        # 1) get board and list of empty cells
        # 2) loop over empty cells and for each one make a move + call minimax
        # 3) track score from that move, and at the end find highest scoring move and return


        #for now deal with just tic tac toe
        if game.get_type() != Type.TIC_TAC_TOE:
            return -1

        board = game.get_board()

        empty = self.get_empty_cells(board)


        best_score, best_move = float('-inf'), None

        for i, j in empty:
            board[i][j] = self._token
            #False bc its oppponents turn now
            score = self.minimax(board, False)
            if score > best_score:
                best_score = score
                best_move = (i, j)
            board[i][j] = None
        
        return best_move
