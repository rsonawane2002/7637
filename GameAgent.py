from typing import Tuple
from Game import Game, Type
from Token import Token


class GameAgent:
    def __init__(self, token: Token):
        self._token = token

    def token(self):
        return self._token


    #helper methods

    #find all empty cells in board currently
    def get_empty_cells(self, board):
        empty = []
        for i in range(len(board)):
            for j in range(len(board[i])):
                if board[i][j] == '':
                    empty.append((i, j))
        return empty
    
    #find which columns are non empty. so 
    # if the topmost value (row 0) is empty
    # we can place an object
    def get_valid_columns(self, board):
        col = len(board[0])
        valid = []
        for i in range(col):
            if board[0][i] == '':
                valid.append(i)

        return valid



    # check rows/cols/diags for 3 in a row
    def check_winner(self, board):
        for i in range(len(board)):
            if board[i][0] != '' and board[i][0] == board[i][1] == board[i][2]:
                return board[i][0]
        for i in range(len(board)):
            if board[0][i] != '' and board[0][i] == board[1][i] == board[2][i]:
                return board[0][i]
        if board[0][0] != '' and board[0][0] == board[1][1] == board[2][2]:
            return board[0][0]
        if board[0][2] != '' and board[0][2] == board[1][1] == board[2][0]:
            return board[0][2]
        return None

    # return opponent token
    def get_opponent(self):
        if self._token.value() == 'X':
            return Token('O')
        return Token('X')

    # minimax for TTT
    def minimax(self, board, is_maximizing, opp_token):
        winner = self.check_winner(board)
        if winner is not None:
            return 10 if winner == self._token.value() else -10
        empty = self.get_empty_cells(board)
        if len(empty) == 0:
            return 0
        if is_maximizing:
            best = float('-inf')
            for (row, col) in empty:
                board[row][col] = self._token.value()
                score = self.minimax(board, False, opp_token)
                board[row][col] = ''
                best = max(best, score)
            return best
        else:
            best = float('inf')
            for (row, col) in empty:
                board[row][col] = opp_token.value()
                score = self.minimax(board, True, opp_token)
                board[row][col] = ''
                best = min(best, score)
            return best

    def make_move(self, game: Game) -> Tuple[int, int] | int:
        if game.get_type() != Type.TIC_TAC_TOE:
            return -1

        if self._token == game.player1_token():
            opp_token = game.player2_token()
        else:
            opp_token = game.player1_token()

        board = game.get_board()
        best_score, best_move = float('-inf'), None

        for i, j in self.get_empty_cells(board):
            board[i][j] = self._token.value()
            score = self.minimax(board, False, opp_token)
            board[i][j] = ''
            if score > best_score:
                best_score = score
                best_move = (i, j)

        return best_move