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


    #given a column index, find the "lowest"
    # empty row
    def get_next_row(self, board, col):
        row = len(board)
        lowest = None
        for i in range(row):
            if board[i][col] == '':
                if lowest is None:
                    lowest=  i
                else:
                    lowest = max(lowest, i)
        return lowest
    
    #given a board state, check if we have a winner (4 in vertical
    # horizontal, or either diagonal)
    def c4_check_winner(self, board, seq):
        rows = len(board)
        cols = len(board[0])

        # horizontal
        for r in range(rows):
            for c in range(cols - seq + 1):
                cell = board[r][c]
                if cell == '':
                    continue
                match = True
                for i in range(seq):
                    if board[r][c + i] != cell:
                        match = False
                        break
                if match:
                    return cell

        # vertical - fix col outer, row inner
        for c in range(cols):
            for r in range(rows - seq + 1):
                cell = board[r][c]
                if cell == '':
                    continue
                match = True
                for i in range(seq):
                    if board[r + i][c] != cell:
                        match = False
                        break
                if match:
                    return cell

        # diagonal down-right
        for r in range(rows - seq + 1):
            for c in range(cols - seq + 1):
                cell = board[r][c]
                if cell == '':
                    continue
                match = True
                for i in range(seq):
                    if board[r + i][c + i] != cell:
                        match = False
                        break
                if match:
                    return cell

        # diagonal down-left
        for r in range(rows - seq + 1):
            for c in range(seq - 1, cols):
                cell = board[r][c]
                if cell == '':
                    continue
                match = True
                for i in range(seq):
                    if board[r + i][c - i] != cell:
                        match = False
                        break
                if match:
                    return cell

        return None


    #scoring function: essentially we slice a window from the board
    # and we teach agent if we have a lot of our tokens thats good
    #and if the opponent has a lot of their tokens its bad
    def score_window(self, window, seq, opp_val):
        my_val = self._token.value()
        
        my_count = 0
        empty_count = 0
        opp_count = 0

        for cell in window:
            if cell == my_val:
                my_count +=  1
            if cell == '':
                empty_count += 1
            if cell == opp_val:
                opp_count += 1

        if my_count == seq:
            return 100
        if my_count == seq - 1 and empty_count == 1:
            return 10
        if my_count == seq - 2 and empty_count == 2:
            return 2
        if opp_count == seq - 1 and empty_count == 1:
            return -50
        if opp_count == seq:
            return -100

        return 0


    #given a board, evaluate the current score
    #that we have by calling score window helper.
    # we check all possible windows of size seq
    # vertically horizontally and diagnoally.
    def evaluate_board(self, board, seq, opp_val):
        rows = len(board)
        cols = len(board[0])
        score = 0

        # center column preference
        center_col = cols // 2
        for r in range(rows):
            if board[r][center_col] == self._token.value():
                score += 3

        # horizontal windows
        for r in range(rows):
            for c in range(cols - seq + 1):
                window = [board[r][c + i] for i in range(seq)]
                score += self.score_window(window, seq, opp_val)

        # vertical windows
        for c in range(cols):
            for r in range(rows - seq + 1):
                window = [board[r + i][c] for i in range(seq)]
                score += self.score_window(window, seq, opp_val)

        # diagonal down-right
        for r in range(rows - seq + 1):
            for c in range(cols - seq + 1):
                window = [board[r + i][c + i] for i in range(seq)]
                score += self.score_window(window, seq, opp_val)

        # diagonal down-left
        for r in range(rows - seq + 1):
            for c in range(seq - 1, cols):
                window = [board[r + i][c - i] for i in range(seq)]
                score += self.score_window(window, seq, opp_val)

        return score
    

    def minimax_c4(self, board, depth, is_maximizing, opp_token, seq):
        winner = self.c4_check_winner(board, seq)
        if winner == self._token.value():
            return 100000
        if winner == opp_token.value():
            return -100000
        
        #get the valid column indices
        valid_cols = self.get_valid_columns(board)
        if len(valid_cols) == 0:
            return 0  # draw
        
        #return curr state of board if cant search more gamne states
        if depth == 0:
            return self.evaluate_board(board, seq, opp_token.value())


        if is_maximizing:
            best = float('-inf')
            for col in valid_cols:
                row = self.get_next_row(board, col)
                board[row][col] = self._token.value()
                score = self.minimax_c4(board, depth -1, False, opp_token, seq)  
                board[row][col] = ''
                best = max(best, score)
            return best
        else:
            best = float('inf')
            for col in valid_cols:
                row = self.get_next_row(board, col)
                board[row][col] = opp_token.value()
                score = self.minimax_c4(board, depth - 1, True, opp_token, seq)
                board[row][col] = ''
                best = min(best, score)
            return best




    # check rows/cols/diags for 3 in a row
    def tt_check_winner(self, board):
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
        winner = self.tt_check_winner(board)
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
        if self._token == game.player1_token():
            opp_token = game.player2_token()
        else:
            opp_token = game.player1_token()

        if game.get_type() == Type.TIC_TAC_TOE:
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

        else:
            board = game.get_board()
            seq = game.number_of_seq_tokens_needed()
            depth = 4
            best_score, best_col = float('-inf'), None
            for col in self.get_valid_columns(board):
                row = self.get_next_row(board, col)
                board[row][col] = self._token.value()
                score = self.minimax_c4(board, depth - 1, False, opp_token, seq)
                board[row][col] = ''
                if score > best_score:
                    best_score = score
                    best_col = col
            return best_col