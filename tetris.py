import random

import cv2
import numpy as np
import itertools
from PIL import Image
import torch

KICKS = {
    0: {  # I-piece
        (0, 90): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (90, 180): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
        (180, 270): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (270, 0): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        # Counter-clockwise (reverse)
        (90, 0): [(0, 0), (2, 0), (-1, 0), (2, 1), (-1, -2)],
        (180, 90): [(0, 0), (1, 0), (-2, 0), (1, -2), (-2, 1)],
        (270, 180): [(0, 0), (-2, 0), (1, 0), (-2, -1), (1, 2)],
        (0, 270): [(0, 0), (-1, 0), (2, 0), (-1, 2), (2, -1)],
    },
    'default': {  # JLSTZ pieces
        (0, 90): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (90, 180): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
        (180, 270): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (270, 0): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        # Counter-clockwise (reverse)
        (90, 0): [(0, 0), (1, 0), (1, 1), (0, -2), (1, -2)],
        (180, 90): [(0, 0), (-1, 0), (-1, -1), (0, 2), (-1, 2)],
        (270, 180): [(0, 0), (-1, 0), (-1, 1), (0, -2), (-1, -2)],
        (0, 270): [(0, 0), (1, 0), (1, -1), (0, 2), (1, 2)],
    }
}

TETROMINOS = {
    0: {  # I - rotates in a 4x4 grid around center (1.5, 1.5)
        0: [(0, 1), (1, 1), (2, 1), (3, 1)],
        90: [(2, 0), (2, 1), (2, 2), (2, 3)],
        180: [(3, 2), (2, 2), (1, 2), (0, 2)],
        270: [(1, 3), (1, 2), (1, 1), (1, 0)],
    },
    1: {  # T - rotates around (1, 1)
        0: [(1, 0), (0, 1), (1, 1), (2, 1)],
        90: [(2, 1), (1, 0), (1, 1), (1, 2)],
        180: [(1, 2), (2, 1), (1, 1), (0, 1)],
        270: [(0, 1), (1, 2), (1, 1), (1, 0)],
    },
    2: {  # L - rotates around (1, 1)
        0: [(2, 0), (0, 1), (1, 1), (2, 1)],
        90: [(2, 2), (1, 0), (1, 1), (1, 2)],
        180: [(0, 2), (2, 1), (1, 1), (0, 1)],
        270: [(0, 0), (1, 2), (1, 1), (1, 0)],
    },
    3: {  # J - rotates around (1, 1)
        0: [(0, 0), (0, 1), (1, 1), (2, 1)],
        90: [(2, 0), (1, 0), (1, 1), (1, 2)],
        180: [(2, 2), (2, 1), (1, 1), (0, 1)],
        270: [(0, 2), (1, 2), (1, 1), (1, 0)],
    },
    4: {  # Z - rotates around (1, 1)
        0: [(0, 0), (1, 0), (1, 1), (2, 1)],
        90: [(2, 0), (2, 1), (1, 1), (1, 2)],
        180: [(2, 2), (1, 2), (1, 1), (0, 1)],
        270: [(0, 2), (0, 1), (1, 1), (1, 0)],
    },
    5: {  # S - rotates around (1, 1)
        0: [(1, 0), (2, 0), (0, 1), (1, 1)],
        90: [(2, 1), (2, 2), (1, 0), (1, 1)],
        180: [(1, 2), (0, 2), (2, 1), (1, 1)],
        270: [(0, 1), (0, 0), (1, 2), (1, 1)],
    },
    6: {  # O - doesn't rotate
        0: [(1, 0), (2, 0), (1, 1), (2, 1)],
        90: [(1, 0), (2, 0), (1, 1), (2, 1)],
        180: [(1, 0), (2, 0), (1, 1), (2, 1)],
        270: [(1, 0), (2, 0), (1, 1), (2, 1)],
    }
}

PIECE_MASK = [
    # I
    [
        [0,0,0,0],
        [1,1,1,1],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # T
    [
        [0,1,0,0],
        [1,1,1,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # L
    [
        [0,0,1,0],
        [1,1,1,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # J
    [
        [1,0,0,0],
        [1,1,1,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # Z
    [
        [1,1,0,0],
        [0,1,1,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # S
    [
        [0,1,1,0],
        [1,1,0,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
    # O
    [
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
        [0,0,0,0],
    ],
]

TSPIN_CHECK = {
    0: ((0, 0), (0, 2), (2, 2), (2, 0)),
    90: ((0, 2), (2, 2), (0, 2), (0, 0)),
    180: ((2, 2), (2, 0), (0, 0), (0, 2)),
    270: ((2, 0), (0, 0), (0, 2), (2, 2),)
}

# Tetris game class
# noinspection PyMethodMayBeStatic
class Tetris:
    """Tetris game class"""

    # BOARD
    MAP_EMPTY = 0
    MAP_BLOCK = 1
    MAP_PLAYER = 2
    BOARD_WIDTH = 10
    BOARD_HEIGHT = 20
    QUEUE_SIZE = 5
    KICKS = KICKS
    

    COLORS = {
        0: (255, 255, 255),
        1: (247, 64, 99),
        2: (0, 167, 247),
    }

    def __init__(self, hidden_rows=2):
        # to avoid warnings just mention the warnings
        self.game_over = False
        self.current_pos = [3, 0]
        self.current_rotation = 0
        self.board = []
        self.bag = []
        self.queue = []
        self.score = 0
        self.round_score = 0
        Tetris.BOARD_HEIGHT += hidden_rows
        self.hidden_rows = hidden_rows
        self.back_to_back = 0

        self.reset()
    

    def reset(self):
        """Resets the game, returning the current state"""
        self.game_step = 0
        self.board = [[0] * Tetris.BOARD_WIDTH for _ in range(Tetris.BOARD_HEIGHT)]
        self.game_over = False
        self.queue = []
        self.bag = list(range(len(TETROMINOS)))
        random.shuffle(self.bag)
        for _ in range(Tetris.QUEUE_SIZE):
            self.queue.append(self.bag.pop(0))
        self._new_round(piece_fall=False)
        self.score = 0
        self.round_score = 0
        self.last_move = None
        self.garbage_queue = []
        self.hold_used = False
        self.held_piece = -1

    def _get_rotated_piece(self, rotation):
        """Returns the current piece, including rotation"""
        return TETROMINOS[self.current_piece][rotation]

    def _get_complete_board(self):
        """Returns the complete board, including the current piece"""
        piece = self._get_rotated_piece(self.current_rotation)
        piece = [np.add(x, self.current_pos) for x in piece]
        board = [x[:] for x in self.board]
        for x, y in piece:
            if 0 <= y < Tetris.BOARD_HEIGHT and 0 <= x < Tetris.BOARD_WIDTH:
                board[y][x] = Tetris.MAP_PLAYER
        return board

    def get_round_score(self):
        """Returns the current game score.

        Each block placed counts as one.
        For lines cleared, it is used BOARD_WIDTH * lines_cleared ^ 2.
        """
        return self.score
    
    def get_round_score(self):
        """Returns the current round score."""
        return self.round_score

    def _new_round(self, piece_fall=False, scoring = True) -> int:
        """Starts a new round (new piece)"""
        score = 0
        if piece_fall:
            # Update board and calculate score
            piece = self._get_rotated_piece(self.current_rotation)
            self.board = self._add_piece_to_board(piece, self.current_pos)
            lines_cleared, self.board = self._clear_lines(self.board)
            score = 1 + (lines_cleared ** 2) * Tetris.BOARD_WIDTH
            if scoring:
                self.score += score
                self.round_score += score

        # Generate new bag with the pieces
        if len(self.bag) == 0:
            self.bag = list(range(len(TETROMINOS)))
            random.shuffle(self.bag)

        self.current_piece = self.queue.pop(0)
        self.queue.append(self.bag.pop(0))
        self.current_pos = [3, 0]
        self.current_rotation = 0

        if not self.is_valid_position(self._get_rotated_piece(self.current_rotation), self.current_pos):
            self.game_over = True
        self.game_step += 1
        return score

    def is_valid_position(self, piece, pos):
        """Check if there is a collision between the current piece and the board.
        :returns: True, if the piece position is _invalid_, False, otherwise
        """
        for x, y in piece:
            x += pos[0]
            y += pos[1]
            if x < 0 or x >= Tetris.BOARD_WIDTH \
                    or y < 0 or y >= Tetris.BOARD_HEIGHT \
                    or self.board[y][x] == Tetris.MAP_BLOCK:
                return False
        return True

    def _rotate(self, angle):
        """Change the current rotation"""
        if self.current_piece == 6:  # O-piece does not rotate
            return True
        
        temp_r = self.current_rotation + angle
        if temp_r == 360:
            temp_r = 0
        if temp_r < 0:
            temp_r += 360
        elif temp_r > 360:
            temp_r -= 360

        rotated_piece = TETROMINOS[self.current_piece][temp_r]

        # Use 'I' for piece 0, otherwise use 'default'
        kick_table = KICKS.get(self.current_piece, KICKS['default'])
        kicks = kick_table.get((self.current_rotation, temp_r), [(0, 0)])

        # Try each kick to find a valid position
        k = 0
        for kick in kicks:
            new_position = [self.current_pos[0] + kick[0], self.current_pos[1] - kick[1]]
            if self.is_valid_position(rotated_piece, new_position):
                self.current_pos = new_position
                self.current_rotation = temp_r
                self.last_move = "rotate"
                return True
            k += 1

        # If no valid position found, rotation fails
        
        return False

    def move_horizontal(self, direction) -> bool:
        new_position = [self.current_pos[0] + direction, self.current_pos[1]]
        if self.is_valid_position(self._get_rotated_piece(self.current_rotation), new_position):
            self.current_pos = new_position
            self.last_move = "move"
            return True
        else:
            return False
    
    def move_down(self, d):
        new_position = [self.current_pos[0], self.current_pos[1] + d]

        if self.is_valid_position(self._get_rotated_piece(self.current_rotation), new_position):
            self.current_pos = new_position
            return True
        else:
            return False


    def _add_piece_to_board(self, piece, pos):
        """Place a piece in the board, returning the resulting board"""
        board = [x[:] for x in self.board]
        for x, y in piece:
            board[y + pos[1]][x + pos[0]] = Tetris.MAP_BLOCK
        return board

    def _clear_lines(self, board, mode = 'line_clear', last_move = "Default"):
        """Clears completed lines in a board"""
        # Check if lines can be cleared
        lines_to_clear = [index for index, row in enumerate(board) if sum(row) == Tetris.BOARD_WIDTH]
        t_spin = "No"
        if lines_to_clear:
            t_spin = self.check_t_spin(board, self.current_pos, self.current_rotation, self.current_piece, last_move)
            """if t_spin == "T Full":
                print("T-SPIN detected!", lines_to_clear)
            elif t_spin == "T Mini":
                print("T-SPIN MINI detected")"""
            board = [row for index, row in enumerate(board) if index not in lines_to_clear]
            # Add new lines at the top
            for _ in lines_to_clear:
                board.insert(0, [0 for _ in range(Tetris.BOARD_WIDTH)])
        if mode == 'line_clear':
            return len(lines_to_clear), board
        elif mode == 'damage':
            return self.calculate_lines_sent(lines_to_clear, t_spin, self.back_to_back), board
        elif mode == 'both':
            return len(lines_to_clear), self.calculate_lines_sent(len(lines_to_clear), t_spin, self.back_to_back), board

    def check_t_spin(self, board, piece_position, piece_rotation, piece_id, last_move):
        """Check if the current move was a T-spin."""
        if last_move == "Default":
            last_move = self.last_move
        if piece_id == 1 and last_move == "rotate":
            occupied = 0
            front_occupied = 0
            for i, coords in enumerate(TSPIN_CHECK[piece_rotation]):
                x = piece_position[0] + coords[0]
                y = piece_position[1] + coords[1]
                if (y >= Tetris.BOARD_HEIGHT or x < 0 or x >= Tetris.BOARD_WIDTH or board[y][x] >= 1):
                    occupied += 1
                    if i < 2:
                        front_occupied +=1
            if occupied >= 3 and front_occupied == 2:
                return("T Full")
            elif occupied >=3:
                return("T Mini")
                
        return "No"
    
    def calculate_lines_sent(self, lines_cleared, t_spin, back_to_back):
        """Calculate how many lines are sent based on T-spin and other conditions."""
        lines_sent = 0
        if lines_cleared > 0:
            if t_spin == "T Full":
                lines_sent = self.calculate_t_spin_lines_sent(lines_cleared, back_to_back)
            else:
                lines_sent = self.calculate_regular_lines_sent(lines_cleared, back_to_back)

        return lines_sent
    
    def calculate_t_spin_lines_sent(self, lines_cleared, back_to_back):
        """Handles the lines sent during a T-spin."""
        if lines_cleared == 1:
            return 2 + back_to_back
        elif lines_cleared == 2:
            return 4 + back_to_back
        elif lines_cleared == 3:
            return 6 + back_to_back
        return 0

    def calculate_regular_lines_sent(self, lines_cleared, back_to_back):
        """Handles regular line clears."""
        if lines_cleared == 1:
            return 0
        elif lines_cleared == 2:
            return 1
        elif lines_cleared == 3:
            return 2
        elif lines_cleared == 4:
            return 4 + back_to_back
        return 0

    def move(self, shift_m, shift_r) -> bool:
        if shift_r != 0:
            if not self._rotate(shift_r):
                return False
        if shift_m[0] != 0:
            if not self.move_horizontal(shift_m[0]):
                return False
        if shift_m[1] != 0:
            moved_down = self.move_down(shift_m[1])
            return moved_down
        return True

    def fall(self) -> bool:
        """:returns: True, if there was a fall move, False otherwise"""
        if not self.move([0, 1], 0):
            # cannot fall further
            # start new round
            self._new_round(piece_fall=True)
            if self.game_over:
                self.score -= 2
                self.round_score -= 2
        return self.game_over

    def hard_drop(self, pos, rotation, render=False):
        """Makes a hard drop given a position and a rotation, returning the reward and if the game is over"""
        self.current_pos = pos
        self.current_rotation = rotation
        # drop piece
        piece = self._get_rotated_piece(self.current_rotation)
        if not self.is_valid_position(piece, self.current_pos):
            print("Invalid position for fast drop:", piece, self.current_pos)
        while self.is_valid_position(piece, self.current_pos):
            if render:
                self.render(wait_key=True)
            self.current_pos[1] += 1
        self.current_pos[1] -= 1
        # start new round
        score = self._new_round(piece_fall=True)
        if self.game_over:
            score -= 2
        if render:
            self.render(wait_key=True)
        return score, self.game_over
    
    def hold_piece(self):
        if not self.hold_used:
            if self.held_piece is None:
                self.held_piece = self.current_piece
                self._new_round(piece_fall=False)
            else:
                temp = self.current_piece
                self.current_piece = self.held_piece
                self.held_piece = temp
                self.current_pos = [3, 0]
                self.current_rotation = 0
                if not self.is_valid_position(self._get_rotated_piece(self.current_rotation), self.current_pos):
                    self.game_over = True
            self.hold_used = True

    def render(self, wait_key=False):
        """Renders the current board"""
        board = self._board_display_array().astype(np.int32)

        palette = np.asarray(
            [Tetris.COLORS[i] for i in range(len(Tetris.COLORS))],
            dtype=np.uint8
        )

        img = palette[board]
        img = img[..., ::-1]  # Convert RRG to BGR (used by cv2)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((Tetris.BOARD_WIDTH * 25, Tetris.BOARD_HEIGHT * 25))
        img = np.array(img)
        cv2.putText(img, str(self.score), (22, 22), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
        cv2.imshow('image', np.array(img))
        if wait_key:
            # this is needed to render during training
            cv2.waitKey(1)

    def _pad_cols(self, a, target, value=0):
        return np.pad(
            a,
            ((0, 0), (0, target - a.shape[1])),
            mode="constant",
            constant_values=value
        )

    def _pad_rows(self, a, target, value=0):
        return np.pad(
            a,
            ((0, target - a.shape[0]), (0, 0)),
            mode="constant",
            constant_values=value
        )

    def _board_display_array(self):
        # Main board
        board = np.array(self._get_complete_board())   # (H, W)

        # Queue (stack pieces vertically)
        queue_pieces = [np.array(PIECE_MASK[p]) for p in self.queue]
        queue = np.vstack(queue_pieces) if queue_pieces else np.zeros((0, 0), dtype=int)

        # --- Horizontal merge: board | queue ---
        max_rows = max(board.shape[0], queue.shape[0])

        board_pad = self._pad_rows(board, max_rows)
        h_zero_pad = np.zeros((max_rows, 1), dtype=int)  # separator column
        queue_pad = self._pad_rows(queue, max_rows)

        h_merged = np.hstack((board_pad, h_zero_pad, queue_pad))

        # Current falling piece (on top)
        piece = np.array(PIECE_MASK[self.current_piece])

        # --- Vertical merge: piece over board+queue ---
        max_cols = max(piece.shape[1], h_merged.shape[1])

        piece_pad = self._pad_cols(piece, max_cols)
        v_zero_pad = np.zeros((1, max_cols), dtype=int)  # separator row
        h_merged_pad = self._pad_cols(h_merged, max_cols)

        final = np.vstack((piece_pad, v_zero_pad,h_merged_pad))

        return final



def window(seq, n=2):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
       NB. taken from https://docs.python.org/release/2.3.5/lib/itertools-example.html
    """
    it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
