from tetris import Tetris, TETROMINOS, window
from typing import List, Tuple, Dict
import itertools
from collections import deque

class DQNEnv(Tetris):
    def __init__(self, hidden_rows=2):
        super().__init__(hidden_rows)
        self.height_at_10 = 0
        self.height_at_20 = 0

    def reset(self):
        super().reset()
        return self._get_board_props(self.board)
    
    def _number_of_holes(self, board):
        """Number of holes in the board (empty square with at least one block above it)"""
        holes = 0

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            holes += len([x for x in tail if x == Tetris.MAP_EMPTY])

        return holes

    def _bumpiness(self, board):
        """Sum of the differences of heights between pair of columns"""
        total_bumpiness = 0
        max_bumpiness = 0
        min_ys = []

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            n = Tetris.BOARD_HEIGHT - len([x for x in tail])
            min_ys.append(n)

        for (y0, y1) in window(min_ys):
            bumpiness = abs(y0 - y1)
            max_bumpiness = max(bumpiness, max_bumpiness)
            total_bumpiness += bumpiness

        return total_bumpiness, max_bumpiness

    def _height(self, board):
        """Sum and maximum height of the board"""
        sum_height = 0
        max_height = 0
        min_height = Tetris.BOARD_HEIGHT

        for col in zip(*board):
            tail = itertools.dropwhile(lambda x: x != Tetris.MAP_BLOCK, col)
            height = len([x for x in tail])

            sum_height += height
            max_height = max(height, max_height)
            min_height = min(height, min_height)

        return sum_height, max_height, min_height

    def _get_board_props(self, board, last_move = "Default", future_props = None) -> List[int]:
        """Get properties of the board"""
        lines, board = self._clear_lines(board, last_move = last_move)
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        return [lines, holes, total_bumpiness, sum_height]

    def get_next_states(self) -> Dict[Tuple[int, int], List[int]]:
        """Get all possible next states"""
        states = {}
        piece_id = self.current_piece

        if piece_id == 6:
            rotations = [0]
        elif piece_id == 0:
            rotations = [0, 90]
        else:
            rotations = [0, 90, 180, 270]

        # For all rotations
        for rotation in rotations:
            piece = TETROMINOS[piece_id][rotation]
            min_x = min([p[0] for p in piece])
            max_x = max([p[0] for p in piece])

            # For all positions
            for x in range(-min_x, Tetris.BOARD_WIDTH - max_x):
                pos = [x, 0]

                # Drop piece
                while self.is_valid_position(piece, pos):
                    pos[1] += 1
                pos[1] -= 1

                # Valid move
                if pos[1] >= 0:
                    board = self._add_piece_to_board(piece, pos)
                    states[(tuple(pos), rotation)] = (self._get_board_props(board), self.get_moves(pos, rotation))

        return states

    def get_state_size(self):
        """Size of the state"""
        return 4
    
    def get_moves(self, position, rotation):
        """Get the move sequence to reach a position and rotation from the starting position"""
        moves = []
        start_x = self.current_pos[0]
        start_rot = self.current_rotation

        # Rotations
        rot_diff = (rotation - start_rot) % 360
        if rot_diff == 90:
            moves.append('e')  # clockwise
        elif rot_diff == 180:
            moves.append('e')
            moves.append('e')
        elif rot_diff == 270:
            moves.append('q')  # counter-clockwise

        # Horizontal moves
        x_diff = position[0] - start_x
        if x_diff < 0:
            moves.extend(['a'] * abs(x_diff))  # left
        elif x_diff > 0:
            moves.extend(['d'] * x_diff)  # right
        moves.append(' ')  # hard drop

        return moves

    def play_moves(self, action, render = False, mode = None, move_seq = None):
        """
        :param self: Description
        :param action: position, rotations
        :param mode: 
            fast = teleports piece in place-- assumes position is valid, good for training
            realistic = plays the sequence of moves
        """
        return self.hard_drop([action[0][0], 0], action[1], render = render)
    
class intEnv(DQNEnv):
    MAX_DEPTH = 20
    def __init__(self):
        super().__init__()
    
    def _get_future_props(self):
        return None

    def get_next_states(self) -> Dict[Tuple[Tuple[int, int], int], Tuple]:
        """
        Get all possible next states using BFS.
        """
        future_props = self._get_future_props()
        states = {}
        visited = set()
        piece_id = self.current_piece
        start_pos = tuple(self.current_pos)
        start_rot = self.current_rotation
        
        

        # default
        initial_pos = list(self.current_pos)
        piece = TETROMINOS[self.current_piece][self.current_rotation]
        while self.is_valid_position(piece, [initial_pos[0], initial_pos[1]+1]):
            initial_pos[1] += 1
        board, dead = self._simulate_placement(piece, initial_pos)
        initial_state = self._get_board_props(board, last_move = self._get_last_move_type([' ']), future_props=future_props)

        states[(tuple(initial_pos), self.current_rotation)] = (initial_state, [" "])
        # Queue: (pos, rotation, move_sequence, depth)
        queue = deque([(start_pos, start_rot, [], 0)])
        
        while queue:
            pos, rot, move_seq, depth = queue.popleft()
            
            # Depth limit to prevent infinite exploration
            if depth > self.MAX_DEPTH:
                continue
            
            # Check if we've visited this state
            state_key = (pos, rot)
            if state_key in visited:
                continue
            visited.add(state_key)
            
            piece = TETROMINOS[piece_id][rot]
            
            # Check if piece can move down
            down_pos = (pos[0], pos[1] + 1)
            can_go_down = self.is_valid_position(piece, down_pos)
            
            if not can_go_down:
                # This is a landing position
                if self.is_valid_position(piece, pos):
                    board, dead = self._simulate_placement(piece, pos)
                    
                    if board and not dead:  # Valid placement and not dead
                        # Store this placement
                        # Key: (position, rotation)
                        # Value: (board_state, is_dead, move_sequence_with_hard_drop)
                        final_seq = self._add_hard_drop(move_seq)
                        placement_key = (pos,rot)
                        
                        # Only keep shortest path to each placement
                        if placement_key not in states or len(final_seq) < len(states[placement_key][1]):
                            state = self._get_board_props(board, last_move = self._get_last_move_type(move_seq), future_props=future_props)
                            states[(pos, rot)] = (state, final_seq)
            
            # === Explore neighboring states ===
            #print("starting to explore neighbors, queue_size", len(queue), "depth", depth)
            
            # Move left
            left_pos = (pos[0] - 1, pos[1])
            if self.is_valid_position(piece, left_pos):
                queue.append((left_pos, rot, move_seq + ['a'], depth + 1))
                #print("Added left move")
            
            # Move right
            right_pos = (pos[0] + 1, pos[1])
            if self.is_valid_position(piece, right_pos):
                queue.append((right_pos, rot, move_seq + ['d'], depth + 1))
                #print("Added right move")
            
            # Move down (soft drop)
            if can_go_down:
                queue.append((down_pos, rot, move_seq + ['s'], depth))
                #print("Added down move")
            
            # Rotations (skip for O-piece)
            if piece_id != 6:
                # Rotate clockwise
                new_rot_cw = (rot + 90) % 360
                self._try_rotation(queue, piece_id, pos, rot, new_rot_cw, 
                                move_seq + ['e'], depth + 1)
                #print("Added clockwise rotation")
                
                # Rotate counter-clockwise
                new_rot_ccw = (rot - 90) % 360
                self._try_rotation(queue, piece_id, pos, rot, new_rot_ccw, 
                                move_seq + ['q'], depth + 1)
                #print("Added counter-clockwise rotation")
        
        return states


    def _try_rotation(self, queue, piece_id, pos, from_rot, to_rot, move_seq, depth):
        """Try rotation with wall kicks"""
        new_piece = TETROMINOS[piece_id][to_rot]
        
        # Get appropriate kick table
        kick_table = self.KICKS.get(piece_id, self.KICKS['default'])
        kicks = kick_table.get((from_rot, to_rot), [(0, 0)])
        
        # Try each kick offset in order
        for kick_x, kick_y in kicks:
            # Apply kick offset (adjust sign based on your coordinate system)
            kicked_pos = (pos[0] + kick_x, pos[1] + kick_y)
            
            if self.is_valid_position(new_piece, kicked_pos):
                queue.append((kicked_pos, to_rot, move_seq, depth + 1))
                break  # Only use first successful kick
    
    
    def _simulate_placement(self, piece, pos):
        """Simulate placing piece and return result"""
        
        # Place piece on board
        board = self._add_piece_to_board(piece, pos)
        
        # Clear lines
        #sim.board, lines_cleared = sim.clear_lines_on_board(sim.board)
        
        # Check game over
        is_dead = any(1 in board[i] for i in range(self.hidden_rows))
        
        # Calculate damage
        #t_spin = sim.check_t_spin(sim.board, position, piece_name, last_move)
        #lines_sent = sim.calculate_lines_sent(lines_cleared, t_spin, game.back_to_back)
        
        board_tuple = tuple(tuple(row) for row in board)
        return (board_tuple, is_dead)
    
    def _get_last_move_type(self, move_seq):
        """Determine if last move was rotation (for T-spin detection)"""
        if not move_seq:
            return ""
        if move_seq[-1] in ['q', 'e']:
            return "rotate"
        return "other"
    
    def _add_hard_drop(self, move_seq):
        """Add hard drop command, removing trailing soft drops"""
        seq = move_seq[:]
        while seq and seq[-1] == 's':
            seq.pop()
        seq.append(' ')  # Space = hard drop
        return seq

    def get_state_size(self):
        """Size of the state"""
        return 4
    
    def play_moves(self, action, render = False, mode = "fast", move_seq = []):
        """
        :param self: Description
        :param action: position, rotations
        :param mode: 
            fast = teleports piece in place-- assumes position is valid, good for training
            realistic = plays the sequence of moves
        """
        if action is None:
            return self.hard_drop(self.current_pos, self.current_rotation, render = render)
        if mode == "fast":
            self.current_pos = list(action[0])
            self.current_rotation = action[1]
            if move_seq:
                move_seq.pop()
                if move_seq:
                    if move_seq[-1] in ("q", "e"):
                        self.last_move = "rotation"
                    elif move_seq[-1] in ("a, d"):
                        self.last_move = "move"
            return self.hard_drop(self.current_pos, action[1], render = render)
        elif mode == "realistic":
            reward = 0
            done = False
            for k in move_seq:
                if k == "a":  # a (left)
                    self.move([-1, 0], 0)
                elif k == "d":  # d (right)
                    self.move([+1, 0], 0)
                elif k == "s":  # s (down)
                    self.move([0, +1], 0)
                elif k == "q":  # q (counter-clockwise rotation)
                    self.move([0, 0], -90)
                elif k == "e":  # e (clockwise rotation)
                    self.move([0, 0], 90)
                elif k == " ":  # space
                    reward, done = self.hard_drop(self.current_pos, self.current_rotation, render=render)
                
                    self.render(wait_key=True)
                    return (reward, done)
                self.render(wait_key=True)
            reward, done = self.hard_drop(self.current_pos, self.current_rotation, render=render)
            return (reward, done)
        
class AdvEnv(intEnv):
    def __init__(self):
        super().__init__()

    def reset(self):
        super().reset()
        return self._get_board_props(self.board)
    
    def _get_future_props(self, hold_used = False):
        future_garbage = self.garbage_queue[0] if self.garbage_queue else 0
        future_garbage_total =self.garbage_queue[1:] if self.garbage_queue and len(self.garbage_queue) >=2 else 0
        held_piece = self._get_hold_piece()
        queue = self._get_queue()
        current_piece = self._get_current_piece()
        if not hold_used:
            future_piece = queue[0]
            future_hold = held_piece
            future_queue = queue[1:] + [-1]
        elif held_piece != -1:
            future_piece = queue[0]
            future_hold = current_piece
            future_queue = queue[1:] + [-1]
        else:
            future_piece = queue[1]
            future_hold = current_piece
            future_queue = queue[2:] + [-1, -1]
        return (future_piece, future_hold, future_queue, future_garbage, future_garbage_total)

    def _get_current_piece(self):
        return self.current_piece
    
    def _get_hold_piece(self):
        hold = self.held_piece if self.held_piece is not None else -1
        return hold
    
    def _get_queue(self):
        return self.queue
    
    def _get_garbage(self):
        garbage = self.garbage_queue[0] if self.garbage_queue else 0
        garbage_total = sum(self.garbage_queue)
        return garbage, garbage_total

    def _get_board_props(self, board, last_move = "Default", future_props = None):
        """Get properties of the board as tuple (board_2d, scalar_features) for LargeModel."""
        lines, damage, board = self._clear_lines(board, last_move = last_move, mode = "both")
        holes = self._number_of_holes(board)
        total_bumpiness, max_bumpiness = self._bumpiness(board)
        sum_height, max_height, min_height = self._height(board)
        if not future_props:
            piece = self._get_current_piece()
            hold = self._get_hold_piece()
            queue = self._get_queue() 
            garbage_impending, garbage_total = self._get_garbage()
        else:
            piece, hold, queue, garbage_impending, garbage_total = future_props
        
        # Return as tuple: (board_2d, scalar_features)
        board_2d = board[-20:]
        # Build scalar features: piece, hold, queue (5 items), damage, lines, holes, bumpiness, height, garbage_impending = 14 total
        scalar_features = [piece, hold] + queue + [damage, lines, holes, total_bumpiness, sum_height, garbage_impending, garbage_total]
        return (board_2d, scalar_features)

    def get_state_size(self):
        """Size of the state"""
        return (10, 20), 13
    
    

    

if __name__ == "__main__":
    from time import time
    game = AdvEnv()
    current_time = time()
    print(game.get_next_states().keys())
    print("Time taken:", time() - current_time, "seconds")