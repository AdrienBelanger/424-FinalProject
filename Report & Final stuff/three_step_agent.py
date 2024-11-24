# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("three_step_agent")
class three_step_Agent(Agent):

  def __init__(self):
    super(three_step_Agent, self).__init__()
    self.name = "three_step_agent"
    self.autoplay = True

  def step(self, chess_board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """

    """
    IDEAS:

      - MTCS: Tree policy: Sim Policy:
      - Simulated Annealing: 

      - Greedy?
      - GPT_greedy_corner works with score and combines multiple methods. 
      
      # IDEA: Implement Minimax, then MCTS, then Endgame special solver (minimax again?)

    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    
    
    # give ourselves a buffer of 0.001 seconds to return.
    time_limit_to_think = 1.98

    # Start with mcts, then min_max when less empty spots
    max_empty_spots = 20
    empty_spots, chess_board_dimensions, spots = count_empty_spots_and_dimensions(chess_board)

    if (empty_spots < max_empty_spots):
      move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think)
    else:
      move = self.mcts_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think)
    




    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    


    if (move == None) :
      print('smt went wrong and we didn\'t find a move :( giving random move...')
      move = random_move(chess_board,player)

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

  def min_max_give_me_ur_best_move(self, chess_board, player, start_time, time_limit, score_function):
    chess_board_dimensions = np.shape(chess_board)
    ops = 3 - player 
    
    valid_moves = get_valid_moves(chess_board, player)
    print("VALID MOVES:", len(valid_moves))
    best_move = None
    best_score = -float('inf')

    alpha = -float('inf')
    beta = float('inf')

     # start at depth 1
    depth = 1
    # We want to do IDS with time search, with a buffer of maybe 0.1 seconds to make sure we return so
    while True : 
      try: 
          for move in valid_moves:
            # If we're out of time then raise an error
            if time.time() - start_time > time_limit: raise TimeoutError
            
            # to try moves we create a copy of the chess_board
            sim_board = deepcopy(chess_board)
            # try the move
            execute_move(sim_board, move, player)

            eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit, score_function)

            if eval_score > best_score:
              best_score = eval_score
              best_move = move

            # update alpha
            alpha = max(alpha, eval_score)

            # Prune
            if beta <= alpha:
                break
            
          # (outside the for loop) augment depth for IDS
          depth += 1
          print("Current depth:", depth)
          if(depth > chess_board_dimensions[0] * chess_board_dimensions[1]): break
          
          


      except TimeoutError:
        break # if we dont have time left, then we return the best we have for now
    return best_move
  def min_max_score(self, chess_board, depth, alpha, beta, max_or_nah, player, ops, start_time, time_limit, score_function):

      if time.time() - start_time > time_limit:
          raise TimeoutError

      # If the game has ended, calculate the score and return it
      (is_end, p1_score, p2_score) = check_endgame(chess_board, player, ops)
      if is_end:
          if player == 1:
              return p1_score - p2_score
          else:
              return p2_score - p1_score

      # If we're at depth 0, end recursion and return heuristic score
      if depth == 0:
          return self.score_function(chess_board, player, ops)

      # Get valid moves for the current player
      valid_moves = get_valid_moves(chess_board, player if max_or_nah else ops)

      # If no valid moves, switch turns but decrement depth
      if len(valid_moves) == 0:
          not_max_or_nah = not max_or_nah
          return self.min_max_score(chess_board, depth - 1, alpha, beta, not_max_or_nah, player, ops, start_time, time_limit, score_function)

      # Determine the player for this step
      current_player = player if max_or_nah else ops

      # Maximizing step
      if max_or_nah:
          max_eval = -float('inf')
          for move in valid_moves:
              # Check if time is short and return if we dont have time anymore
              if time.time() - start_time > time_limit:
                  raise TimeoutError

              sim_board = deepcopy(chess_board)
              execute_move(sim_board, move, current_player)

              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit)

              # Maximize the score
              max_eval = max(max_eval, eval_score)

              # Update alpha
              alpha = max(alpha, eval_score)

              # Prune
              if beta <= alpha:
                  break
          return max_eval

      # Minimizing step
      else:
          min_eval = float('inf')
          for move in valid_moves:
              # Check if time is short and return if we dont have time anymore
              if time.time() - start_time > time_limit:
                  raise TimeoutError

              sim_board = deepcopy(chess_board)
              execute_move(sim_board, move, current_player)

              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, True, player, ops, start_time, time_limit)

              # Minimize the score
              min_eval = min(min_eval, eval_score)

              # Update beta
              beta = min(beta, eval_score)

              # Prune
              if beta <= alpha:
                  break
          return min_eval


# definitely need to improve this heuristic
  def greedy_score(self, chess_board, player, ops):
      # count the number of brown and blue, simple greedy way to evaluate, could maybe use helper, but oh well, works.. find better heuristic?
      player_score = np.sum(chess_board == player)
      opponent_score = np.sum(chess_board == ops)
      return player_score - opponent_score


# score function from ./gpt_greedy_corners.py
  def gpt_score(self, chess_board, player, ops):
      # Player and opponent scores
      player_score = np.sum(chess_board == player)
      opponent_score = np.sum(chess_board == ops)

      # Corner positions are highly valuable
      corners = [(0, 0), (0, chess_board.shape[1] - 1), 
                (chess_board.shape[0] - 1, 0), 
                (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
      corner_score = sum(1 for corner in corners if chess_board[corner] == player) * 10
      corner_penalty = sum(1 for corner in corners if chess_board[corner] == ops) * -10

      # Mobility: the number of moves the opponent can make
      opponent_moves = len(get_valid_moves(chess_board, ops))
      mobility_score = -opponent_moves

      # Combine scores
      total_score = player_score - opponent_score + corner_score + corner_penalty + mobility_score
      return total_score


# Inspired by the corner heuristic, which led to https://kartikkukreja.wordpress.com/2013/03/30/heuristic-function-for-reversiothello/
  def stability_score(self, chess_board, player, ops):
    board_size = chess_board.shape[0]

    def generate_weights(size):
        # make the whole board zeros
        weights = np.zeros((size, size))
        
        # corners are really valuable, give them a weight of 4
        corners = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        for corner in corners:
            weights[corner] = 4
        
        # Values next to corners are really weak
        adjacent = [
            (0, 1), (1, 0), (1, 1),
            (0, size - 2), (1, size - 1), (1, size - 2),
            (size - 2, 0), (size - 1, 1), (size - 2, 1),
            (size - 2, size - 1), (size - 1, size - 2), (size - 2, size - 2)
        ]

        # bad values next to corners
        for adj in adjacent:
            weights[adj] = -3
        
        # assign values on top rows that are not adjacent to corners high values
        for i in range(1, size - 1):
            weights[0, i] = 2  # Top
            weights[size - 1, i] = 2  # Bottom
            weights[i, 0] = 2  # Left
            weights[i, size - 1] = 2  # Right

        # rest is low value
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if weights[i, j] == 0:
                    weights[i, j] = 1 if (i + j) % 2 == 0 else -1
        
        return weights

    weights = generate_weights(board_size)

    # Calculate the stability score
    stability_score = np.sum(weights[chess_board == player]) - np.sum(weights[chess_board == ops])

    # greedy score
    player_score = np.sum(chess_board == player)
    opponent_score = np.sum(chess_board == ops)


    return stability_score + 0.5 * (player_score - opponent_score)


def mcts_give_me_ur_best_move(chess_board, player, start_time, time_limit):
   return None


def count_empty_spots_and_dimensions(chess_board):
    count = 0
    spots = 0
    for i in range(np.shape(chess_board)[0]):
        x +=1
        spots +=1
        for j in range(np.shape(chess_board)[1]):
            y +=1
            spots +=1
            if chess_board[i][j] == 0:
                count += 1
    return (count, (x, y))
      
