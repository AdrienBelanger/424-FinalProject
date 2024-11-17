##### TEMP CODE UNTIL WE FINISH STUDENT AGENT #####

"""
Helpers.py is a collection of functions that primarily make up the Reversi/Othello game logic.
Beyond a few things in the World init, which can be copy/pasted this should be almost
all of what you'll need to simulate games in your search method.

Functions:
    get_directions    - a simple helper to deal with the geometry of Reversi moves - get the direction vectors [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)] returned
    count_capture     - how many flips does this move make. Game logic defines valid moves as those with >0 returns from this function.    returns  int The number of stones that will be captured making this move, including all directions. Zero indicates any form of invalid move.
    count_capture_dir - Check if placing a disc at move_pos captures any discs in the specified direction. Returns int Number of stones captured in this direction (unlikely to be used, used internally in above)
    execute_move      - update the chess_board by simulating a move Play the move specified by altering the chess_board. Note that chess_board is a pass-by-reference in/output parameter. Consider copy.deepcopy() of the chess_board if you want to consider numerous possibilities.
    flip_disks        - a helper for the above, unlikely to be used externally
    check_endgame     - check for termination, who's won but also helpful to score non-terminated games     Check if the game ends and compute the final score.  Note that the game may end when a) the board is full or  b) when it's not full yet but both players are unable to make a valid move. One reason for b) occurring is when one player has no stones left. In human play this is sometimes scored as the max possible win (e.g. 64-0), but  we do not implement this scoring here and simply count the stones. Returns as a tuple of 3:  is_endgame : bool Whether the game ends. player_1_score : int  The score of player 1. player_2_score : int  The score of player 2.
    get_valid_moves   - use this to get the children in your tree     Get all valid moves given the chess board and player. Returns valid_moves : [(tuple)] random_move       - basis of the random agent and can be used to simulate play
   
    
     For all, the chess_board is an np array of integers, size nxn and integer values indicating square occupancies.
    The current player is (1: Blue, 2: Brown), 0's in the board mean empty squares.
    Move pos is a tuple holding [row,col], zero indexed such that valid entries are [0,board_size-1]
"""




# Student agent: Add your own agent here DONT FORGET TO REMOVE THE GAME. bcs its only for our folder structure
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves




@register_agent("three_step_agent")
class three_step_Agent(Agent): # student agent best agent

  def __init__(self):
    super(three_step_Agent, self).__init__()
    self.name = "three_step_Agent"
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
    time_taken = time.time() - start_time

    # give ourselves a buffer of 0.1 seconds to return.
    time_limit_to_think = 1.9



    move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think)







    print("My AI's turn took ", time_taken, "seconds.")


    if (move == None) :
      print('smt went wrong and min_max didnt return :( giving random move...')
      move = random_move(chess_board,player)

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

  def min_max_give_me_ur_best_move(self, chess_board, player, start_time, time_limit):

    ops = 3 - player 
    
    valid_moves = get_valid_moves(chess_board, player)

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

            eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit)

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

      except TimeoutError:
        break # if we dont have time left, then we return the best we have for now
    return best_move

  def min_max_score(self, chess_board, depth , alpha, beta, max_or_nah, player, ops, start_time, time_limit):

    if time.time() - start_time > time_limit:
      raise TimeoutError
    
    (is_end, p1_score, p2_score) = check_endgame(chess_board)

    # if its the end of the game then return the score of the move
    if (is_end) :
      if (player == 1):
        return p1_score - p2_score
      else :
        return p2_score - p1_score
    
    # if were at depth 0 then we end recursion
    if depth == 0:
      return self.heuristic_score(chess_board, player, ops)

    valid_moves = get_valid_moves(chess_board, player)

    if (len(valid_moves) == 0):
      # if we cant move were switching sides
      not_max_or_nah = not max_or_nah
      return self.min_max_score(chess_board, depth, alpha, beta, not_max_or_nah, player, ops, start_time, time_limit)
    
    # we need to know who were simulating as (max or not as this step)
    current_player = player if max_or_nah else ops


    valid_moves = get_valid_moves(chess_board, current_player)

    # if were maximizing, check the max step
    if max_or_nah:
        max_eval = -float('inf')
        for move in valid_moves:
            # Check if time is short and return if we dont have time anymore
            if time.time() - start_time > time_limit:
                raise TimeoutError
            sim_board = deepcopy(chess_board)
            execute_move(sim_board, move, current_player)

            # reccurence, check for next layer and minimize
            eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit)

            # maximise the score
            max_eval = max(max_eval, eval_score)

            # update alpha
            alpha = max(alpha, eval_score)

            # prune
            if beta <= alpha:
                break
        return max_eval
    
    # else were minimising.. same thing just
    else:
        min_eval = float('inf')
        for move in valid_moves:
            # Check if time is short and ....
            if time.time() - start_time > time_limit:
                raise TimeoutError

            simulated_board = deepcopy(chess_board)
            execute_move(simulated_board, move, current_player)

            eval_score = self.min_max_score(simulated_board, depth - 1, alpha, beta, True, player, ops, start_time, time_limit)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval



def heuristic_score(self, chess_board, player, ops):
    # count the number of brown and blue, simple greedy way to evaluate, could maybe use helper, but oh well, works.. find better heuristic?
    player_score = np.sum(chess_board == player)
    opponent_score = np.sum(chess_board == ops)
    return player_score - opponent_score