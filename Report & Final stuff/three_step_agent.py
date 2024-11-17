##### TEMP CODE UNTIL WE FINISH STUDENT AGENT #####




# IDEA: Implement Minimax, then MCTS, then Endgame special solver

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
from Game.agents.agent import Agent
from Game.store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from Game.helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

def min_max(chessboard, player,):

  return random_move(chessboard, player,)


@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

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
      

    """

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return random_move(chess_board,player)
