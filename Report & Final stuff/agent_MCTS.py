##### TEMP CODE UNTIL WE FINISH STUDENT AGENT #####

# Attempt 1)

# Student agent: Add your own agent here DONT FORGET TO REMOVE THE GAME. bcs its only for our folder structure

from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

    # add parameters that agent learns here
    self.state_tree = []

    # reuse info across various searches

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
    def get_move_scores(chess_board,player):

      moves = get_valid_moves(chess_board,player)
      scores = np.zeros(len(moves))
      
      for i in range(len(moves)):
        # executes move
        CB = np.copy(chess_board)
        execute_move(CB,moves[i],player)
        fin, player_score, opp_score = check_endgame(CB,player,opponent)
        # evaluates move initial score
        scores[i] = player_score - opp_score

    def swap_players(p,opp):
      if p == player:
        p = opponent
        opp = player
      else:
        p = player
        opp = opponent
      return p, opp
        
      return scores

    def get_loc_in_tree(state_tree,state):

      for i in range(len(state_tree)):
        if state_tree[i] == state:
          return i
      return -1

    def simulate_to_end(chess_board,p,q):
      # simulates to end using random moves for both agents
      is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
      while not is_endgame:
        # picks random move
        move = random_move(chess_board,p)
        execute_move(chess_board,move,player)
        p,q = swap_players(p,q)
        # checks for endgame
        is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
      return (p_score-opp_score) > 0
        
        

      

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    time_taken = time.time() - start_time

    print("My AI's turn took ", time_taken, "seconds.")

    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example

    # Step 1: gets list of possible moves
    moves = get_valid_moves(chess_board,player)
    N = len(moves)
    move_counts = np.zeros(N) # tracks num move uses
    scores = np.zeros(N) # tracks move priority (higher num is better)
    num_sim = 0

    # Step 2: searches tree
    tree_states = [chess_board] # saved as (board_state,parent_ind) pairs
    parent_nodes = [None]
    node_moves = [moves]
    node_score = [0]
    node_visits = [0]

    while time.time() - start_time < 1.9:
      s = np.copy(chess_board)
      p = player
      q = opponent
      depth = 1
      
      while True:
        # checks if node has been visited
        ind = get_loc_in_tree(tree_states,s)
        if ind == -1:
          break
        # plays best fit move
        next_move = np.argmax(node_moves[ind])
        execute_move(s,next_move,p)
        # swaps player every node
        p,q = swap_players(p,q)
        # updates search depth
        depth += 1

      # if found node not in tree > add it
      tree_states.append(s)
      parent_nodes.append(ind)
      node_moves.append(get_move_scores(s,p))
      # runs simulation
      victorious =  simulate_to_end(chess_board,p,q)
      # backpropagate results
      ind = len(tree_states) - 1
      for d in range(depth):
        node_visits[ind] += 1
        if victorious: node_score += 1
        ind = parent_nodes[ind]

      # update move scores too !!!
      
    


    
    return random_move(chess_board,player)
