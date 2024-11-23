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
    def get_base_move_scores(chess_board,player,moves):

      scores = np.zeros(len(moves))
      
      for i in range(len(moves)):
        # executes move
        CB = np.copy(chess_board)
        execute_move(CB,moves[i],player)
        _, player_score, opp_score = check_endgame(CB,player,opponent)
        # evaluates move initial score
        scores[i] = player_score - opp_score
        
      return scores
        
    def count_empty_spaces(chess_board):
      return np.sum(np.equal(chess_board,0))

    def swap_players(p,opp):
      if p == player:
        p = opponent
        opp = player
      else:
        p = player
        opp = opponent
      return p, opp
        
    def get_loc_in_tree(state_tree,state):
      N = len(state_tree)
      for i in range(N-1,-1,-1):
        if state_tree[i] == state:
          return i
      return -1

    def simulate_to_end(chess_board,p,q):
      # simulates to end using random moves for both agents
      is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
      while not is_endgame:
        # picks random move
        move = random_move(chess_board,p)
        execute_move(chess_board,move,p)
        p,q = swap_players(p,q)
        # checks for endgame
        is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
      return p_score-opp_score     
    
    def compute_move_scores(exploit,explore,n_prev,k):
      return exploit + k*np.sqrt(np.log(n_prev)/explore)

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    
    # Step 0: choose appropriate running mode
    num_empty = count_empty_spaces(chess_board)
    
    # based on num empty spots >> 3 diff phases
    
    K = 1 # hyperparam for UCT 

    # Step 1: gets list of possible moves
    POSSIBLE_MOVES = get_valid_moves(chess_board,player)
    num_sim = 0

    # Step 2: searches tree
    tree_states = [chess_board] # saved as (board_state,parent_ind) pairs
    node_moves = [POSSIBLE_MOVES]
    exploit = [get_base_move_scores(chess_board,player,POSSIBLE_MOVES)]
    explore = [np.ones(len(POSSIBLE_MOVES))] # num explorations
    node_scores = [1] # num node visits

    while time.time() - start_time < 1.9:
      
      s = np.copy(chess_board)
      p = player
      q = opponent
      depth = 0
      prev_node_score = 1
      
      node_inds = [] # backpointer to all prev explored nodes
      move_inds = [] # backpoitner to all executed moves
      
      while True:
        # checks if node has been visited
        ind = get_loc_in_tree(tree_states,s)
        if ind == -1:
          break
        
        # stores backpointers as lists
        depth += 1
        node_inds.append(ind)
        
        # plays best fit move
        cur_scores = compute_move_scores(exploit[ind],explore[ind],prev_node_score,K)
        move_ind = np.argmax(cur_scores)
        move_inds.append(move_ind)
        execute_move(s,node_moves[ind][move_ind],p)
        
        # swaps player every node
        p,q = swap_players(p,q)
        prev_node_score = node_scores[ind]

      # if found node not in tree > add it
      new_moves = get_valid_moves(s,p)
      
      # if no moves left >> doesn't add to struct
      if new_moves is not None:
        tree_states.append(s)
        node_moves.append(new_moves)
        exploit.append(get_base_move_scores(s,p,new_moves))
        explore.append(np.ones(len(new_moves)))
        node_scores.append(1)
      
      # runs simulation
      sim_score =  simulate_to_end(chess_board,p,q)
      
      # backpropagate results
      for d in range(depth): 
        ind = node_inds[depth-1-d]              # retrieves prev node explored in path
        node_scores[ind] += 1                   # updates node score by 1
        move_ind = move_inds[depth-1-d]         # retrieves move index in list
        explore[ind][move_ind] += 1             # explore score increases by 1
        exploit[ind][move_ind] += sim_score     # exploits score increases by win magnitude

      num_sim += 1
    
    print(f"Agent ran {num_sim} simulations.")
    best_move = np.argmax(exploit[0]) # final decision? only exploit or also explore?
    return POSSIBLE_MOVES[best_move]
