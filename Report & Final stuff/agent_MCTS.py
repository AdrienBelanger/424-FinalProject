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
    chess_board = deepcopy(chess_board)

   
    # give ourselves a buffer to return.
    

    # Start with mcts, then min_max when less empty spots
    third_step = 30
    second_step = 60
    empty_spots = np.sum(chess_board == 0)
    # print(f"empty_spots: {empty_spots}") :: Debugging
    
    if (empty_spots < third_step):
        time_limit_to_think = 1.995
        print('Using greedy min max')
        move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think, self.greedy_score)
    elif (empty_spots < second_step):
        time_limit_to_think = 1.995
        print('Using stability min max')
        move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think, self.stability_score)
    else: # First step
        time_limit_to_think = 1.97
        print('Using MCTS')
        move = self.mcts_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think)





    if (move == None) :
      print('smt went wrong and we didn\'t find a move :( giving random move...')
      move = random_move(chess_board,player)
    
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    
    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

  def min_max_give_me_ur_best_move(self, chess_board, player, start_time, time_limit, score_function):
    chess_board_dimensions = np.shape(chess_board)
    ops = 3 - player 
    
    valid_moves = get_valid_moves(chess_board, player)
    #print("VALID MOVES:", len(valid_moves))
    best_move = None
    best_score = -float('inf')

    alpha = -float('inf')
    beta = float('inf')

     # start at depth 1
    depth = 1
    max_depth = 1
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
          max_depth +=1
          #print("Current depth:", depth)
          if(depth > chess_board_dimensions[0] * chess_board_dimensions[1]): break
          
          


      except TimeoutError:
        print(max_depth)
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
          return score_function(chess_board, player, ops)

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

              sim_board = chess_board
              execute_move(sim_board, move, current_player)

              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, False, player, ops, start_time, time_limit, score_function)

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

              sim_board = chess_board
              execute_move(sim_board, move, current_player)

              eval_score = self.min_max_score(sim_board, depth - 1, alpha, beta, True, player, ops, start_time, time_limit, score_function)

              # Minimize the score
              min_eval = min(min_eval, eval_score)

              # Update beta
              beta = min(beta, eval_score)

              # Prune
              if beta <= alpha:
                  break
          return min_eval

  def mcts_give_me_ur_best_move(self, chess_board, player, start_time, time_limit):
    opponent = 3 - player

    def get_base_move_scores(chess_board,player,moves):

      scores = np.zeros(len(moves))
      
      for i in range(len(moves)):
        # executes move
        CB = chess_board
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
        if np.array_equal(state_tree[i], state):
          return i
      return -1

    def simulate_to_end(chess_board,p,q):
      # simulates to end using random moves for both agents
      is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
      while not is_endgame:
        # picks random move
        valid_moves = get_valid_moves(chess_board,p) #### ADRIEN: Added choice with valid move since it returned empty
        # print(valid_moves) ADRIEN:  For debugging
        if len(valid_moves) == 0:
            # print(f"No valid moves left for player {player}.") ADRIEN: Spammed at this point
            break
        move = valid_moves[np.random.randint(len(valid_moves))]
        
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
    tree_states = [chess_board] # saved as (board_state,parent_ind) pairs ## Adrien: Use np copy so we dont accidentally play the move
    node_moves = [POSSIBLE_MOVES]
    exploit = [get_base_move_scores(chess_board,player,POSSIBLE_MOVES)]
    explore = [np.ones(len(POSSIBLE_MOVES))] # num explorations
    node_scores = [1] # num node visits
    parent_nodes = [-1] # Make sure we have a way to track the parent, so back prop

    while time.time() - start_time < time_limit:
      s = np.copy(chess_board) ### ADRIEN: Update the chessboard and copy it
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

        if not node_moves[ind]:  # Make sure node_moves is not empyy
            break  
        
        # stores backpointers as lists
        depth += 1
        node_inds.append(ind)
        
        # plays best fit move
        cur_scores = compute_move_scores(exploit[ind],explore[ind],prev_node_score,K)
        if cur_scores.size == 0: break
        move_ind = np.argmax(cur_scores)
        move_inds.append(move_ind)
        if not move_inds:  # Skip backpropagation if no moves were executed
            continue

        execute_move(s,node_moves[ind][move_ind],p)
        
        # swaps player every node
        p,q = swap_players(p,q)
        prev_node_score = node_scores[ind]

      # if found node not in tree > add it
      new_moves = get_valid_moves(s,p)
      
      # if no moves left >> doesn't add to struct
      if new_moves is not None:
        tree_states.append(np.copy(s)) #### ADRIEN: Make sure we work on a copy of the state, so when we come back it isnt already
        node_moves.append(new_moves)
        exploit.append(get_base_move_scores(s,p,new_moves))
        explore.append(np.ones(len(new_moves)))
        node_scores.append(1)
        parent_nodes.append(node_inds[-1] if node_inds else -1) ### Add to parent node, right now we arent

      # runs simulation
      sim_score =  simulate_to_end(s,p,q) ### ADRIEN: Simulate starting at the current state, not at the Chessboard
      
      # backpropagate results
      for d in range(depth): 
        ind = node_inds[depth-1-d]              # retrieves prev node explored in path
        node_scores[ind] += 1                   # updates node score by 1
        move_ind = move_inds[depth-1-d]         # retrieves move index in list
        explore[ind][move_ind] += 1             # explore score increases by 1
        exploit[ind][move_ind] += sim_score     # exploits score increases by win magnitude

      num_sim += 1 # Nice
    
    print(f"Agent ran {num_sim} simulations.")
    best_move = np.argmax(exploit[0]) # final decision? only exploit or also explore?
    return POSSIBLE_MOVES[best_move]
