# Student agent: Add your own agent here
from shutil import move
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
  
  PRUNECOUNT = 0

  def greedy_score(self, chess_board, player, ops):
      # count the number of brown and blue, simple greedy way to evaluate, could maybe use helper, but oh well, works.. find better heuristic?
      player_score = np.sum(chess_board == player)
      opponent_score = np.sum(chess_board == ops)

      chess_board_tuple = np.shape(chess_board)

      chess_board_area = chess_board_tuple[0] * chess_board_tuple[1]

      return player_score - opponent_score / chess_board_area


# score function from ./gpt_greedy_corners.py
  def corner_score(self, chess_board, player, ops):


      # Corner positions are highly valuable
      corners = [(0, 0), (0, chess_board.shape[1] - 1), 
                (chess_board.shape[0] - 1, 0), 
                (chess_board.shape[0] - 1, chess_board.shape[1] - 1)]
      corner_score = sum(1 for corner in corners if chess_board[corner] == player) * 10
      corner_penalty = sum(1 for corner in corners if chess_board[corner] == ops) * -10

      # Combine scores
      total_score = corner_score + corner_penalty
      return total_score

  def mobility_score (self, chess_board, player, ops):
        opponent_moves = len(get_valid_moves(chess_board, ops))
        mobility_score = -opponent_moves
        return mobility_score

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


    return stability_score
  
  def flip_score(self, chess_board, player, ops, move):
     corner_score = self.corner_score(chess_board, player, ops)
     return count_capture(chess_board, move, player) + 1000 * corner_score

  def ultimate_heuristic_min_max(self, chess_board, player, ops):
    stability_score = self.stability_score(chess_board, player, ops)
    mobility_score = self.mobility_score(chess_board, player, ops)
    corner_score = self.corner_score(chess_board, player, ops)
    greedy_score = self.greedy_score(chess_board, player, ops) / (np.shape(chess_board)[0] * np.shape(chess_board)[1])
    
    # Number of empty spots to evaluate game phase
    num_empty_spots = np.sum(chess_board == 0)
    
    if 20 < num_empty_spots:
      s = 30
      m = 20
      c = 1000
      g = 80 
      e = 0   
    elif 10 < num_empty_spots: # Just greedy and mobility at the end
      s = 0
      m = 10
      c = 1000
      g = 80
      e = 0

    elif 5 < num_empty_spots: # Just greedy and mobility at the end
      s = 0
      m = 0
      c = 1000
      g = 100
      e = 0

    else:
      s = 8 
      m = 12  
      c = 15  
      g = 80  
      e = 5  


    e_score = self.edge_stability_score(chess_board, player, ops)

    total = (s * stability_score +m * mobility_score + c * corner_score + g * greedy_score +
            e * e_score)
    
    #print(f"Stability: {s * stability_score}, Mobility: {m * mobility_score}, "f"Corners: {c * corner_score}, Greedy: {g * greedy_score}, "f"Edge Stability: {e * e_score}, "f"Total: {total}")
    return total


  def print_ultimate_move_score_min_max(self, chess_board, player, ops, ):
    stability_score = self.stability_score(chess_board, player, ops)
    mobility_score = self.mobility_score(chess_board, player, ops)
    corner_score = self.corner_score(chess_board, player, ops)
    greedy_score = self.greedy_score(chess_board, player, ops)
    
    # Number of empty spots to evaluate game phase
    num_empty_spots = np.sum(chess_board == 0)
    
    if 20 < num_empty_spots:
      s = 2
      m = 20
      c = 0 
      g = 10 
      e = 0   
    elif 10 < num_empty_spots: # Just greedy and mobility at the end
      s = 0
      m = 20
      c = 0
      g = 10
      e = 0

    else:
      s = 2 
      m = 12  
      c = 15  
      g = 8  
      e = 5  


    e_score = self.edge_stability_score(chess_board, player, ops)

    total = (s * stability_score +m * mobility_score + c * corner_score + g * greedy_score +
            e * e_score)
    
    print(f"Stability: {s * stability_score}, Mobility: {m * mobility_score}, "f"Corners: {c * corner_score}, Greedy: {g * greedy_score}, "f"Edge Stability: {e * e_score}, "f"Total: {total}")
    


  def ultimate_heuristic_MCTS(self,chess_board,player,ops):
    stability_score = self.stability_score(chess_board, player, ops)
    mobility_score = self.mobility_score(chess_board, player, ops)
    corner_score = self.corner_score(chess_board, player, ops)
    greedy_score = self.greedy_score(chess_board, player, ops) / (np.shape(chess_board)[0] * np.shape(chess_board)[1])

    s = 0 
    m = 0  
    c = 0
    g = 1

    total = (s * stability_score +m * mobility_score + c * corner_score + g * greedy_score)
    
    #print(f"Stability: {s * stability_score}, Mobility: {m * mobility_score}, "f"Corners: {c * corner_score}, Greedy: {g * greedy_score}, "f"Edge Stability: {e * e_score}, "f"Total: {total}")
    return total

  def print_ultimate_move_score_MCTS(self, chess_board, player, ops, move):
     corner_score = self.corner_score(chess_board, player, ops)
     print( count_capture(chess_board, move, player) + 1000 * corner_score)
    

  def edge_stability_score(self, chess_board, player, ops):
    edges = []
    size = chess_board.shape[0]
    for i in range(size):
        edges.extend([(0, i), (size-1, i), (i, 0), (i, size-1)]) 

    edge_score = 0
    for edge in edges:
        if chess_board[edge] == player:
            edge_score += 2
        elif chess_board[edge] == ops:
            edge_score -= 2
    return edge_score

  def locked_in_score(chess_board, player, ops):
    return 0

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

    self.PRUNECOUNT = 0
   
    # give ourselves a buffer to return.

    # Start with mcts, then min_max when less empty spots
    second_step = 30
    empty_spots = np.sum(chess_board == 0)
    print(f"empty_spots: {empty_spots}")
    
    # what agout a locked-in heuristic, that gets zones that cant be stolen back?



    ### Might wanna go with len(possible_moves), instead of the number of plays remaining --- nvm doesnt work great
    #print(f"Valid moves amount: {len(get_valid_moves(chess_board, player))}")
    if (empty_spots < second_step):
        time_limit_to_think = 1.995
        print('Using min max')
        
        move = self.min_max_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think, self.ultimate_heuristic_min_max)

        self.print_ultimate_move_score_min_max(chess_board, player, opponent)
        
        
    else: # First step MCTS
        time_limit_to_think = 1.95 # MCTS needs more time to return
        print('Using MCTS')
        move = self.mcts_give_me_ur_best_move(chess_board, player, time.time(), time_limit_to_think, num_sim_per_node=10, num_empty_spots=empty_spots, treshold=second_step, score_function=self.flip_score)
        self.print_ultimate_move_score_MCTS(chess_board, player, opponent, move)





    if (move == None) :
      print('smt went wrong and we didn\'t find a move :( giving random move...')
      move = random_move(chess_board,player)
    print(f"PRUNE COUNT: {self.PRUNECOUNT}")
    time_taken = time.time() - start_time
    print("My AI's turn took ", time_taken, "seconds.")
    
    # Dummy return (you should replace this with your actual logic)
    # Returning a random valid move as an example
    return move

  def min_max_give_me_ur_best_move(self, chess_board, player, start_time, time_limit, score_function):
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
                self.PRUNECOUNT += 1
                break
            
          # (outside the for loop) augment depth for IDS
          depth += 1
          max_depth +=1
          
          
          
          


      except TimeoutError:
        print(f"Agent got to depth {max_depth}")
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
          max_eval = 0
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
                  self.PRUNECOUNT +=1
                  break
          return max_eval

      # Minimizing step
      else:
          min_eval = 0
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
                  self.PRUNECOUNT +=1
                  break
          return min_eval

  def mcts_give_me_ur_best_move(self, chess_board, player, start_time, time_limit, num_sim_per_node, treshold, num_empty_spots, score_function):
    opponent = 3 - player

    def get_base_move_scores(chess_board,player,moves, score_function):
      scores = np.zeros(len(moves))
      for i,move in enumerate(moves):
        # executes move
        CB = chess_board
        scores[i] = score_function(CB, player, 3-player, move)
        
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

    def simulate_to_end(chess_board,p,q,num_sim):
      total_sim_score = 0
      # simulates to end using random moves for both agents
      for sim in range(num_sim):
        chess_board_sim = np.copy(chess_board)
        is_endgame, p_score, opp_score = check_endgame(chess_board_sim,player,opponent)
        while not is_endgame:
          # picks random move
          valid_moves = get_valid_moves(chess_board_sim,p) #### ADRIEN: Added choice with valid move since it returned empty
          valid_moves_ops = get_valid_moves(chess_board_sim,q)
          if len(valid_moves) == 0 and len(valid_moves_ops) ==0:
            break
          # print(valid_moves) ADRIEN:  For debugging
          if len(valid_moves) == 0:
              # print(f"No valid moves left for player {player}.") ADRIEN: Spammed at this point
              p,q = swap_players(p,q)
          move = random_move(chess_board_sim,p)
          execute_move(chess_board_sim,move,p)
          p,q = swap_players(p,q)
          # checks for endgame
          is_endgame, p_score, opp_score = check_endgame(chess_board,player,opponent)
        total_sim_score += p_score - opp_score 
      return total_sim_score / num_sim     
    
    def compute_move_scores(exploit,explore,n_prev,k,d):
      
      total = d * exploit + k * np.sqrt(np.log(n_prev) / explore)
      #print(f"Compute Move Scores: {total}")
      return total

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()
    
    # based on num empty spots >> 3 diff phases
    
    TOTAL = 10
    D = 10
    K = 1
    
    
    NUM_SIM_PER_NODE = int(num_sim_per_node * (treshold / num_empty_spots))
    if NUM_SIM_PER_NODE >= 1: NUM_SIM_PER_NODE 
    else: NUM_SIM_PER_NODE = 1
    print(f"Using {NUM_SIM_PER_NODE} simulations per node")
    # Step 1: gets list of possible moves
    POSSIBLE_MOVES = get_valid_moves(chess_board,player)
    num_sim = 0

    # Step 2: searches tree
    tree_states = [chess_board] # saved as (board_state,parent_ind) pairs ## Adrien: Use np copy so we dont accidentally play the move
    node_moves = [POSSIBLE_MOVES]
    exploit = [get_base_move_scores(chess_board,player,POSSIBLE_MOVES, score_function)]
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
      
      while time.time() - start_time < time_limit:
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
        cur_scores = compute_move_scores(exploit[ind],explore[ind],prev_node_score,K, D)
        #print(f"curscores: {cur_scores}, EXPLORE: {explore[ind]}, EXPLOIT: {exploit[ind]}")
        #if cur_scores.size == 0: break
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
        exploit.append(get_base_move_scores(s,p,new_moves, score_function))
        explore.append(np.ones(len(new_moves)))
        node_scores.append(1)
        parent_nodes.append(node_inds[-1] if node_inds else -1) ### Add to parent node, right now we arent

      # runs simulation
      sim_score =  simulate_to_end(s,p,q, NUM_SIM_PER_NODE)
      
      # backpropagate results
      print(f"max_depth: {depth}")
      for d in range(depth): 
        ind = node_inds[depth-1-d]              # retrieves prev node explored in path
        node_scores[ind] += 1                   # updates node score by 1
        move_ind = move_inds[depth-1-d]         # retrieves move index in list
        explore[ind][move_ind] += NUM_SIM_PER_NODE             # explore score increases by 1
        exploit[ind][move_ind] += sim_score     # exploits score increases by win magnitude

      num_sim += 1 # Nice
    
    print(f"Agent ran {num_sim} simulations.")
    best_move = np.argmax(exploit[0]) # final decision? only exploit or also explore? 
    print(f"EXPLORE SCORES: {explore[0]}")
    print(f"END Move Scores: {compute_move_scores(explore[0], exploit[0], 1, K, D)} with K = {K} and D = {D}")
    print(f"EXPLOIT SCORES: {exploit[0]}")
    return POSSIBLE_MOVES[best_move]
