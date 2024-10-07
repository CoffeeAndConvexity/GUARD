import numpy as np

def format_strategies(strategy_matrix):
    """
    Post-processes a strategy matrix to a 2D array where each row is a strategy
    and each column corresponds to a timestep. Supports multiple players, and squeezes
    any extra dimensions.
    
    Input:
    - strategy_matrix: 3D numpy array of shape (num_strategies, num_timesteps, num_players)
    
    Returns:
    - 2D numpy array where each row is a strategy and each column corresponds to a timestep.
      If there's only one player, it removes the extra dimension.
    """
    num_strategies, num_timesteps, num_players = strategy_matrix.shape
    
    # Squeeze out the extra dimension if there's only one attacker
    if num_players == 1:
        formatted_strategies = np.squeeze(strategy_matrix, axis=2)
    else:
        formatted_strategies = []
        for strategy in strategy_matrix:
            formatted_strategy = []
            for timestep in range(num_timesteps):
                attacker_positions = tuple(strategy[timestep][a] for a in range(num_players))
                formatted_strategy.append(attacker_positions)
            formatted_strategies.append(formatted_strategy)
        
        formatted_strategies = np.array(formatted_strategies)
    
    return formatted_strategies