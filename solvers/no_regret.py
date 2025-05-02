import numpy as np
import time

def compute_utility(utilities, strategy, second_player_utilities = None):
    eus = [[0 for _ in range(len(utilities))], [0 for _ in range(len(utilities[0]))]]
    for i in range(len(utilities)):
        for j in range(len(utilities[0])):
            eus[0][i] += utilities[i][j] * strategy[1][j]
            if second_player_utilities == None:
                eus[1][j] -= utilities[i][j] * strategy[0][i]
            else:
                eus[1][j] += second_player_utilities[i][j] * strategy[0][i]
    return eus


def get_averaging_denom(player, iterations_made, averaging):
    if averaging == 0:
        return 1.0 / iterations_made[player]
    elif averaging == 1:
        return 2.0 / ( iterations_made[player] * (iterations_made[player] + 1) )
    elif averaging == 2:
        return 6.0 / ( iterations_made[player] * (iterations_made[player] + 1) * (2 * iterations_made[player] + 1) )
    else:
        return None

def regret_matching(utilities, runtime = None, interval=100, second_player_utilities = None, iterations=10000, averaging = 2, alternations = True, plus = True, predictive = True, verbose=False, precision = 1e-5):
    n_players = 2
    m_actions = [len(utilities), len(utilities[0])]

    regrets   = [[0] * m_actions[i] for i in range(n_players)]
    strategy_sum = [[0] * m_actions[i] for i in range(n_players)]
    strategy  = [[1 / m_actions[i]] * m_actions[i] for i in range(n_players)]

    players = [0] if alternations else [_ for _ in range(n_players)]

    iterations_made = [0 for _ in range(n_players)]

    gaps = []
    
    if runtime:
        start = time.time()

    interval_times = []
    interval_time_start = time.time()
    for iter_idx in range(iterations):
        if iter_idx % interval == 0 and iter_idx > 0:
            interval_times.append(time.time()-interval_time_start)
            # print(time.time()-interval_time_start)
            average_strategy = [[s * get_averaging_denom(p, iterations_made, averaging) for s in player] for p, player in enumerate(strategy_sum)]
            action_utilities = compute_utility(utilities, average_strategy, second_player_utilities = second_player_utilities)
            
            if second_player_utilities == None:
                dual_gap = max(action_utilities[0]) + max(action_utilities[1])
            else:
                expected_utilities = [sum([strategy[p][a] * action_utilities[p][a] for a in range(m_actions[p])]) for p in range(n_players)]
                dual_gap = sum([max(action_utilities[p]) - expected_utilities[p] for p in range(n_players)])
            interval_time_start = time.time()
            if verbose:
                print('Iteration', iter_idx, 'gap: ',dual_gap)
            # for i in range(n_players):
            #     print('\t player', i, 'strategy:', average_strategy[i])

            gaps.append(dual_gap)
            # print(dual_gap)
            
            if dual_gap < precision: break

        action_utilities = compute_utility(utilities, strategy, second_player_utilities = second_player_utilities)
        expected_utilities = [sum([strategy[p][a] * action_utilities[p][a] for a in range(m_actions[p])]) for p in range(n_players)]
      
        for i in players:
            for a in range(m_actions[i]):
                regrets[i][a] = max(0 if plus else float("-inf"), regrets[i][a] + action_utilities[i][a] - expected_utilities[i])
        
        for i in players:
            positive_regrets = [max(0, (regrets[i][a] + action_utilities[i][a] - expected_utilities[i]) if predictive else regrets[i][a]) for a in range(m_actions[i])]
            normalizing_sum = sum(positive_regrets)
            strategy[i] = [r / normalizing_sum if normalizing_sum > 0 else 1 / m_actions[i] for r in positive_regrets]
        
        for i in players:
            iterations_made[i] += 1
            for a in range(m_actions[i]):
                strategy_sum[i][a] += pow(iterations_made[i], averaging) * strategy[i][a]

        players = [ (iter_idx + 1) % n_players ] if alternations else [_ for _ in range(n_players)]

        if runtime:
            checkpoint = time.time()
            if checkpoint - start >= runtime:
                break
    
    average_strategy = [[s * get_averaging_denom(p, iterations_made, averaging) for s in player] for p, player in enumerate(strategy_sum)]
    action_utilities = compute_utility(utilities, average_strategy, second_player_utilities = second_player_utilities)
    if verbose: print()
    return average_strategy, action_utilities, gaps, interval_times
