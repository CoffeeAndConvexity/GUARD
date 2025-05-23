import os
import numpy as np
import pickle
import h5py
from abc import ABC, abstractmethod

class DomainSpecificSG(ABC):
    @abstractmethod
    def draw_graph(self):
        pass

    @abstractmethod
    def generate(self):
        pass
        
    def to_game(self, filename, general_sum=False):
        if not filename.endswith(".game"):
            raise ValueError("Filename must end with .game")

        num_defender_actions = self.utility_matrix.shape[0] if not general_sum else self.attacker_utility_matrix.shape[0]
        num_attacker_actions = self.utility_matrix.shape[1] if not general_sum else self.attacker_utility_matrix.shape[1]

        header = [
            "# Generated by RealGame v1.0\n",
            "# Realistic Security Game\n",
            "# Game Parameter Values:\n",
            "# Random seed: N/A\n",
            "# Cmd Line: -g RealGame\n",
            f"# Players: 2\n",
            f"# Actions: {num_defender_actions} {num_attacker_actions}\n",
            "#\n"
        ]

        lines = []
        if general_sum:
            matrix1 = self.attacker_utility_matrix
            matrix2 = self.defender_utility_matrix
            for i in range(num_defender_actions):
                for j in range(num_attacker_actions):
                    lines.append(f"[{i} {j}] :\t[{matrix1[i, j]} {matrix2[i, j]}]\n")
        else:
            matrix = self.utility_matrix
            for i in range(num_defender_actions):
                for j in range(num_attacker_actions):
                    lines.append(f"[{i} {j}] :\t[{matrix[i, j]}]\n")

        with open(filename, "w") as f:
            f.writelines(header + lines)

    def to_h5(self, filename, general_sum=False, schedule_form=False):
        if not filename.endswith(".h5"):
            raise ValueError("Filename must end with .h5")

        with h5py.File(filename, "w") as f:
            if schedule_form:
                for k, v in self.schedule_form_dict.items():
                    f.create_dataset(k, data=np.array(v, dtype='S'))  # store values as strings
            elif general_sum:
                f.create_dataset("attacker_utility_matrix", data=self.attacker_utility_matrix)
                f.create_dataset("defender_utility_matrix", data=self.defender_utility_matrix)
            else:
                f.create_dataset("utility_matrix", data=self.utility_matrix)

    def to_pkl(self, filename, general_sum=False, schedule_form=False):
        if not filename.endswith(".pkl"):
            raise ValueError("Filename must end with .pkl")

        if schedule_form:
            data = self.schedule_form_dict
        elif general_sum:
            data = {
                "attacker_utility_matrix": self.attacker_utility_matrix,
                "defender_utility_matrix": self.defender_utility_matrix
            }
        else:
            data = self.utility_matrix

        with open(filename, "wb") as f:
            pickle.dump(data, f)

    def to_nfg(self, filename, general_sum=False):
    if not filename.endswith(".nfg"):
        raise ValueError("Filename must end with .nfg")

    game_title = "ExportedGame"
    players = ["Defender", "Attacker"]

    if general_sum:
        matrix1 = self.attacker_utility_matrix
        matrix2 = self.defender_utility_matrix
    else:
        matrix = self.utility_matrix
        matrix1 = matrix
        matrix2 = -matrix  # Zero-sum: defender gets negative of attacker

    num_def_actions = matrix1.shape[0]
    num_att_actions = matrix1.shape[1]

    # Strategy labels: just use stringified indices
    def_strategies = [f"D{i}" for i in range(num_def_actions)]
    att_strategies = [f"A{j}" for j in range(num_att_actions)]

    # Outcome definitions (avoids duplicates)
    outcome_map = {}
    outcomes = []
    outcome_counter = 1

    outcome_indices = []
    for i in range(num_def_actions):
        for j in range(num_att_actions):
            payoff = (matrix1[i, j], matrix2[i, j])
            if payoff not in outcome_map:
                outcome_map[payoff] = outcome_counter
                outcomes.append((payoff, outcome_counter))
                outcome_counter += 1
            outcome_indices.append(outcome_map[payoff])

    with open(filename, "w") as f:
        # Prologue
        f.write(f'NFG 1 R "{game_title}" {{ "{players[0]}" "{players[1]}" }}\n\n')
        f.write("{\n")
        f.write(f'{{ {" ".join(f"\\"{s}\\"" for s in def_strategies)} }}\n')
        f.write(f'{{ {" ".join(f"\\"{s}\\"" for s in att_strategies)} }}\n')
        f.write("}\n\n")

        # Outcome definitions
        f.write("{\n")
        for (payoff, idx) in outcomes:
            f.write(f'{{ "" {payoff[0]}, {payoff[1]} }}\n')
        f.write("}\n")

        # Outcome indices (body)
        f.write(" ".join(str(idx) for idx in outcome_indices) + "\n")