import os
import pickle
import h5py
import numpy as np
import re

class LoadedGame:
    def __init__(self, filename):
        self.filename = filename
        self.utility_matrix = None
        self.attacker_utility_matrix = None
        self.defender_utility_matrix = None
        self.schedule_form_dict = None

        if filename.endswith(".game"):
            self._load_from_game_file()
        elif filename.endswith(".pkl"):
            self._load_from_pickle()
        elif filename.endswith(".h5"):
            self._load_from_h5()
        else:
            raise ValueError("Unsupported file format. Must be .game, .pkl, or .h5")

    def _load_from_game_file(self):
        with open(self.filename, "r") as f:
            lines = f.readlines()

        # Parse metadata (ignore comments and blank lines)
        data_lines = [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]

        # Determine matrix size
        indices = []
        values = []
        for line in data_lines:
            parts = re.split(r'\[|\]|:', line)
            index_str = parts[1].strip()
            value_str = parts[2].strip()
            
            i, j = map(int, index_str.split())
            val_list = list(map(float, value_str.strip("[]").split()))
            
            indices.append((i, j))
            values.append(val_list)

        num_rows = max(i for i, _ in indices) + 1
        num_cols = max(j for _, j in indices) + 1

        # Detect whether general-sum or zero-sum
        if len(values[0]) == 2:
            attacker_matrix = np.zeros((num_rows, num_cols))
            defender_matrix = np.zeros((num_rows, num_cols))
            for (i, j), (a_val, d_val) in zip(indices, values):
                attacker_matrix[i, j] = a_val
                defender_matrix[i, j] = d_val
            self.attacker_utility_matrix = attacker_matrix
            self.defender_utility_matrix = defender_matrix
        else:
            matrix = np.zeros((num_rows, num_cols))
            for (i, j), [val] in zip(indices, values):
                matrix[i, j] = val
            self.utility_matrix = matrix

    def _load_from_pickle(self):
        with open(self.filename, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and "schedule_form" in data:
            self.schedule_form_dict = data
        elif isinstance(data, tuple) and len(data) == 2:
            self.attacker_utility_matrix, self.defender_utility_matrix = data
        else:
            self.utility_matrix = data

    def _load_from_h5(self):
        with h5py.File(self.filename, "r") as f:
            if "utility_matrix" in f:
                self.utility_matrix = f["utility_matrix"][:]
            elif "attacker_utility_matrix" in f and "defender_utility_matrix" in f:
                self.attacker_utility_matrix = f["attacker_utility_matrix"][:]
                self.defender_utility_matrix = f["defender_utility_matrix"][:]
            elif "schedule_form" in f:
                # You'll need to define structure for this if you want complex schedule form support.
                self.schedule_form_dict = {
                    k: f[k][:] for k in f.keys()
                }
            else:
                raise ValueError("Unrecognized HDF5 format.")

    def is_zero_sum(self):
        return self.utility_matrix is not None

    def is_general_sum(self):
        return self.attacker_utility_matrix is not None and self.defender_utility_matrix is not None

    def is_schedule_form(self):
        return self.schedule_form_dict is not None