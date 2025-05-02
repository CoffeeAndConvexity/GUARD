from solvers.nash import nash
from solvers.mip import mip
from solvers.double_oracle import double_oracle
from solvers.double_oracle_sf import double_oracle_sf
from solvers.simple_sse_lp import solve_sse_lp
from solvers.nfg_sse_lp import solve_general_sum_normal_form
from solvers.no_regret import regret_matching

class Solver:
    def __init__(self, method: str, **kwargs):
        self.method = method.lower()
        self.params = kwargs
        self.supported_methods = {
            "nash": self._solve_nash,
            "mip": self._solve_mip,
            "double_oracle": self._solve_double_oracle,
            "double_oracle_sf": self._solve_double_oracle_sf,
            "simple_sse_lp": self._solve_simple_sse_lp,
            "nfg_sse_lp": self._solve_nfg_sse_lp,
            "regret_matching": self._solve_regret_matching
        }
    
    def solve(self):
        if self.method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {self.method}")
        return self.supported_methods[self.method]()

    # --- Solver Methods (1 per file in /solvers) ---

    def _solve_nash(self):
        from solvers.nash import nash
        return nash(self.params["utility_matrix"])

    def _solve_mip(self):
        from solvers.mip import mip
        return mip(self.params["utility_matrix"], self.params["support_bound"])

    def _solve_double_oracle(self):
        from solvers.double_oracle import double_oracle
        return double_oracle(
            schedule_form_di=self.params["schedule_form_di"],
            initial_subgame_size=self.params.get("initial_subgame_size", 2),
            eps=self.params.get("eps", 1e-6),
            verbose=self.params.get("verbose", True)
        )

    def _solve_double_oracle_sf(self):
        from solvers.double_oracle_sf import double_oracle_sf
        return double_oracle_sf(
            game=self.params["game"],
            tau=self.params["tau"],
            eps=self.params.get("eps", 1e-6),
            initial_subgame_size=self.params.get("initial_subgame_size", 2),
            verbose=self.params.get("verbose", True)
        )

    def _solve_simple_sse_lp(self):
        from solvers.sse_lp import solve_sse_lp
        return solve_sse_lp(
            targets=self.params["targets"],
            resources=self.params["resources"],
            A_r=self.params["A_r"],
            u_d_covered=self.params["u_d_covered"],
            u_d_uncovered=self.params["u_d_uncovered"],
            u_a_covered=self.params["u_a_covered"],
            u_a_uncovered=self.params["u_a_uncovered"]
        )

    def _solve_nfg_sse_lp(self):
        from solvers.nfg_sse_lp import solve_general_sum_normal_form
        return solve_general_sum_normal_form(
            defender_matrix=self.params["defender_matrix"],
            attacker_matrix=self.params["attacker_matrix"]
        )

    def _solve_regret_matching(self):
        from solvers.regret_matching import regret_matching
        return regret_matching(
            utilities=self.params["utilities"],
            second_player_utilities=self.params.get("second_player_utilities"),
            iterations=self.params.get("iterations", 10000),
            averaging=self.params.get("averaging", 2),
            alternations=self.params.get("alternations", True),
            plus=self.params.get("plus", True),
            predictive=self.params.get("predictive", True),
            verbose=self.params.get("verbose", False),
            precision=self.params.get("precision", 1e-2)
        )