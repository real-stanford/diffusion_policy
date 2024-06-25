import sys
import os
# hack to import adept envs
ADEPT_DIR = os.path.join(os.path.dirname(__file__), 'relay_policy_learning', 'adept_envs')
sys.path.append(ADEPT_DIR)

import logging
import numpy as np
import adept_envs
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
logger = logging.getLogger()


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = (
        True  # This allows for the tasks to be completed in arbitrary order.
    )

    def __init__(
        self, dataset_url=None, ref_max_score=None, ref_min_score=None, 
        use_abs_action=False,
        **kwargs
    ):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.goal_masking = True
        super(KitchenBase, self).__init__(use_abs_action=use_abs_action, **kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.0
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(
            task=self.TASK_ELEMENTS, actually_return_goal=True
        )  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            complete = distance < BONUS_THRESH
            condition = (
                complete and all_completed_so_far
                if not self.COMPLETE_IN_ANY_ORDER
                else complete
            )
            if condition:  # element == self.tasks_to_complete[0]:
                print("Task {} completed!".format(element))
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break
        env_info["completed_tasks"] = set(self.TASK_ELEMENTS) - set(
            self.tasks_to_complete
        )
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                dict(
                    states=data["observations"][start : end_idx + 1],
                    actions=data["actions"][start : end_idx + 1],
                )
            )
            start = end_idx + 1
        return seqs
