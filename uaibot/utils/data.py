import os
import numpy as np
import urllib.request
import hashlib


def get_fishbotics_mp_problems() -> dict:
    """
    Gets a set of many motion planning problems from the 'FishBotics' dataset.
    They were designed for the Franka Emika Panda robot.
    See: https://github.com/fishbotics/robometrics


    It is a dictionary in which each key is the name of the problem.
    If 'all_problems' is the output of this function, use
     list(all_problems.keys()) to get the keys (the problem names).

    Each problem is again a dictionary, which contains the following keys:

    'all_obs': a list of UAIBot objects representing the obstacles.
    'q0': the initial configuration.
    'htm_base': the initial pose for the base.
    'htm_tg': the target HTML for the end-effector.


    Parameters:
    ----------
    None.

    Returns
    -------
    all_problems : dictionary
        The dictionary with all the problems.
    """

    def load_npz_from_url_cached(url, cache_dir=".cache"):
        os.makedirs(cache_dir, exist_ok=True)

        # Create a unique filename based on the URL hash
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        local_path = os.path.join(cache_dir, f"{url_hash}.npz")

        # If cached, load from disk
        if os.path.exists(local_path):
            return np.load(local_path, allow_pickle=True)

        # Else, download and save
        with urllib.request.urlopen(url) as response:
            data = response.read()
        with open(local_path, "wb") as f:
            f.write(data)

        return np.load(local_path, allow_pickle=True)

    allproblems_1 = load_npz_from_url_cached(
        "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/MotionPlanningProblems/fishbotics_mp_problems_part_1.npz"
    )
    allproblems_2 = load_npz_from_url_cached(
        "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/MotionPlanningProblems/fishbotics_mp_problems_part_2.npz"
    )
    allproblems_3 = load_npz_from_url_cached(
        "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/MotionPlanningProblems/fishbotics_mp_problems_part_3.npz"
    )
    allproblems_4 = load_npz_from_url_cached(
        "https://cdn.jsdelivr.net/gh/UAIbot/uaibot_data@master/MotionPlanningProblems/fishbotics_mp_problems_part_4.npz"
    )

    allproblems_1 = allproblems_1["arr_0"].item()
    allproblems_2 = allproblems_2["arr_0"].item()
    allproblems_3 = allproblems_3["arr_0"].item()
    allproblems_4 = allproblems_4["arr_0"].item()

    return {**allproblems_1, **allproblems_2, **allproblems_3, **allproblems_4}
