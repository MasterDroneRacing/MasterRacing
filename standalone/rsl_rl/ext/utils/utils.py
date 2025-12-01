import torch
import csv
import atexit
import git 
import os
import pathlib
import torch
class InfoLogger:
    _instance = None

    def __new__(
        cls, 
        file_name="info_log.csv", 
        data_type=["throttle", "cmd_wx", "cmd_wy", "cmd_wz", "roll", "pitch", "yaw", "wx", "wy", "wz", "ax", "ay", "az"]
        ):
        if cls._instance is None:
            cls._instance = super(InfoLogger, cls).__new__(cls)
            cls._instance._initialize(file_name, data_type)
        return cls._instance
    
    def _initialize(self, file_name, data_type:list[str]):
        self.file_name = file_name
        self.file = open(self.file_name, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        self.data_type = data_type
        # write header
        self.file.seek(0)
        if self.file.tell() == 0:
            self.writer.writerow(data_type)
        
        atexit.register(self._shutdown)

    def log_frame(self, frame_data:dict[str, float]):
        try:
            # convert to numpy
            frame_data = [frame_data.get(key, 'unknown') for key in self.data_type]

            # write to file
            self.writer.writerow(frame_data)
            self.file.flush()
        except Exception as e:
            print(f"[ERROR]: {e}")
            print("[INFO]: Failed to log frame data.")

    def _shutdown(self):
        if not self.file.closed:
            self.file.close()
            print(f"[INFO]: {self.file_name} closed.")


import numpy as np
import torch
import random
import os
import warp as wp
def set_seed(seed, deterministic=True):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
        # will cause training slow down[only for reproduciblity]
        if deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            torch.use_deterministic_algorithms(True)
    wp.rand_init(seed)





def split_and_pad_trajectories(tensor, dones):
    """Splits trajectories at done indices. Then concatenates them and pads with zeros up to the length og the longest trajectory.
    Returns masks corresponding to valid parts of the trajectories
    Example:
        Input: [ [a1, a2, a3, a4 | a5, a6],
                 [b1, b2 | b3, b4, b5 | b6]
                ]

        Output:[ [a1, a2, a3, a4], | [  [True, True, True, True],
                 [a5, a6, 0, 0],   |    [True, True, False, False],
                 [b1, b2, 0, 0],   |    [True, True, False, False],
                 [b3, b4, b5, 0],  |    [True, True, True, False],
                 [b6, 0, 0, 0]     |    [True, False, False, False],
                ]                  | ]

    Assumes that the inputy has the following dimension order: [time, number of envs, additional dimensions]
    """
    dones = dones.clone()
    dones[-1] = 1
    # Permute the buffers to have order (num_envs, num_transitions_per_env, ...), for correct reshaping
    flat_dones = dones.transpose(1, 0).reshape(-1, 1)

    # Get length of trajectory by counting the number of successive not done elements
    done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
    trajectory_lengths = done_indices[1:] - done_indices[:-1]
    trajectory_lengths_list = trajectory_lengths.tolist()
    # Extract the individual trajectories
    trajectories = torch.split(tensor.transpose(1, 0).flatten(0, 1), trajectory_lengths_list)
    # add at least one full length trajectory
    trajectories = trajectories + (torch.zeros(tensor.shape[0], tensor.shape[-1], device=tensor.device),)
    # pad the trajectories to the length of the longest trajectory
    padded_trajectories = torch.nn.utils.rnn.pad_sequence(trajectories)
    # remove the added tensor
    padded_trajectories = padded_trajectories[:, :-1]

    trajectory_masks = trajectory_lengths > torch.arange(0, tensor.shape[0], device=tensor.device).unsqueeze(1)
    return padded_trajectories, trajectory_masks


def unpad_trajectories(trajectories, masks):
    """Does the inverse operation of  split_and_pad_trajectories()"""
    # Need to transpose before and after the masking to have proper reshaping
    return (
        trajectories.transpose(1, 0)[masks.transpose(1, 0)]
        .view(-1, trajectories.shape[0], trajectories.shape[-1])
        .transpose(1, 0)
    )


def store_code_state(logdir, repositories) -> list:
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []
    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            # skip if not a git repository
            continue
        # get the name of the repository
        repo_name = pathlib.Path(repo.working_dir).name
        t = repo.head.commit.tree
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        # check if the diff file already exists
        if os.path.isfile(diff_file_name):
            continue
        # write the diff file
        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        with open(diff_file_name, "x", encoding="utf-8") as f:
            content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            f.write(content)
        # add the file path to the list of files to be uploaded
        file_paths.append(diff_file_name)
    return file_paths
