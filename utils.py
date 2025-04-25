import os
import random
import shutil
import time

import numpy as np
import torch
import subprocess 


def plot_tensorboard(writer, path, obj, idx, labels=None):
    if isinstance(obj, dict):
        writer.add_scalars(path, obj, idx)
    elif type(obj) in np.ScalarType:
        writer.add_scalar(path, obj, idx)
    elif isinstance(obj, np.ndarray) or torch.is_tensor(obj):
        assert obj.ndim == 1
        if labels is None:
            n_item = len(obj)
            labels = [str(i + 1) for i in range(n_item)]
        dic = {labels[i]: item for i, item in enumerate(obj)}
        writer.add_scalars(path, dic, idx)
    else:
        raise NotImplemented("Type {} plotting is not implemented!".format(type(obj)))


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def count_element(arr):
    keys = torch.unique(arr)
    cnt = []
    for key in keys:
        cnt.append((arr == key).int().sum())
    return cnt


def count_element_vector(arr):  # hard coded
    q = arr[:, 0] * 15 + arr[:, 1] * 5 + arr[:, 2]
    return count_element(q)


def get_datasets_from_tensor_with_cnt(data, label, cnt, cuda=False):
    train_data, train_label, test_data, test_label = [], [], [], []
    st = 0
    for num in cnt:
        test_num = int(num * 0.2)

        train_data.append(data[st + test_num : st + num])
        train_label.append(label[st + test_num : st + num])
        test_data.append(data[st : st + test_num])
        test_label.append(label[st : st + test_num])

        st += num

    train_data = torch.cat(train_data, dim=0)
    train_label = torch.cat(train_label, dim=0)
    test_data = torch.cat(test_data, dim=0)
    test_label = torch.cat(test_label, dim=0)

    if cuda:
        train_data = train_data.cuda()
        train_label = train_label.cuda()
        test_data = test_data.cuda()
        test_label = test_label.cuda()

    train_set = torch.utils.data.TensorDataset(train_data, train_label)
    test_set = torch.utils.data.TensorDataset(test_data, test_label)

    return train_set, test_set


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Feb. 11, 2023 version
def run_commands(
    gpus, commands, suffix, call=False, shuffle=True, delay=0.5, ext_command=""
):
    # Create command directory
    command_dir = os.path.join("commands", suffix)
    if len(commands) == 0:
        return
        
    # Clear existing directory
    if os.path.exists(command_dir):
        shutil.rmtree(command_dir)
    os.makedirs(command_dir, exist_ok=True)

    # Create stop script
    stop_path = os.path.join("commands", f"stop_{suffix}.sh")
    with open(stop_path, "w") as fout:
        fout.write(f"kill $(ps aux | grep 'bash {command_dir}' | awk '{{print $2}}')\n")

    # Shuffle if needed
    if shuffle:
        random.shuffle(commands)
        random.shuffle(gpus)

    # Generate and execute command files
    n_gpu = len(gpus)
    for i, gpu in enumerate(gpus):
        i_commands = commands[i::n_gpu]
        if not i_commands:
            continue
            
        # Create command file with proper path
        sh_path = os.path.join(command_dir, f"run{i}.sh")
        with open(sh_path, "w") as fout:
            fout.write("#!/bin/bash\n")
            fout.write(f"export CUDA_VISIBLE_DEVICES={gpu}\n")
            for com in i_commands:
                full_cmd = f"{com} {ext_command.format(i=i)}"
                fout.write(f"{full_cmd}\n")
        
        # Make executable
        os.chmod(sh_path, 0o755)
        
        # Execute if requested
        if call:
            try:
                # Use full path when executing
                subprocess.Popen(
                    ["bash", os.path.abspath(sh_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                time.sleep(delay)
            except Exception as e:
                print(f"Error executing {sh_path}: {e}")
