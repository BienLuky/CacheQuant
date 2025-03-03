import logging
import torch
import torch.nn as nn
import numpy as np
logger = logging.getLogger(__name__)

class AttentionMap:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        # self.ac_output = []

    def hook_fn(self, module, input, output):
        self.out = output
        self.feature = input
        # self.ac_output.append(output)

    def remove(self):
        self.hook.remove()


class AttentionMap_add:
    def __init__(self, module, interval_seq, end_t):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.out = []
        self.feature = []
        self.interval_seq = interval_seq
        self.current_t = 0
        self.end_t = end_t

    def hook_fn(self, module, input, output):
        if self.current_t in self.interval_seq:
            self.out.append(output.cpu())
        self.current_t += 1
        if self.current_t == self.end_t:
            self.current_t = 0

    def removeInfo(self):
        self.out.clear()
        self.feature.clear()

    def remove(self):
        self.hook.remove()


class AttentionMap_input_add:
    def __init__(self, module, interval_seq, end_t, split=192):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.out = []
        self.feature = []
        self.interval_seq = interval_seq
        self.current_t = 0
        self.end_t = end_t
        self.split = split

    def hook_fn(self, module, input, output):
        if self.current_t in self.interval_seq:
            self.feature.append(input[0][:, :self.split, ...].cpu())
        self.current_t += 1
        if self.current_t == self.end_t:
            self.current_t = 0

    def removeInfo(self):
        self.out.clear()
        self.feature.clear()

    def remove(self):
        self.hook.remove()


def seed_everything(seed):
    '''
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class Fisher():
    def __init__(self, samples, class_num, target="class_target"):
        self.target = target
        self.samples = samples
        self.class_num = class_num
        self.sample_num = len(samples)

    def get_class_ave(self, samples, start, end):
        class_ave = 0.0
        for i in range(start, end):
            class_ave += samples[i]
        class_ave = class_ave / (end - start)
        return class_ave

    def get_class_diameter(self, samples, start, end):
        class_diameter = 0.0
        for i in range(start, end):
            class_diameter += torch.sum(torch.abs(samples[i] - samples[start])) # F1
        return class_diameter

    def feature_to_interval_seq(self):
        split_loss_result = np.zeros((self.sample_num + 1, self.class_num + 1))
        split_class_result = np.zeros((self.sample_num + 1, self.class_num + 1))

        for n in range(1, self.sample_num + 1):
            split_loss_result[n, 1] = self.get_class_diameter(self.samples, 0, n)
            split_class_result[n, 1] = 1

        for k in range(2, self.class_num + 1):
            for n in range(k, self.sample_num + 1):
                loss = []
                sample_list = range(k - 1, n)
                for j in sample_list:
                    loss.append(split_loss_result[j, k - 1] + self.get_class_diameter(self.samples, j, n))
                split_loss_result[n, k], index = torch.stack(loss).min(dim=0)
                split_class_result[n, k] = sample_list[index]+1
        interval_seq = [None] * self.class_num
        sample_num = self.sample_num
        for k in reversed(range(self.class_num)):
            sample_num = int(split_class_result[sample_num][k+1]) - 1
            interval_seq[k] = sample_num
        return interval_seq

    def feature_to_interval_seq_optimal(self, interval):
        min_itval = int(interval/2) + (interval%2)
        max_itval = interval * 2

        split_loss_result = np.zeros((self.sample_num + 1, self.class_num + 1))
        split_class_result = np.zeros((self.sample_num + 1, self.class_num + 1))

        for n in range(min_itval, max_itval + 1): 
            split_loss_result[n, 1] = self.get_class_diameter(self.samples, 0, n)
            split_class_result[n, 1] = 0

        for k in range(2, self.class_num + 1):
            up_bound = k * max_itval if k * max_itval <= self.sample_num else self.sample_num 
            for n in range(k * min_itval, up_bound + 1):
                loss = []
                skip_for_max_itval = 0
                sample_up_bound = (k - 1) * max_itval if (k - 1) * max_itval <= self.sample_num else self.sample_num
                sample_list = range((k - 1) * min_itval, sample_up_bound + 1)
                for j in sample_list:
                    if n - j > max_itval:
                        skip_for_max_itval = skip_for_max_itval + 1
                    if n - j < min_itval or n - j > max_itval: 
                        continue
                    loss.append(split_loss_result[j, k - 1] + self.get_class_diameter(self.samples, j, n))
                split_loss_result[n, k], index = torch.stack(loss).min(dim=0)
                split_class_result[n, k] = sample_list[index + skip_for_max_itval]
        interval_seq = [None] * self.class_num
        sample_num = self.sample_num
        for k in reversed(range(self.class_num)):
            sample_num = int(split_class_result[sample_num][k+1])
            interval_seq[k] = sample_num
        return interval_seq
