import math
from torch.optim.lr_scheduler import LRScheduler


class CosineWarmUpScheduler(LRScheduler):
    def __init__(self, optimizer, num_warmup_steps, total_steps, eta_min=0, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super(CosineWarmUpScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.num_warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * self.last_epoch / self.num_warmup_steps for base_lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) * (1 + math.cos((self.last_epoch - self.num_warmup_steps) * math.pi / (self.total_steps - self.num_warmup_steps))) / 2 for base_lr in self.base_lrs]
