from torch import nn
from training.utils.utils import batches_to_device, get_default_device

class Threshold(nn.Module):
    def __init__(self, threshold) -> None:
        super().__init__()
        self.t = threshold

    def forward(self, x):
        floor = x.floor()
        rest = x - floor
        return (rest >= self.t).int().float() + floor

def round_thresh(x, t):
    floor = x.floor()
    rest = x - floor
    return (rest >= t).int().float() + floor

def find_threshold(f, test_loader, freedom = 250):
    f.eval()

    step = 1 / freedom
    best_threshold = step
    best_acc = 0

    th_ac = {}
    batches_to_device(test_loader, get_default_device())

    for i in range(freedom):
        current_threshold = step * i
        good = 0
        total = 0
        for batch in test_loader:
            x, y = batch
            y_ = f(x)
            y_ = round_thresh(y_, current_threshold)

            good += (y_ == y).sum().item()
            total += y_.shape[0]

        current_acc = 100 * (good / total)
        print(f"current acc {current_acc}, th {current_threshold}")
        if best_acc < current_acc:
            best_acc = current_acc
            best_threshold = current_threshold
            print(f"|||||||||current best acc {best_acc}, th {best_threshold}")

        th_ac[current_threshold] = current_acc

    sorted_th_ac = sorted(th_ac.items(), key=lambda x:x[1], reverse=True)

    top_5 = sorted_th_ac[:5]
    
    # lets find the threhold where the surrounding values are the best
    top_mean = 0
    top_t = -1
    for thr in top_5:
        curr_mean = (th_ac[thr] + th_ac[thr - step] + th_ac[thr - 2 * step] + th_ac[thr + step] + th_ac[thr + 2 * step]) / 5
        if curr_mean > top_mean:
            top_t = thr


    return Threshold(top_t)
    


def get_th(t=0.512):
    return Threshold(t)

