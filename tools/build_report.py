import csv
import re
from os import listdir
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def pattern(key: str): return f'(\'{key}\':.+?(,|}}))'
def search(re_pattern: str, string: str): return (re.findall(re_pattern, string) or [""])[0]
def filter_num(num: str): return re.sub(r'[^0-9|\.]+', '', num)
def search_n_filter(re_pattern: str, string: str): return filter_num((search(re_pattern, string) or [""])[0])

VAL_LOSS = 'val_loss'
VAL_ACC = 'val_acc'
TRAIN_LOSS = 'train_loss'
TRAIN_ACC = 'train_acc'
EPOCH = 'epoch'
TIME = 'training_time'

OBJECT_PATTERN = r'\{.*\}'
TRAIN_LOSS_PATTERN = pattern(TRAIN_LOSS)
TRAIN_ACC_PATTERN = pattern(TRAIN_ACC)
VAL_LOSS_PATTERN = pattern(VAL_LOSS)
VAL_ACC_PATTERN = pattern(VAL_ACC)
EPOCH_PATTERN = pattern(EPOCH)
TIME_PATTERN = pattern(TIME)

HEADER = ['Epoch', 'Time for epoch (min)', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy']

class EpochResume:
    def __init__(self, train_loss: str, train_acc: str, val_loss: str, val_acc: str, epoch: str, time: str):
        self.train_loss = float(train_loss) if train_loss else -1
        self.train_acc = float(train_acc) if train_acc else -1
        self.val_loss = float(val_loss) if val_loss else -1
        self.val_acc = float(val_acc) if val_acc else -1
        self.epoch = int(epoch) if epoch else -1
        self.time = float(time) if time else -1
        self.time_in_min = self.time / 60.0

def extract_resume_from_logs(log_path: str):
    resume = []
    with open(log_path, "r") as f:
        for line in f:
            epoch_resume_str = search(OBJECT_PATTERN, line)
            if epoch_resume_str:
                resume += [EpochResume(train_loss=search_n_filter(TRAIN_LOSS_PATTERN, epoch_resume_str),
                                       train_acc=search_n_filter(TRAIN_ACC_PATTERN, epoch_resume_str),
                                       val_loss=search_n_filter(VAL_LOSS_PATTERN, epoch_resume_str),
                                       val_acc=search_n_filter(VAL_ACC_PATTERN, epoch_resume_str),
                                       epoch=search_n_filter(EPOCH_PATTERN, epoch_resume_str),
                                       time=search_n_filter(TIME_PATTERN, epoch_resume_str))]
    resume.sort(key=lambda res: res.epoch)
    return resume

def build_spreadsheet(epoch_resume: List[EpochResume], report_name: str, title: bool = False):
    with open(f'{report_name}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f, lineterminator = '\n')
        if title: writer.writerow([report_name])
        writer.writerow(HEADER)

        for res in epoch_resume: writer.writerow([res.epoch, res.time_in_min, res.train_loss, res.train_acc, res.val_loss, res.val_acc])

def build_graphs(data, path, name):
    epochs = list(range(len(data)))
    data = pd.DataFrame({'Epoch' : epochs, name : data})
    sp = sb.scatterplot(data=data, x='Epoch', y=name)
    plt.savefig(f"{path}_{name}.png")
    plt.figure().clear()


def build_spreadsheet_from_logs(log_path, report_name):
    resume = extract_resume_from_logs(log_path)
    build_spreadsheet(resume, report_name)

def build_graphs_from_logs(log_path):
    resume = extract_resume_from_logs(log_path)
    train_losss = [resume_line.train_loss for resume_line in resume]
    train_accs = [resume_line.train_acc for resume_line in resume]
    val_losss = [resume_line.val_loss for resume_line in resume]
    val_accs = [resume_line.val_acc for resume_line in resume]
    build_graphs(train_losss, log_path, name='train_losss')
    build_graphs(train_accs, log_path, name='train_accs')
    build_graphs(val_losss, log_path, name='val_losss')
    build_graphs(val_accs, log_path, name='val_accs')

def build_spreadsheet_from_log_folder(log_path):
    for log in listdir(log_path):
        resume = extract_resume_from_logs(f"{log_path}\{log}")
        build_spreadsheet(resume, log.replace(".out", ""))
