import argparse
import pandas as pd
from datetime import datetime
from training.utils.utils import load_model
from tools.fps_benchmark import measure_fps
from training.metrics.metrics import count_model_parameters
from dataset.LLVIP import LLVIPDataset
from training.training_manager import TrainingManager
from tools.config_utils import load_config
from training.utils.utils import mkdir
import numpy as np
from scipy.optimize import linear_sum_assignment

def args():
    parser = argparse.ArgumentParser(description="DLTrainer")
    parser.add_argument("--mode", type=str, default='False', help="Whether to evaluation the 'accuracy', 'localization' or the 'speed'.")
    parser.add_argument("--model_path", type=str, default="vit", help="Path of the model to be evaluated.")
    parser.add_argument("--dataset_path", type=str, default="vit", help="Path of the dataset.")
    parser.add_argument("--split", type=str, default="vit", help="Whether to evaluate on the test, validation or training split.")

    opt = parser.parse_args()

    return opt

def evaluate_accuracy(model_path = None, split: str = 'test', dataset_path: str = '/data/llvip'):
    model = load_model(model_path)
    config = load_config([0, "convnext_configs"])
    dataset = LLVIPDataset(root=dataset_path, batch_size=config['hyperparameters']['batch_size'], classification=True)

    trainer = TrainingManager(config)
    trainer.evaluate(model, dataset, split)

def evaluate_localization(model_path = None, split: str = 'test', dataset_path: str = '/data/llvip'):
    model = load_model(model_path)
    config = load_config([0, "convnext_configs"])
    dataset = LLVIPDataset(root=dataset_path, batch_size=1, with_bbox=True)
    split_map = {'train' : 0, 'validation' : 1, 'test' : 2}
    loader = dataset.get_dataloaders()[split_map[split.lower()]]
    eval_path = '/evaluation/location/'
    date = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
    mkdir(eval_path)

    convnext_preds_list = []
    gt_list = []
    with open(f'{eval_path}CONVNEXT_LOCALIZATION_PREDS_{date}.txt', 'w+') as convnext_preds, open(f'{eval_path}GROUND_TRUTH_{date}.txt', 'w+') as gt:
        for d in loader:
            im, label = d
            anno, bbox = label
            bbox = bbox.reshape(-1, 5)
            bbox = [[bb[1].item(), bb[2].item()] for bb in bbox]
            out_conv = model(im)

            if anno == 0:
                continue
            peoples_location = model.locate_people(im, out_conv)
            convnext_preds_str = ''
            gt_str = ''
            convnext_preds_list.append(peoples_location)
            gt_list.append(bbox)
            for p, i in zip(peoples_location, range(len(peoples_location))):
                if i != 0:
                    convnext_preds_str += ' '
                convnext_preds_str += f'{p[0]} {p[1]}'
            for b, i in zip(bbox, range(len(bbox))):
                if i != 0:
                    gt_str += ' '
                gt_str += f'{b[0]} {b[1]}'
            convnext_preds.write(f"{convnext_preds_str}\n")
            gt.write(f"{gt_str}\n")

    matched=[]
    for k in range (len(p)):
        cost_matrix = np.zeros((len(gt[k]), len(p[k])))
        for i in range(len(gt[k])):
            for j in range(len(p[k])):
                cost_matrix[i][j] = np.linalg.norm(np.array(gt_list[k][i]) - np.array(convnext_preds_list[k][j]))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_coordinates = [(gt_list[k][i], convnext_preds_list[k][j]) for i, j in zip(row_ind, col_ind)]
        matched.append(matched_coordinates)

    total=[]
    for r in range(len(matched)):
        distances = 0
        for i, (ground_truth_point, predicted_point) in enumerate(matched[r]):
            distance = np.linalg.norm(np.array(ground_truth_point) - np.array(predicted_point))
            distances+=distance
        penalty = abs(len(convnext_preds_list[i]) - len(gt_list[i]))
        total.append((distances + penalty)/len(matched[r]))
    
    print(f"mAD : {sum(total)/len(total)}")

def measure_speed(model, device = "cpu"):
    report = measure_fps(model, device=device, repetitions=1000)
    return [model.__class__.__name__] + [report['MEAN_FPS']] + [report['STD_FPS']] + [count_model_parameters(model, False)] +['device']

def evaluate_speed(model_path = None):
    results = [["Models", "Mean FPS", "Std FPS", "Parameters Count", "device"]]
    model = load_model(model_path)
    results += measure_speed(model, 'cpu')
    results += measure_speed(model, 'cuda')

    df = pd.DataFrame(results)
    print(df)

if __name__ == '__main__':
    opt = args()
    
    if   opt.mode.lower() in 'accuracy':     evaluate_accuracy(opt.model_path, opt.split, opt.dataset_path)
    elif opt.mode.lower() in 'localization': evaluate_localization(opt.model_path, opt.split, opt.dataset_path)
    elif opt.mode.lower() in 'speed':        evaluate_speed(opt.model_path)