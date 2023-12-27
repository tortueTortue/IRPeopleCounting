import sys, copy, re, argparse, torch

from torch import nn
from training.training_manager import TrainingManager, DEFAULT_CONFIG
from tools.config_utils import load_config
from dataset.LLVIP import LLVIPDataset
from dataset.DistechUnlabelledSequences import DistechSequencesDatasetV19
from dataset.DistechV11 import DistechV11Dataset
from dataset.DistechV19 import DistechV19Dataset
from models.CrowdViT import MAEViT, VeryBigCrowdViT
from models.CrowdConvNext import CrowdConvNext, MAEConvNext
from training.utils.utils import load_model, load_checkpoint
from tools.fps_benchmark import measure_fps
from training.metrics.metrics import print_loss, compute_weighted_count_acc

def parse_checkpoint(cp_path):
    pat = r'checkpoint_\d{1,4}'
    out = re.search(pat, cp_path).group(0)
    out = re.sub("[^0-9]", "", out)
    out = int(out)

    return out

def args():
    def to_bool(str_): return 'true' in str_.lower()
    parser = argparse.ArgumentParser(description="DLTrainer")
    parser.add_argument("--mae", type=str, default='False', help="Whether to use Masked Autoencoder pretraining or not.")
    parser.add_argument("--model_type", type=str, default="vit", help="ViT or ConvNeXt")
    parser.add_argument("--pretrained", type=str, default='True')
    parser.add_argument("--head", type=str, default="classification", help="classification or regression")
    parser.add_argument("--dataset_root", type=str, default='/data/LLVIP/infrared/', help="Dataset Root path")
    parser.add_argument("--mae_cp_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--sub_ratio", type=float, default=1, help="Train ratio")
    parser.add_argument("--small", type=str, default='False', help="Whether to use the small version or normal of the model")
    
    if opt.model_type.lower() in 'convnext': opt.model_type = 'cnn'
    opt = parser.parse_args()
    opt.mae = to_bool(opt.mae)
    opt.pretrained = to_bool(opt.pretrained)
    opt.small = to_bool(opt.small)

    return opt



def train(pretrained: bool = True, mae: bool = False, cp_path: str = None, model_type: str = '', head_type: str = 'classification', sub_ratio: float = 1, small: bool = False, dataset_root = '/data/LLVIP/infrared/'):
    is_vit = not (model_type in 'cnn')
    classification_head = head_type in 'classification'
    config = load_config([0, "vit_configs" if is_vit else "convnext_configs"])
    # config['model_name'] += f'_{dataset}'
    if classification_head:
        config['loss'] = 'CrossEntropyLoss'
        config['model_name'] += '_CLASS_HEAD'
    else:
        config['model_name'] += '_REG_HEAD'
    
    if sub_ratio < 1:
        config['model_name'] += f'_RATIO_{str(sub_ratio).replace(".", "_")}'

    if small:
        config['model_name'] += f'_small'

    if is_vit:
        model = VeryBigCrowdViT(pretrained, classification_head, layers=3 if small else 4)
    else :
        model = CrowdConvNext(pretrained=pretrained, classification_head=classification_head, layers=5 if small else 9)
    
    if mae:
        mae_config = load_config([0, "mae_vit_configs" if is_vit else "mae_convNext_configs"])
        # mae_config['model_name'] += f'_{dataset}'
        if small:
            mae_config['model_name'] += f'_small'
        mae_dataset = LLVIPDataset(root=dataset_root, batch_size=mae_config['hyperparameters']['batch_size'], decode=True, merge_n_split=True, sub_ratio=sub_ratio)
        maeModel = MAEViT(model) if is_vit else MAEConvNext(model)
        if cp_path:
            if is_vit:
                maeModel = MAEViT(VeryBigCrowdViT(pretrained, classification_head, layers=3 if small else 4))
            if 'checkpoint' in cp_path:
                load_checkpoint(maeModel, cp_path)
                cp_epoch = parse_checkpoint(cp_path)
                mae_config['hyperparameters']['warmup_epochs'] = 2
                mae_config['hyperparameters']['epochs'] -= cp_epoch
                mae_trainer = TrainingManager(mae_config)
                mae_trainer.fit(maeModel, mae_dataset, test=False)
            else:
                maeModel = load_model(cp_path)
                if small:
                    if is_vit:
                        maeModel.encoder.vit.encoder.layers = maeModel.encoder.vit.encoder.layers[:3]
                    else:
                        maeModel.encoder.convnextv1.features[5] = maeModel.encoder.convnextv1.features[5][:5]
        else:
            mae_trainer = TrainingManager(mae_config)
            mae_trainer.fit(maeModel, mae_dataset, test=False)
        model = VeryBigCrowdViT(backbone=maeModel.encoder.vit, classification_head=model.classification_head) if is_vit else maeModel.get_encoder(model.classification_head)
        config['model_name'] += '_POST_MAE'
        if is_vit:
            config['hyperparameters']['learning_rate'] = 1E-5
            config['hyperparameters']['warmup_epochs'] = 5
            config['hyperparameters']['beta_2'] = 0.999
            config['hyperparameters']['weight_decay'] = 0.05
        else : 
            config['hyperparameters']['learning_rate'] = 2E-4
            config['hyperparameters']['weight_decay'] = 0.05

    dataset = LLVIPDataset(root=dataset_root, batch_size=config['hyperparameters']['batch_size'], classification=classification_head, sub_ratio=sub_ratio)

    trainer = TrainingManager(config)
    trainer.fit(model, dataset)


if __name__ == '__main__':
    opt = args()
    
    train(pretrained=True,
          mae=opt.mae,
          model_type=opt.model_type,
          head_type=opt.head,
          cp_path=opt.mae_cp_path,
          sub_ratio=opt.sub_ratio,
          small=opt.small,
          dataset_root=opt.dataset_root)