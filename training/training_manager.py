import torch
import time
import os
import logging

from pyclbr import Function
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, CrossEntropyLoss, MSELoss, BCELoss
from training.loss.mae_loss import MaskedAutoEncoderLoss, FCMaskedAutoEncoderLoss
from torch.optim import SGD, Adam, AdamW
from statistics import mean

from training.utils.utils import batches_to_device, get_default_device, to_device, save_checkpoints, unpack, cuda, mkdir
from training.metrics.metrics import accuracy, prob_f1, print_accuracy_per_class, print_accuracy, count_model_parameters, print_loss
from training.utils.logger import start_training_logging
from datetime import datetime
from tools.build_report import build_spreadsheet_from_logs, build_graphs_from_logs
from training.utils.utils import save_model, load_model, load_checkpoint

ADAM, SGD_, ADAMW = 'ADAM', 'SGD', 'ADAMW'
LOSS = {'CrossEntropyLoss' : CrossEntropyLoss, 'MSELoss' : MSELoss, 'BCELoss' : BCELoss, 'MaskedAutoEncoderLoss' : MaskedAutoEncoderLoss, 'FCMaskedAutoEncoderLoss' : FCMaskedAutoEncoderLoss}
DEFAULT_CONFIG = {'hyperparameters' : {  'epochs'         : 100,
                                         'learning_rate'  : 0.001,
                                         'momentum'       : 0.9,
                                         'batch_size'     : 128,
                                         'beta_1'        : 0.9,
                                         'beta_2'        : 0.999},
                  'loss'            : "CrossEntropyLoss",
                  'optimizer'       : "ADAM",
                  'debug'           : False,
                  'model_name'      : "model_x",
                  'note'            : "",
                  'model_directory' : f"/models/trained_models/",
                  'checkpoint_dir'  : f"/training/checkpoints/",
                  'with_scheduling' : True }


class TrainingManager:

    def __init__(self, config: dict = None):
        self.config = config if config else DEFAULT_CONFIG
        self.model_name = self.config['model_name']
        self.checkpoint_dir = self.config['checkpoint_dir']
        self.debug = self.config['debug']
        self.loss = LOSS[self.config['loss']]()

        self.top_size = self.config.get("top_size", 3)
        self.leaderboard = [{'epoch': - (epoch + 1), 'score': 0} for epoch in range(self.top_size)] # Top epochs
        """If False, test on last epoch"""
        self.test_on_top_1: bool = self.config.get("test_on_top_1", True)

        self.epochs = self.config['hyperparameters']['epochs']
        self.with_scheduling = self.config.get("with_scheduling", True)
        self.warmup_epochs = self.config['hyperparameters'].get("warmup_epochs", 40)
        date = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
        self.run_dir = f"runs/{self.model_name}_{date}/"
        mkdir(self.run_dir)
        self.epoch_resume_file_name = f"{self.run_dir}/{self.model_name}_checkpoint_details.log"
        mkdir(self.config['model_directory'])
        mkdir(self.config['checkpoint_dir'])

    def __re_rank(self, epoch, score):
        self.leaderboard = ([{'epoch': epoch, 'score': score}] + self.leaderboard)
        self.leaderboard.sort(key=lambda epoch : epoch['score'], reverse=True)
        self.leaderboard = self.leaderboard[:-1]

    def __init_optimizer(self):
        optimizer = self.config['optimizer']
        params = self.model.parameters()
        momentum = self.config['hyperparameters']['momentum']
        betas = (self.config['hyperparameters']['beta_1'], self.config['hyperparameters']['beta_2'])
        lr = self.config['hyperparameters']['learning_rate']

        if optimizer == ADAM:
            self.optimizer = Adam(params, lr=lr, betas=betas)
        elif optimizer == ADAMW:
            self.optimizer = AdamW(params, lr=lr, betas=betas, weight_decay=self.config['hyperparameters'].get('weight_decay', 0.3))
        elif optimizer == SGD_:
            self.optimizer = SGD(params, lr=lr, momentum=momentum)
        
        if self.with_scheduling:
            self.epochs += self.warmup_epochs
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, self.warmup_epochs)


    def show_training_configs(self):
        info = [f"*********************************Training   Parameters*********************************",
                f"Learning rate      : {self.config['hyperparameters']['learning_rate']}",
                f"Batch Size         : {self.config['hyperparameters']['batch_size']}",
                f"Epochs             : {self.config['hyperparameters']['epochs']}",
                f"Num of Heads       : {self.config['hyperparameters']['num_heads']}",
                f"Loss Function      : {self.config['loss']}",
                f"Dataset            : {self.dataset.__class__.__name__}",
                f"Dataset Length     : {len(self.dataset)}",
                f"Optimizer          : {self.optimizer}",
                f"With Scheduling?   : {self.with_scheduling} {f', Scheduler : {self.scheduler.__class__.__name__}, Warmup Steps : {self.warmup_epochs}' if self.with_scheduling else ''}",
                f"Note               : {self.config['note']}",
                f"*********************************Training {self.config['model_name']}*********************************",
                f"Num of layers : {self.model.num_layers if hasattr(self.model, 'num_layers') else len(self.model._modules)}",
                f"Parameters    : {count_model_parameters(self.model, False)}",
                f"*********************************Training Epoch Results*********************************"]
        
        for line in info: print(line)

    def load_dataset(self, dataset: Dataset, device):
        train_loader, val_loader, test_loader = dataset.get_dataloaders(self.config['hyperparameters']['batch_size'])

        batches_to_device(train_loader, device)
        batches_to_device(val_loader, device)
        batches_to_device(test_loader, device)

        return train_loader, val_loader, test_loader

    def load_model(self, device):
        to_device(self.model, device)

    def log_results(self, results: str, epoch_training_time: float):
        results['training_time'] = epoch_training_time
        print(results)
        with open(self.epoch_resume_file_name, 'a+') as f:
            f.write(str(results)+'\n')

    def save_checkpoints(self, epoch, epoch_block_size = 1):
        if epoch % epoch_block_size == 0 :
            save_checkpoints(epoch, self.model, self.optimizer, self.loss,  f"{os.getcwd()}{self.config['checkpoint_dir']}/checkpoint_{epoch}_{self.config['model_name']}.pt")

    def update(self, current_loss):
        self.optimizer.zero_grad()
        current_loss.backward()
        self.optimizer.step()

    def compile_iteration_results(self, results, out, labels, current_loss):
        results['losses'].append(current_loss.item())
        if not isinstance(self.loss, MSELoss):
            _, preds = torch.max(out, dim=1)
            results['total_preds'] += len(preds)
            results['correct_preds'] += torch.sum(preds == labels).item()

        return results

    def training_step_results(self, iteration_results):
        losses = iteration_results['losses']
        if not isinstance(self.loss, MSELoss):
            correct = iteration_results['correct_preds']
            total = iteration_results['total_preds']
            return {'train_loss': mean(losses), 'train_acc': correct/total}
        else:
            return {'train_loss': mean(losses)}

    def training_step(self, batch, epoch, iteration):
        inputs, labels = batch
        if torch.cuda.is_available():
            inputs, labels = cuda(inputs), cuda(labels)

        out = self.model(inputs)
        current_loss = self.loss(out, labels)

        self.update(current_loss)

        if self.iteration_end_hook: self.iteration_end_hook(model=self.model, batch=batch, epoch=epoch, iteration=iteration)

        return out, labels, current_loss

    def classify_batch(self, batch):
        inputs, labels = batch
        if torch.cuda.is_available(): inputs, labels = cuda(inputs), cuda(labels)
        out = self.model(inputs)

        return {'val_loss': self.loss(out, labels).detach()} if isinstance(self.loss, MaskedAutoEncoderLoss) \
               else {'val_loss': torch.nn.functional.mse_loss(out.squeeze(-1), labels.squeeze(-1)).detach(), 'val_acc': accuracy(out.reshape(labels.shape), labels)}

    def validation_step(self, val_loader: DataLoader, epoch: int):
        self.model.eval()
        outputs = [self.classify_batch(batch) for batch in val_loader]
        self.model.train()

        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean().item()

        if isinstance(self.loss, MaskedAutoEncoderLoss):
            self.__re_rank(epoch, -epoch_loss)
            return {'val_loss': epoch_loss,'epoch' : epoch}
        else :
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean().item()

            self.__re_rank(epoch, epoch_acc)

            return {'val_loss': epoch_loss, 'val_acc': epoch_acc, 'epoch' : epoch}

    def update_lr(self):
        if self.scheduler:
            self.scheduler.step()

    def build_report(self):
        build_spreadsheet_from_logs(self.epoch_resume_file_name, self.epoch_resume_file_name.replace('.log', ''))
        build_graphs_from_logs(self.epoch_resume_file_name,)

    def save(self):
        """Save Last"""
        save_model(self.model, f"{self.config['model_name']}_last", f"{os.getcwd()}{self.config['model_directory']}")
        
        """Save Best"""
        model_path = f"{os.getcwd()}/training/checkpoints/checkpoint_{self.leaderboard[0]['epoch']}_{self.config['model_name']}.pt"
        load_checkpoint(self.model, model_path) #load_model
        save_model(self.model, f"{self.config['model_name']}_best", f"{os.getcwd()}{self.config['model_directory']}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        # 1. Epoch loop
        self.model.train()
        for epoch in range(self.epochs):
            # 1.1 Start epoch counter
            start_time = time.perf_counter()
            # 1.2 Training phase
            iteration_results = {'losses' : [], 'correct_preds' : 0, 'total_preds': 0}

            for iteration, batch in enumerate(train_loader):
                # 1.2.1 Training Step
                out, labels, current_loss = self.training_step(batch, epoch, iteration)
                # 1.2.1 Compiling results
                self.compile_iteration_results(iteration_results, out, labels, current_loss)
            
            # 1.3 Stop epoch counter
            if self.epoch_end_hook: self.epoch_end_hook(model=self.model, epoch=epoch)

            # 1.4 Stop epoch counter
            epoch_training_time = time.perf_counter() - start_time
            
            # 1.5 Validation Step
            epoch_results = {**self.validation_step(val_loader, epoch), **self.training_step_results(iteration_results)}
            
            # 1.6 Show results
            self.log_results(epoch_results, epoch_training_time)

            # 1.7 Save Checkpoint
            self.save_checkpoints(epoch)

            # 1.8 Update LR
            self.update_lr()


    def test(self, test_loader, classes, test = True):
        if test:
            print(f"*********************************Testing  {self.config['model_name']}*********************************")
            model = self.model
            model.eval()

            if isinstance(self.loss, CrossEntropyLoss):
                class_results = print_accuracy_per_class(model, classes, self.config['hyperparameters']['batch_size'], test_loader)
                results = print_accuracy(model, classes, self.config['hyperparameters']['batch_size'], test_loader)
                with open(f"{self.run_dir}/Testing_Results", 'a+') as f:
                    f.write(f"*********************************Testing  {self.config['model_name']}*********************************\n")
                    for line in class_results: f.write(line+'\n')
                    f.write(results+'\n')


    def fit(self, model: Module, dataset: Dataset, iteration_end_hook = None, epoch_end_hook = None, test = True):
        self.model = model
        self.__init_optimizer()
        self.epoch_end_hook = epoch_end_hook
        self.iteration_end_hook = iteration_end_hook
        self.dataset = dataset

        self.show_training_configs()

        #  Load model and dataset
        device = get_default_device()
        train_loader, val_loader, test_loader = self.load_dataset(dataset, device)
        self.load_model(device)

        self.train(train_loader, val_loader)
        self.build_report()
        self.save()

        self.test(test_loader, dataset.classes, test)

    def evaluate(self, model, dataset, split: str = 'test'):
        self.model = model
        self.dataset = dataset

        # 1. Show training configs
        self.show_training_configs()

        # 2. Load model and dataset
        device = get_default_device()
        train_loader, val_loader, test_loader = self.load_dataset(dataset, device)
        if split.lower() in 'training':
            selected_loader = train_loader
        elif split.lower() in 'validation':
            selected_loader = val_loader
        else:
            selected_loader = test_loader
        self.load_model(device)

        # 5. Test model
        self.test(selected_loader, dataset.classes)
