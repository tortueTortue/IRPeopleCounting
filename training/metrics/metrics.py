"""
Here are the the methods computing the metrics used for evaluaton
"""
import torch
from torch.nn import Module, MSELoss, L1Loss
from training.utils.utils import unpack, cuda

def prob_f1(outputs: torch.Tensor, labels: torch.Tensor, beta: float = 1, epsilon: float = 1e-15, on_gpu: bool = True):
    y_true_count = 0
    ctp = 0
    cfp = 0

    for idx in range(len(labels)):
        prediction = min(max(outputs[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction

    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp)
    c_recall = ctp / y_true_count if y_true_count else epsilon
    if (c_precision > 0 and c_recall > 0):
        result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        return result
    else:
        return cuda(torch.Tensor([0.0])) if on_gpu else torch.Tensor([0.0])

def accuracy(outputs, labels):
    """
    Accuracy = no_of_correct_preidctions / no_of_predictions

    *Note: Use this when the classes have about the amount of occurences.
    """
    preds = outputs

    return torch.tensor(torch.sum(preds.int() == labels.int()).item() / len(preds))

def print_accuracy(model: Module, classes: list, batch_size: int, test_loader):
    class_correct = 0
    class_total = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            i, l, idx = unpack(batch)
            if torch.cuda.is_available():
                i, l = cuda(i), cuda(l)
                if idx is not None: idx = cuda(idx)
            predicted = model(i, indices=idx) if idx is not None else model(i)
            c = (predicted.reshape(l.shape).int() == l.int())
            for i in range(l.shape[0]):
                class_correct += c[i].item()
                class_total += 1

    print('Accuracy of : %2f %%' % (100 * class_correct / class_total))

def classify_batch(model, batch, with_count_acc: bool = True, weighted_count_acc: bool = True):
    loss = MSELoss()
    inputs, labels = batch
    if torch.cuda.is_available(): inputs, labels = cuda(inputs), cuda(labels)
    out = model(inputs)
        
    report = {'loss': loss(out, labels).detach()}

    if with_count_acc:
        good = (out.int() == labels.int()).sum().detach().item()
        total = labels.shape[0]
        report['good'] = good
        report['total'] = total

    if weighted_count_acc:
        accs_per_class = {}
        for o, l in zip(out, labels):
            l_ = int(l.detach().item())
            class_count_acc = accs_per_class.get(l_, {'total' : 0, 'good' : 0})
            class_count_acc['total'] += 1
            class_count_acc['good'] += int(l.int() == o.int())
            accs_per_class[l_] = class_count_acc
        
        report['accs_per_class'] = accs_per_class

    return report

def print_loss(model: Module, test_loader, with_count_acc: bool = True, print_: bool = True, weighted_count_acc: bool = True):
    model.eval()
    outputs = [classify_batch(model, batch) for batch in test_loader]

    batch_losses = [x['loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean().item()

    report = [epoch_loss]

    if print_: print(f'Loss :{epoch_loss}')

    if with_count_acc:
        batch_good = [x['good'] for x in outputs]
        batch_total = [x['total'] for x in outputs]
        epoch_good = sum(batch_good)
        epoch_acc = 100 * (epoch_good / sum(batch_total))

        if print_: print(f'Count Accuracy :{epoch_acc} %')
        report += [epoch_acc]

    if weighted_count_acc:
        accs_per_class = {}
        for o in outputs:
            for l, r in o['accs_per_class'].items():
                curr = accs_per_class.get(l, {'total' : 0, 'good' : 0})
                curr['total'] += r['total']
                curr['good']  += r['good']
                accs_per_class[l] = curr
        report += [accs_per_class]

    return report

def compute_weighted_count_acc(class_accs):
    non_empty_classes_count = sum([0 if o['total'] == 0 else 1 for o in class_accs.values() ])
    accuracy = 0
    for _, stats in class_accs.items():
        weight = 1 / non_empty_classes_count
        class_accuracy = stats['good'] / stats['total'] if stats['total'] > 0 else 0
        accuracy += weight * class_accuracy

    return accuracy

def print_accuracy_per_class(model: Module, classes: list, batch_size: int, test_loader):
    class_amount = len(classes)
    class_correct = list(0. for i in range(class_amount))
    class_total = list(0. for i in range(class_amount))
    l2 = MSELoss()
    l1 = L1Loss()
    l2_loss = 0
    l1_loss = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            i, l, idx = unpack(batch)
            if torch.cuda.is_available():
                i, l = cuda(i), cuda(l)
                if idx is not None: idx = cuda(idx)
            predicted = model(i, indices=idx) if idx is not None else model(i)
            c = (predicted.reshape(l.shape).int() == l.int())
            l1_loss += l1(predicted.reshape(l.shape).float(), l.float())
            l2_loss += l2(predicted.reshape(l.shape).float(), l.float())
            for i in range(l.shape[0]):
                label = l[i].int().item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    for i in range(class_amount):
        print('Accuracy of %5s : %2f %%' % (
            classes[i], 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else -1))
    
    num_classes = 0
    wacc = 0
    for i in range(class_amount):
        if class_total[i] > 0:
            num_classes += 1
            wacc += class_correct[i] / class_total[i]

    print(f'Weighted Accuracy of {(wacc/num_classes) * 100 if num_classes  > 0 else -1} %')
    num_batches = len(test_loader)
    print(f'MSE of {l2_loss / num_batches if num_classes  > 0 else -1}')
    print(f'MAE of {l1_loss / num_batches if num_classes  > 0 else -1}')

def count_model_parameters(model: Module, trainable: bool):
    """
    Returns the total amount of parameters in a model.
    
    args:
        model : model to count the parameters of
        trainable : whether to count the trainable params or not
    """
    return (sum(p.numel() for p in model.parameters()) ) if not trainable else \
           (sum(p.numel() for p in model.parameters() if p.requires_grad))
