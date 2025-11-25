import os
import yaml
import torch
import torch.nn as nn

def yaml_config_hook(config_file):
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]
    return cfg


def write_log(log_path, write_info):
    print(write_info)
    with open(log_path, 'a') as f:
        f.write(str(write_info) + '\n')


def initWeights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(
                module.weight.data, mode="fan_in", nonlinearity="relu"
            )


def sgd_optimizer(model, lr, momentum=0.9, weight_decay=1e-4):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        apply_lr = lr
        apply_wd = weight_decay
        if 'bias' in key:
            apply_lr = 2 * lr       #   Just a Caffe-style common practice. Made no difference.
        if 'depth' in key:
            apply_wd = 0
        params += [{'params': [value], 'lr': apply_lr, 'weight_decay': apply_wd}]
    optimizer = torch.optim.SGD(params, lr, momentum=momentum)
    return optimizer


def accuracy(outputs, labels):
    _, argmax = torch.max(outputs, 1)
    return (labels == argmax.squeeze()).float().mean()