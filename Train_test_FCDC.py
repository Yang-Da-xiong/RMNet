import torch
import torch.nn.functional as F
import numpy as np

from tqdm import tqdm


def test(net, loss_func, test_dataloader, args):
    net.eval()
    for param in net.parameters():
        param.requires_grad = False

    test_loss = 0
    correct = 0.0
    for _, (images, labels) in enumerate(tqdm(test_dataloader)):
        images = images.to('cuda')
        labels = labels.to('cuda')
        output,_ = net(images)
        test_loss += loss_func(output, labels)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
    
    test_loss /= len(test_dataloader)
    args.write_log(
        args.log_path, "Test1 set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100 * correct / len(test_dataloader.dataset),
        )
    )

    return correct / len(test_dataloader.dataset)