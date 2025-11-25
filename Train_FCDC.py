import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.nn as nn
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def train(epoch, net, optimizer, loss_1, loss_2, loss_3, train_dataloader_cross,  args, writer):
    net.train()
    for param in net.parameters():
        param.requires_grad = True
    
    all_feats = []
    all_labels = []
    total_loss = 0
    for step, data in enumerate(tqdm(train_dataloader_cross)):
        images, labels = data['images'].to('cuda'), data['labels'].to('cuda')
        images = images.view(-1, 1, 256, 256)
        labels = labels.view(-1)


        #加载cover-wow
        optimizer.zero_grad()
        output, feats_gap = net(images)
        #print(feats.shape)
        loss_cross = loss_1(output, labels)
        
        #print(feats_gap.shape)
        labels_all = labels
        
        #feats_gap = gap(feats_1).view(feats_1.size(0), -1)
        #labels_all = labels
        #print(labels_all.shape)

        if epoch % 5 ==0 and step % 25 == 0:
            # 提取特征和标签
            all_feats.append(feats_gap.detach().cpu().numpy())
            all_labels.append(labels_all.detach().cpu().numpy())
        #print(feats_gap.shape)  #(64,256)
        #print(labels_all.shape)  #(64)

        loss_weight_affinity = 5
        loss_affinity = loss_2(feats_gap, labels_all)
        #loss_affinity = torch.tensor(0.0).to('cuda')

        #print(labels_all)
        loss_CC = loss_3(feats_gap, labels_all)
        
        loss_weight_CC = 5
        #loss_CC = torch.tensor(0.0).to('cuda')

        #权重为1:1:1
        loss = loss_cross + loss_weight_affinity * loss_affinity + loss_weight_CC * loss_CC
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if step % int(len(train_dataloader_cross) / 8) == 0:
            b_pred = output.max(1, keepdim=True)[1]
            b_correct = b_pred.eq(labels.view_as(b_pred)).sum().item()
            b_accu = b_correct / (labels.size(0))

            # 记录到 TensorBoard
            writer.add_scalar('Train/Accuracy', b_accu, epoch * len(train_dataloader_cross) + step)
            writer.add_scalar('Train/Loss_Affinity', loss_weight_affinity * loss_affinity.item(), epoch * len(train_dataloader_cross) + step)
            writer.add_scalar('Train/Loss_Cross', loss_cross.item(), epoch * len(train_dataloader_cross) + step)
            writer.add_scalar('Train/Loss_CC', loss_weight_CC * loss_CC.item(), epoch * len(train_dataloader_cross) + step)
            writer.add_scalar('Train/Total_Loss', loss.item(), epoch * len(train_dataloader_cross) + step)

            args.write_log(
                args.log_path,
                "Train Epoch: {} [{}/{} ({:.0f}%)]\ttrain_accuracy: {:.6f}\t Loss_cross:{:.6f}\t Loss_affinity: {:.6f}\t Loss_CC: {:.6f}".format(
                    epoch,
                    (step + 1) * len(data),
                    len(train_dataloader_cross.dataset),
                    100.0 * (step + 1) / len(train_dataloader_cross),
                    b_accu,
                    loss_cross.item(),
                    loss_affinity.item(),
                    loss_CC.item(),
                )
            )
    
    args.write_log(
        args.log_path, "train Epoch: {}\tavgLoss: {:.6f}".format(epoch, total_loss / len(train_dataloader_cross))
    )
    
    
    # 生成并保存T-SNE图像
    if epoch % 5 == 0:
        generate_tsne(epoch, all_feats, all_labels, loss_2.centers.cpu().detach().numpy(), args)
    

    return net

def generate_tsne(epoch, all_feats, all_labels, centers, args):

    # 将提取的特征和标签转换为NumPy数组
    all_feats_np = np.concatenate(all_feats, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    # 合并特征和中心点
    combined_feats = np.concatenate([all_feats_np, centers], axis=0)

    #print(combined_feats.shape)
    #print(all_labels_np.shape)

    # 生成T-SNE图
    tsne = TSNE(n_components=2, random_state=42)
    tsne_combined = tsne.fit_transform(combined_feats)

    # 分离特征和中心点的嵌入
    tsne_feats = tsne_combined[:-centers.shape[0]]  # 特征
    tsne_centers = tsne_combined[-centers.shape[0]:]  # 中心点

    # 绘制T-SNE图
    plt.figure(figsize=(10, 8))
    colors = ['r', 'g', 'b', 'y']  # 2个类，准备2种颜色，r是自然图像，g是wow, b是SUNIWARD，y是HUGO
    labels_name = ['Cover', 'MiPOD', 'S-UNIWARD', 'HUGO']
    
    #这个plt的alpha是什么
    for i in range(2):
        plt.scatter(tsne_feats[all_labels_np == i, 0], tsne_feats[all_labels_np == i, 1], c=colors[i], marker='o', label=f'{labels_name[i]}', alpha=0.5, s=10)

    '''
    for i in range(2):
        plt.scatter(tsne_centers[i, 0], tsne_centers[i, 1], c=colors[i], marker='x', s=300, label=f'{labels_name[i]} Center')
    '''
    
    plt.legend()
    plt.title(f'T-SNE of Features and Centers at Epoch {epoch}')
    
    # 确保保存目录存在
    save_dir = os.path.join(args.model_path, "T-SNE")
    os.makedirs(save_dir, exist_ok=True)
    
    plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.jpg"))
    plt.show()

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
        args.log_path, "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)".format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return 100.0 * correct / len(test_dataloader.dataset)

def valid(net, loss_func, test_dataloader, args):
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
        args.log_path, "Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)".format(
            test_loss,
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )

    return 100.0 * correct / len(test_dataloader.dataset)