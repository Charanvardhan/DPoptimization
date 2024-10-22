import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
import argparse
import csv
import itertools
import matplotlib.pyplot as plt
from resnet import resnet20

def save_results_to_csv(results, filename):
    if len(results) > 0:
        keys = results[0].keys()
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

def train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm):
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=trainloader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm
    )
    running_loss = 0.0
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    accuracy = evaluate_model(model, testloader)
    epsilon = privacy_engine.get_epsilon(args.delta)
    return running_loss / len(trainloader), accuracy, epsilon

def evaluate_model(model, testloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def run_sgd(args, trainloader, testloader):
    grid_params = {
        'batch_size': [128],
        'lr': [0.1, 0.01, 0.05],
        'noise_multiplier': [0.5, 1.0],
        'max_grad_norm': [1.0, 0.5],
    }
    results = []
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']
        model = resnet20(num_classes=10).cuda()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm)
        results.append({
            'batch_size': args.batch_size, 'learning_rate': lr, 'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm, 'loss': loss, 'accuracy': accuracy, 'epsilon': epsilon,
        })
    save_results_to_csv(results, 'sgd_tuning_results.csv')
    visualize_results(results)

def visualize_results(results):
    batch_sizes = [r['batch_size'] for r in results]
    learning_rates = [r['learning_rate'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    epsilons = [r['epsilon'] for r in results]
    plt.figure(figsize=(10, 6))
    for batch_size in set(batch_sizes):
        indices = [i for i, b in enumerate(batch_sizes) if b == batch_size]
        lr_vals = [learning_rates[i] for i in indices]
        acc_vals = [accuracies[i] for i in indices]
        plt.plot(lr_vals, acc_vals, label=f'Batch Size {batch_size}')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Learning Rate for Different Batch Sizes (SGD)')
    plt.legend()
    plt.xscale('log')
    plt.grid(True)
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.scatter(epsilons, accuracies, c=learning_rates, cmap='viridis', marker='o')
    plt.colorbar(label='Learning Rate')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epsilon for SGD Hyperparameter Tuning')
    plt.grid(True)
    plt.show()

def parse_args():
    parser = argparse.ArgumentParser(description="DP Training for CIFAR-10 using ResNet20")
    parser.add_argument('--optimizer', choices=['sgd'], required=True, help='Optimizer to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta for differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='Target epsilon for DP')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    if args.optimizer == 'sgd':
        run_sgd(args, trainloader, testloader)
