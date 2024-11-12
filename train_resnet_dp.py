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
        keys = results[0].keys()  # Extract column names from the first result
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)

def train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm, csv_filename):
    privacy_engine = PrivacyEngine()
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model, optimizer=optimizer, data_loader=trainloader,
        noise_multiplier=noise_multiplier, max_grad_norm=max_grad_norm
    )
    
    results = []
    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate the training loss and accuracy
        train_loss = running_loss / len(trainloader)
        test_accuracy = evaluate_model(model, testloader)
        
        # Get the privacy epsilon value
        epsilon = privacy_engine.get_epsilon(args.delta)

        # Save results for this epoch
        epoch_result = {
            'epoch': epoch + 1,
            'batch_size': args.batch_size,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'train_loss': train_loss,
            'test_accuracy': test_accuracy,
            'epsilon': epsilon
        }
        results.append(epoch_result)

        # Print training progress
        print(f"Epoch [{epoch+1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%, Epsilon: {epsilon:.2f}")

        # Save results after each epoch
        save_results_to_csv(results, csv_filename)

    return results

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
    accuracy = 100 * correct / total
    return accuracy

def run_sgd(args, trainloader, testloader):
    grid_params = {
        'batch_size': [128],
        'lr': [0.1, 0.01, 0.05],
        'noise_multiplier': [0.5, 1.0],
        'max_grad_norm': [1.0, 0.5],
    }

    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']

        # Load ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # CSV file name for storing results
        csv_filename = f'sgd_results_episilon_0.1_batch_{args.batch_size}_lr{lr}_noise{noise_multiplier}_gradnorm{max_grad_norm}.csv'

        # Train the model and store results
        train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm, csv_filename)

def run_sgd_momentum(args, trainloader, testloader):
    grid_params = {
        'batch_size': [128,256],
        'lr': [0.1, 0.01, 0.05],
        'momentum': [0.9],
        'noise_multiplier': [0.5, 1.0],
        'max_grad_norm': [1.0, 0.5],
        'epsilon': [0.1, 0.5, 1.0],
    }

    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        momentum = param_dict['momentum']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']
        epsilon = param_dict['epsilon']

        # Load ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define SGD with Momentum optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        # CSV file name for storing results
        csv_filename = f'sgd_momentum_results_epsilon_{epsilon}_batch_{args.batch_size}_lr{lr}_momentum{momentum}_noise{noise_multiplier}_gradnorm{max_grad_norm}.csv'

        # Train the model and store results
        train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm, csv_filename)

def run_adam(args, trainloader, testloader):
    grid_params = {
        'batch_size': [128,256],
        'lr': [0.01, 0.05, 0.1],
        'beta1': [0.9],
        'beta2': [0.999],
        'noise_multiplier': [0.5, 1.0],
        'max_grad_norm': [1.0, 0.5],
        'epsilon': [0.1, 0.5, 1.0],
    }

    # Iterate over all combinations of hyperparameters
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        beta1 = param_dict['beta1']
        beta2 = param_dict['beta2']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']
        epsilon = param_dict['epsilon']

        # Load ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define Adam optimizer
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2))

        # CSV file name for storing results
        csv_filename = f'adam_results_epsilon_{epsilon}_batch_{args.batch_size}_lr{lr}_beta1{beta1}_beta2{beta2}_noise{noise_multiplier}_gradnorm{max_grad_norm}.csv'

        # Train the model and store results
        train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm, csv_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="DP Training for CIFAR-10 using ResNet20")
    parser.add_argument('--optimizer', choices=['sgd', 'sgd_momentum', 'adam'], required=True, help='Optimizer to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta for differential privacy')
    # parser.add_argument('--target_epsilon', type=float, default=3.0, help='Target epsilon for DP')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Define the dataset transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Run the optimizer specified by the user
    if args.optimizer == 'sgd':
        run_sgd(args, trainloader, testloader)
    elif args.optimizer == 'sgd_momentum':
        run_sgd_momentum(args, trainloader, testloader)
    elif args.optimizer == 'adam':
        run_adam(args, trainloader, testloader)
