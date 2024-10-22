import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from torch.optim import Adam, RMSprop
from resnet import resnet20
import argparse
import csv
import itertools

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Differentially Private Training for CIFAR-10 using ResNet20")
    
    parser.add_argument('--optimizer', choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop'], required=True, help='Optimizer to run')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta for differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='Target epsilon for DP')
    args = parser.parse_args()
    return args

# Function to save results into a CSV file
def save_results_to_csv(results, filename):
    keys = results[0].keys()  # Column names
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

# Main training function
from opacus import PrivacyEngine

def train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm):
    print(f"Training ResNet20 with {optimizer.__class__.__name__} under DP constraints (ε ≤ {args.target_epsilon}, δ = {args.delta})")

    # Initialize the Privacy Engine
    privacy_engine = PrivacyEngine()

    # Attach the privacy engine to the optimizer and model
    model, optimizer, trainloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=trainloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    for epoch in range(args.epochs):
        running_loss = 0.0
        model.train()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate after each epoch
        accuracy = evaluate_model(model, testloader)
        print(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')

        # Track and display privacy budget
        epsilon = privacy_engine.get_epsilon(args.delta)  # Only returns epsilon, no need to unpack
        print(f"Epoch {epoch+1}: (ε = {epsilon:.2f}, δ = {args.delta})")
        
        if epsilon > args.target_epsilon:
            print(f"Warning: Privacy budget exceeded! ε = {epsilon:.2f}, which is greater than the target ε = {args.target_epsilon}")
            break  # Stop training if privacy budget exceeded

    return running_loss / len(trainloader), accuracy, epsilon

# Evaluation function
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

# Function to run grid search for SGD optimizer
def run_sgd(args, trainloader, testloader):
    grid_params = {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'noise_multiplier': [0.5, 1.1],
        'max_grad_norm': [1.0, 0.5],
    }

    results = []
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']

        # Load the ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define the SGD optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr)

        # Train the model
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm)

        # Save the results
        results.append({
            'optimizer': 'sgd',
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'loss': loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        })

    save_results_to_csv(results, 'sgd_results.csv')

# Function to run grid search for SGD with Momentum optimizer
def run_sgd_momentum(args, trainloader, testloader):
    grid_params = {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'noise_multiplier': [0.5, 1.1],
        'max_grad_norm': [1.0, 0.5],
    }

    results = []
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']

        # Load the ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define the SGD with Momentum optimizer
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Train the model
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm)

        # Save the results
        results.append({
            'optimizer': 'sgd_momentum',
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'loss': loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        })

    save_results_to_csv(results, 'sgd_momentum_results.csv')

# Function to run grid search for Adam optimizer
def run_adam(args, trainloader, testloader):
    grid_params = {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'noise_multiplier': [0.5, 1.1],
        'max_grad_norm': [1.0, 0.5],
    }

    results = []
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']

        # Load the ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define the Adam optimizer
        optimizer = Adam(model.parameters(), lr=lr)

        # Train the model
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm)

        # Save the results
        results.append({
            'optimizer': 'adam',
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'loss': loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        })

    save_results_to_csv(results, 'adam_results.csv')

# Function to run grid search for RMSprop optimizer
def run_rmsprop(args, trainloader, testloader):
    grid_params = {
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'noise_multiplier': [0.5, 1.1],
        'max_grad_norm': [1.0, 0.5],
    }

    results = []
    for params in itertools.product(*grid_params.values()):
        param_dict = dict(zip(grid_params.keys(), params))
        args.batch_size = param_dict['batch_size']
        lr = param_dict['lr']
        noise_multiplier = param_dict['noise_multiplier']
        max_grad_norm = param_dict['max_grad_norm']

        # Load the ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Define the RMSprop optimizer
        optimizer = RMSprop(model.parameters(), lr=lr)

        # Train the model
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader, noise_multiplier, max_grad_norm)

        # Save the results
        results.append({
            'optimizer': 'rmsprop',
            'batch_size': args.batch_size,
            'learning_rate': lr,
            'noise_multiplier': noise_multiplier,
            'max_grad_norm': max_grad_norm,
            'loss': loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        })

    save_results_to_csv(results, 'rmsprop_results.csv')

if __name__ == "__main__":
    args = parse_args()

    # Load CIFAR-10 dataset
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

    # Call the appropriate function based on the optimizer argument
    if args.optimizer == 'sgd':
        run_sgd(args, trainloader, testloader)
    elif args.optimizer == 'sgd_momentum':
        run_sgd_momentum(args, trainloader, testloader)
    elif args.optimizer == 'adam':
        run_adam(args, trainloader, testloader)
    elif args.optimizer == 'rmsprop':
        run_rmsprop(args, trainloader, testloader)
