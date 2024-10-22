import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from opacus import PrivacyEngine
from opacus.optimizers import DPOptimizer
from torch.optim import Adam, RMSprop
from resnet import resnet20
import argparse
import csv
import itertools

# Function to parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Differentially Private Training for CIFAR-10 using ResNet20")
    
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size for testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
    parser.add_argument('--noise_multiplier', type=float, default=1.1, help='Noise multiplier for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping')
    parser.add_argument('--optimizer', choices=['sgd', 'sgd_momentum', 'adam', 'rmsprop'], default='sgd', help='Optimizer type')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta for differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=3.0, help='Target epsilon for DP')
    
    args = parser.parse_args()
    return args

# Function to save results into a CSV file
def save_results_to_csv(results, filename='resnet20_tuning_results.csv'):
    keys = results[0].keys()  # Column names
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

# Main training function
def train_model(args, model, optimizer, trainloader, testloader):
    print(f"Training ResNet20 with {args.optimizer.upper()} under DP constraints (ε ≤ {args.target_epsilon}, δ = {args.delta})")

    # Initialize the Privacy Engine
    privacy_engine = PrivacyEngine(
        model,
        sample_rate=args.batch_size / len(trainloader.dataset),  # Correct sample rate based on batch size
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    optimizer = DPOptimizer(
        optimizer=optimizer,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        expected_batch_size=args.batch_size,
    )

    privacy_engine.attach(optimizer)

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
        epsilon, _ = privacy_engine.get_privacy_spent(args.delta)
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

# Perform a grid search for ResNet20
def perform_grid_search(grid_params, trainloader, testloader, csv_filename):
    results = []
    for params in itertools.product(*grid_params.values()):
        args = parse_args()
        param_dict = dict(zip(grid_params.keys(), params))

        # Override parsed args with grid search values
        args.batch_size = param_dict['batch_size']
        args.lr = param_dict['lr']
        args.noise_multiplier = param_dict['noise_multiplier']
        args.max_grad_norm = param_dict['max_grad_norm']
        args.optimizer = param_dict['optimizer']

        # Load the ResNet20 model
        model = resnet20(num_classes=10).cuda()

        # Choose optimizer
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr)
        elif args.optimizer == 'sgd_momentum':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = Adam(model.parameters(), lr=args.lr)
        elif args.optimizer == 'rmsprop':
            optimizer = RMSprop(model.parameters(), lr=args.lr)

        # Train the model and get loss, accuracy, and epsilon
        loss, accuracy, epsilon = train_model(args, model, optimizer, trainloader, testloader)

        # Save results to list
        result = {
            'optimizer': args.optimizer,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'noise_multiplier': args.noise_multiplier,
            'max_grad_norm': args.max_grad_norm,
            'loss': loss,
            'accuracy': accuracy,
            'epsilon': epsilon,
        }
        results.append(result)

    # Save all results to CSV
    save_results_to_csv(results, csv_filename)

if __name__ == "__main__":
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

    # Define the grid of hyperparameters for ResNet20 tuning
    grid_params = {
        'optimizer': ['sgd', 'adam', 'rmsprop'],  # Add optimizers you want to test
        'batch_size': [64, 128],
        'lr': [0.01, 0.001],
        'noise_multiplier': [0.5, 1.1],
        'max_grad_norm': [1.0, 0.5]
    }

    # Perform the grid search and save results to CSV
    csv_filename = 'resnet20_tuning_results_with_epsilon.csv'
    perform_grid_search(grid_params, trainloader, testloader, csv_filename)
