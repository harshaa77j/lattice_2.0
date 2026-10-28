import torch
from torch import nn

def train_one_epoch(model, loader, optimizer, device):
    model.train()                     
    loss_fn = nn.CrossEntropyLoss()
    total_loss, batches = 0.0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()     
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()         
        optimizer.step()           
        total_loss += loss.item()
        batches += 1
    return total_loss / max(batches, 1)

def evaluate(model, loader, device):
    model.eval()                      
    correct, total = 0, 0
    with torch.no_grad():             
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)