import torch
from torch import nn

class Cnn_Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128*4*4, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
          
    def forward(self, x):
        return self.layers(x)
    
if __name__ == "__main__":
    model=Cnn_Small()
    x = torch.randn(4, 3, 32, 32)  #4 fake images
    output=model(x)
    print(output.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    for name, p in model.named_parameters():
        print(name, tuple(p.shape))