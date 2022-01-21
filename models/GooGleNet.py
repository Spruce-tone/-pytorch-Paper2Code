from types import SimpleNamespace
from matplotlib.pyplot import MaxNLocator
from torch import batch_norm, nn, relu
import torch

class InceptionBlock(nn.Module):
    def __init__(self, c_in,
    c_red: dict,
    c_out: dict):
        super().__init__()
        
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(c_in, c_out['1x1'], kernel_size=1),
            nn.BatchNorm2d(c_out['1x1']),
            nn.ReLU(),
        )

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(c_in, c_red['3x3'], kernel_size=1),
            nn.BatchNorm2d(c_red['3x3']),
            nn.ReLU(),
            nn.Conv2d(c_red['3x3'], c_out['3x3'], kernel_size=3, padding=1),
            nn.BatchNorm2d(c_out['3x3']),
            nn.ReLU(),
            )

        self.conv_5x5 = nn.Sequential(
            nn.Conv2d(c_in, c_red['5x5'], kernel_size=1),
            nn.BatchNorm2d(c_red['5x5']),
            nn.ReLU(),
            nn.Conv2d(c_red['5x5'], c_out['5x5'], kernel_size=5, padding=2),
            nn.BatchNorm2d(c_out['5x5']),
            nn.ReLU(),
            )
        
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(c_in, c_out['max'], kernel_size=1),
            nn.BatchNorm2d(c_out['max']),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_1x1 = self.conv_1x1(x)
        x_3x3 = self.conv_3x3(x)
        x_5x5 = self.conv_5x5(x)
        x_max_pool = self.max_pool(x)
        out = torch.cat([x_1x1, x_3x3, x_5x5, x_max_pool], dim=1)
        return out

class GoogleNet(nn.Module):
    def __init__(self, c_in, num_classes, base_dim=64):
        super().__init__()

        self.hparams = SimpleNamespace(
            c_in=c_in,
            num_classes=num_classes,
            base_dim=base_dim
        )
        self._network()
        self._initialize_weights()
    
    def _network(self):
        self.input = nn.Sequential(
            nn.Conv2d(self.hparams.c_in, self.hparams.base_dim, kernel_size=7, padding=3, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(self.hparams.base_dim, self.hparams.base_dim*3, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception3 = nn.Sequential(
            InceptionBlock(
                c_in=self.hparams.base_dim*3, 
                c_red={'3x3' : 96, '5x5' : 16}, 
                c_out={'1x1' : 64, '3x3' : 128, '5x5' : 32, 'max' : 32}),
            InceptionBlock(
                c_in=self.hparams.base_dim*4, 
                c_red={'3x3' : 128, '5x5' : 32}, 
                c_out={'1x1' : 128, '3x3' : 192, '5x5' : 96, 'max' : 64}),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
        )

        self.inception4 = nn.Sequential(
            InceptionBlock(
                c_in=480, 
                c_red={'3x3' : 96, '5x5' : 16}, 
                c_out={'1x1' : 192, '3x3' : 208, '5x5' : 48, 'max' : 64}),
            InceptionBlock(
                c_in=512, 
                c_red={'3x3' : 112, '5x5' : 24}, 
                c_out={'1x1' : 160, '3x3' : 224, '5x5' : 64, 'max' : 64}),
            InceptionBlock(
                c_in=512, 
                c_red={'3x3' : 128, '5x5' : 24}, 
                c_out={'1x1' : 128, '3x3' : 256, '5x5' : 64, 'max' : 64}),
            InceptionBlock(
                c_in=512, 
                c_red={'3x3' : 144, '5x5' : 32}, 
                c_out={'1x1' : 112, '3x3' : 288, '5x5' : 64, 'max' : 64}),
            InceptionBlock(
                c_in=528, 
                c_red={'3x3' : 160, '5x5' : 32}, 
                c_out={'1x1' : 256, '3x3' : 320, '5x5' : 128, 'max' : 128}),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1), 
        )

        self.inception5 = nn.Sequential(
            InceptionBlock(
                c_in=832, 
                c_red={'3x3' : 160, '5x5' : 32}, 
                c_out={'1x1' : 256, '3x3' : 320, '5x5' : 128, 'max' : 128}),
            InceptionBlock(
                c_in=832, 
                c_red={'3x3' : 192, '5x5' : 48}, 
                c_out={'1x1' : 384, '3x3' : 384, '5x5' : 128, 'max' : 128}),
            nn.AvgPool2d(kernel_size=7, stride=1, padding=0), 
        )

        self.out = nn.Sequential(
            nn.Dropout2d(0.4),
            nn.Flatten(),
            nn.Linear(1024, self.hparams.num_classes),
        )
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.out(x)
        return x