from torch import nn
from torch import optim


class Training_Setup:
    def __init__(
            self,
            lr:float,
            momentum:float,
            weight_decay:float,
            gamma:float,
            milestones:list,
            nesterov:bool = False
            ) -> None:
        
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.milestones = milestones
        self.nesterov = nesterov


    def create_training_setup(self, model):
        loss_function = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum = self.momentum, weight_decay=self.weight_decay, nesterov=self.nesterov)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, self.milestones, self.gamma)
        return (loss_function, optimizer, scheduler)
    

    

    

