import torch.nn as nn
import torch

class CE_Loss(nn.Module):
    def __init__(self, classifier, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.Y_pred = 0
 
    def forward(self, inputs, targets):  
        self.Y_pred = self.classifier(inputs) # prediction before softmax
        return self.ce_loss(self.Y_pred, targets)
    
    def conf(self,inputs):
        return self.softmax(self.classifier(inputs))
    
    def prox(self):
        return
        
class CE_GALoss(nn.Module):
    def __init__(self, classifier, c, device, delta_min = 0.001, delta_max = 0.9):
        super(CE_GALoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.classifier = classifier.to(device)
        self.delta = nn.Parameter(torch.ones(c)*0.9)
        self.delta_min = delta_min
        self.delta_max = delta_max
 
    def forward(self, inputs, targets):        
        Y = self.I[targets]
        logits = self.classifier(inputs)
        loss = self.ce_loss(Y*logits + self.delta*(1-Y)*logits,targets) 
        loss+= self.nll_loss(self.delta*logits,targets)
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        torch.clamp_(self.delta, self.delta_min, self.delta_max)
        self.classifier.prox()


class BCE_DUQLoss(nn.Module):
    
    def __init__(self, classifier, c, device):
        super(BCE_DUQLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.classifier = classifier.to(device)
        self.Y_pred = 0 #predicted class probabilities
        self.Y= 0
    
    def forward(self, inputs, targets):
        self.Y = self.I[targets]
        self.Y_pred = torch.exp(self.classifier(inputs))
        loss = self.bce_loss(self.Y_pred, self.Y)
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        return
