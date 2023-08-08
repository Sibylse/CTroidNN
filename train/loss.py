import torch.nn as nn
import torch
import torch.nn.functional as F

class CE_Loss(nn.Module):
    def __init__(self, classifier, c, device):
        super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.logits = 0
 
    def forward(self, inputs, targets):  
        self.logits = self.classifier(inputs) # prediction before softmax
        return self.ce_loss(self.logits, targets)
    
    def conf(self,inputs):
        return self.softmax(self.classifier(inputs))
    
    def prox(self):
        return
        
class CE_CTLoss(nn.Module):
    def __init__(self, classifier, c, device, delta_min = 0.001, delta_max = 0.999):
        super(CE_CTLoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.classifier = classifier.to(device)
        self.delta = nn.Parameter(torch.ones(c)*0.9)
        self.delta_min = delta_min
        self.delta_max = delta_max
 
    def forward(self, inputs, targets):        
        Y = self.I[targets].float()
        logits = self.classifier(inputs)
        loss = self.ce_loss(Y*logits + self.delta*(1-Y)*logits,targets) 
        loss+= self.nll_loss(logits,targets)
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        torch.clamp_(self.delta, self.delta_min, self.delta_max)
        self.classifier.prox()

class CTLoss(nn.Module):
    def __init__(self, classifier, c, device):
        super(CTLoss, self).__init__()
        self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        self.classifier = classifier.to(device)
 
    def forward(self, inputs, targets):        
        #Y = self.I[targets].float().unsqueeze(1) #m x c
        logits_views = self.classifier(inputs) # m x d/d_view x c
        #logits_views = Y*logits_views + self.delta*(1-Y)*logits_views
        logits = logits_views.transpose(1,2)
        targets_rep = targets.repeat(logits.size(2),1).t()
        loss = self.ce_loss(logits,targets_rep) 
        loss+= self.nll_loss(logits,targets_rep)
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        #torch.clamp_(self.delta, self.delta_min, self.delta_max)
        self.classifier.prox()


class BCE_DUQLoss(nn.Module):
    
    def __init__(self, classifier, c, device):
        super(BCE_DUQLoss, self).__init__()
        #self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.classifier = classifier.to(device)
        self.Y_pred = 0 #predicted class confidences
        self.Y= 0
    
    def forward(self, inputs, targets):
        self.Y = self.I[targets].float()
        self.Y_pred = torch.exp(self.classifier(inputs))
        #loss = self.bce_loss(self.Y_pred, self.Y)
        loss = F.binary_cross_entropy(self.Y_pred, self.Y, reduction="mean")
        return loss
    
    def conf(self,inputs):
        return self.classifier.conf(inputs)
    
    def prox(self):
        return
