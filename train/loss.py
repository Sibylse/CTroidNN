import torch.nn as nn
import torch
import torch.nn.functional as F


def gradient_penalty(self, inputs, outputs):
    gradients = torch.autograd.grad(
            outputs=outputs,
            inputs=inputs,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
        )[0]
    gradients = gradients.flatten(start_dim=1)
    # L2 norm
    grad_norm = gradients.norm(2, dim=1)
    # Two sided penalty
    gradient_penalty = ((grad_norm - 1) ** 2).mean()
    return gradient_penalty
    
class CE_Loss:
    def __init__(self, c, device):
        #super(CE_Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        #self.classifier = classifier.to(device)
        self.softmax = nn.Softmax(dim=1)
        self.logits = 0
 
    #def forward(self, inputs, targets):  
    #    self.logits = self.classifier(inputs) # prediction before softmax
    #    return self.ce_loss(self.logits, targets)

    def loss(self, inputs, targets, net):
        logits = net(inputs)
        return self.ce_loss(logits, targets), self.softmax(logits)
    
    def conf(self, inputs, net):
        logits = net(inputs)
        return self.softmax(logits)
    
    def prox(self, net):
        return

class CTLoss():
    def __init__(self, c, device):
        #super(CTLoss, self).__init__()
        #self.I = torch.eye(c).to(device)
        self.ce_loss = nn.CrossEntropyLoss()
        self.nll_loss = nn.NLLLoss()
        #self.classifier = classifier.to(device)
 
    #def forward(self, inputs, targets):        
        #Y = self.I[targets].float().unsqueeze(1) #m x c
    #    logits_views = self.classifier(inputs) # m x d/d_view x c
        #logits_views = Y*logits_views + self.delta*(1-Y)*logits_views
    #    logits = logits_views.transpose(1,2)
    #    targets_rep = targets.repeat(logits.size(2),1).t()
    #    loss = self.ce_loss(logits,targets_rep) 
    #    loss+= self.nll_loss(logits,targets_rep)
    #    return loss

    def loss(self, inputs, targets, net):
        logits_views = net(inputs) # m x d/d_view x c
        logits = logits_views.transpose(1,2)
        targets_rep = targets.repeat(logits.size(2),1).t()
        loss = self.ce_loss(logits,targets_rep) 
        loss+= self.nll_loss(logits,targets_rep)
        return loss, torch.exp(torch.sum(logits_views,1))
    
    def conf(self, inputs, net):
        logits_views = net(inputs)
        return torch.exp(torch.sum(logits_views,1))
    
    def prox(self,net):
        #torch.clamp_(self.delta, self.delta_min, self.delta_max)
        net.classifier.prox()


class BCE_DUQLoss():
    
    def __init__(self, c, device, weight_gp):
        #super(BCE_DUQLoss, self).__init__()
        #self.bce_loss = nn.BCELoss()
        self.I = torch.eye(c).to(device)
        self.weight_gp = weight_gp
        self.embedding = 0
        #self.classifier = classifier.to(device)
        #self.Y_pred = 0 #predicted class confidences
        self.Y= 0
    
    #def forward(self, inputs, targets):
    #    self.Y = self.I[targets].float()
    #    self.Y_pred = torch.exp(self.classifier(inputs))
    #    #loss = self.bce_loss(self.Y_pred, self.Y)
    #    loss = F.binary_cross_entropy(self.Y_pred, self.Y, reduction="mean")
    #    return loss

    def loss(self, inputs, targets, net):
        if self.weight_gp >0:
            inputs.requires_grad_(True)
        self.Y = self.I[targets].float()
        self.embedding = net.embed(inputs)
        Y_pred = torch.exp(net.classifier(self.embedding))
        loss = F.binary_cross_entropy(Y_pred, Y, reduction="mean")
        if self.weight_gp > 0:
            gp = gradient_penalty(inputs, Y_pred)
            loss += weight_gp_pred * gp
        return loss, Y_pred
    
    def conf(self, inputs, net):
        logits = net(inputs)
        return torch.exp(logits)
    
    def prox(self, net):
        net.eval()
        net.classifier.update_centroids(self.embedding, self.Y)
        net.train()
