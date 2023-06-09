'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Gauss_CTroid(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self,in_features,out_features, gamma, gamma_min=0.05,gamma_max=1000):
        super(Gauss_CTroid, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gamma=nn.Parameter(gamma*torch.ones(out_features)) #exp(-gamma_k||D_j.^T - C_.k||^2)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features)) # (cxd) centroids
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def forward(self, D):
        out = D.unsqueeze(2) - self.weight.t().unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        return -self.gamma*torch.sum((out**2),1) # (mxc)
    
    def conf(self,D):
        return torch.exp(self.forward(D))
    
    def prox(self):
        torch.clamp_(self.gamma, self.gamma_min, self.gamma_max)
            
    def get_margins(self):
        #X is dxc, out is cxc matrix, containing the distances ||X_i-X_j||
        # only the upper triangle of out is needed
        X = self.weight.data.t()
        out = X.t().unsqueeze(2) - X.unsqueeze(0) #D is mxd, weight.t() (centroids) is dxc 
        out= torch.sqrt(torch.sum((out**2),1))
        triu_idx = torch.triu_indices(out.shape[0], out.shape[0],1)
        return out[triu_idx[0],triu_idx[1]]
    
class Gauss_DUQ(nn.Module):
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, gamma, N_init=None, m_init=None, alpha=0.999):
        super(Gauss_DUQ, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "gamma", gamma 
        )
        #self.gamma=gamma
        self.alpha=alpha
        if N_init==None:
            N_init = torch.ones(out_features)*10
        if m_init==None:
            m_init = torch.normal(torch.zeros(in_features, out_features), 0.05)
        self.register_buffer("N", N_init) # 
        self.register_buffer(
            "m", m_init # (dxc)
        )
        self.m = self.m * self.N
        self.W = nn.Parameter(torch.zeros(in_features, out_features, in_features)) # (dxcxr) (r=d)
        nn.init.kaiming_normal_(self.W, nonlinearity="relu")

    def forward(self, D):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)
        Z = self.m / self.N.unsqueeze(0) # centroids (dxc)
        out = DW - Z.unsqueeze(0)
        return -self.gamma*torch.mean((out**2),1) # (mxc)
    

    def conf(self,D):
        return torch.exp(self.forward(D))
    
    def update_centroids(self, D, Y):
        DW = torch.einsum("ij,mnj->imn", D, self.W) # (mxdxc)

        # normalizing value per class, assumes y is one_hot encoded
        self.N = self.alpha * self.N + (1 - self.alpha) * Y.sum(0)

        # compute sum of embeddings on class by class basis
        features_sum = torch.einsum("ijk,ik->jk", DW, Y)

        self.m = self.alpha * self.m + (1 - self.alpha) * features_sum

class Gauss_Process(nn.Module): #SNGP final layer
    __constants__ = ['in_features', 'out_features']

    def __init__(self, in_features, out_features, rff_features=1024, ridge_penalty=1.0, rff_scalar=None, mean_field_factor=25):
        super(Gauss_Process, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.ridge_penalty=ridge_penalty
        self.mean_field_factor = mean_field_factor
        
        self.rff = RandomFourierFeatures(in_features, rff_features, rff_scalar)
        self.logit = nn.Linear(rff_features, out_features) #multiply with beta matrix, why is there a bias? Might be a mistake.
        
        precision = torch.eye(rff_features) * self.ridge_penalty
        self.register_buffer("precision", precision)
        self.register_buffer("covariance", torch.eye(rff_features)) #precision is inverse of covariance
        

    def forward(self, D):
        Phi = self.rff(D)
        pred = self.logit(Phi)

        if self.training:
            self.precision += Phi.t() @ Phi
        else: #the covariance has to be updated before by invoking eval()
            with torch.no_grad():
                pred_cov = Phi @ ((self.covariance @ Phi.t()) * self.ridge_penalty)
            if self.mean_field_factor is None:
                return pred, pred_cov
            # Do mean-field approximation as alternative to MC integration of Gaussian-Softmax
            # Based on: https://arxiv.org/abs/2006.07584
            logits_scale = torch.sqrt(1.0 + torch.diag(pred_cov) * self.mean_field_factor)
            if self.mean_field_factor > 0:
                pred = pred / logits_scale.unsqueeze(-1)
        return pred
    
    #def conf(self,D):
    #    return torch.exp(self.forward(D))
    
    def train(self,mode=True):
        if mode: #training is starting (optimizer calls train() each epoch)
            identity = torch.eye(self.precision.shape[0], device=self.precision.device)
            self.precision = identity * self.ridge_penalty
            print("reset precision matrix")
        elif self.training: #switch from training to eval mode
            self.update_covariance()
            print("updated covariance matrix")
        return super().train(mode)
        
    
    def update_covariance(self):
        with torch.no_grad():
            eps = 1e-7  
            jitter = eps * torch.eye(self.precision.shape[1],device=self.precision.device)
            u, info = torch.linalg.cholesky_ex(self.precision + jitter)
            assert (info == 0).all(), "Precision matrix inversion failed!"
            torch.cholesky_inverse(u, out=self.covariance)
        

class RandomFourierFeatures(nn.Module):
    __constants__ = ['in_features', 'rff_features']
    
    def __init__(self, in_features, rff_features, rff_scalar=None):
        super().__init__()
        if rff_scalar is None:
            rff_scalar = math.sqrt(rff_features / 2)

        self.register_buffer("rff_scalar", torch.tensor(rff_scalar))

        if rff_features <= in_features:
            W = self.random_ortho(in_features, rff_features)
        else:
            # generate blocks of orthonormal rows which are not neccesarily orthonormal
            # to each other.
            dim_left = rff_features
            ws = []
            while dim_left > in_features:
                ws.append(self.random_ortho(in_features, in_features))
                dim_left -= in_features
            ws.append(self.random_ortho(in_features, dim_left))
            W = torch.cat(ws, 1)

        # From: https://github.com/google/edward2/blob/d672c93b179bfcc99dd52228492c53d38cf074ba/edward2/tensorflow/initializers.py#L807-L817
        feature_norm = torch.randn(W.shape) ** 2
        W = W * feature_norm.sum(0).sqrt()
        self.register_buffer("W", W)

        b = torch.empty(rff_features).uniform_(0, 2 * math.pi)
        self.register_buffer("b", b)

    def forward(self, x):
        k = torch.cos(x @ self.W + self.b)
        k = k / self.rff_scalar
        return k
    
    def random_ortho(self,n, m):
        q, _ = torch.linalg.qr(torch.randn(n, m))
        return q
  
