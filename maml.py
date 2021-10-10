#implementation of MAML for updating coefficients of Label Shift task.

import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

# torch.manual_seed(0)
torch.set_default_dtype(torch.double) #bug fix - float matmul

#enable cuda if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MAML():
    """ 
    Implementation of Model-Agnostic Meta Learning algorithm for performing meta-gradient update
    Label Shift weights.
    """
    def __init__(self, X, y, model, weights, alpha:float=0.01, beta:float=0.05):
        """ 
        Initialize params.
        @Params:
        - X : (torch.tensor) validation data
        - y : (torch.tensor) validation labels
        - model : (Network) DNN
        - weights :  (array) label shift weights
        """
        #store model
        self.f = model.double()
        #define number of classes
        self.cls = len(np.unique(X.numpy()))
        #define parameters, theta
        self.theta = Variable(torch.DoubleTensor(weights), requires_grad=True).to(device)
        #define single task
        self.tasks = [X] #use batches for multi-task setting
        self.y = y.double()
        #define MAML hyperparameters
        self.alpha = alpha
        self.beta = beta
        #define loss and optimizer
        self.criteon = nn.MSELoss() #weight=self.theta
        self.meta_optim = optim.SGD([self.theta], lr=self.beta)
        
    def update(self, max_norm=1.0):
        """ Run a single iteration of MAML algorithm """
        
        theta_prime = []

        for i, batch in enumerate(self.tasks):
            y_hat = self.constraint(self.theta, self.f(batch)) # gather predictions to single dimension
            loss = self.criteon( y_hat, self.y )
            #compute gradients
            grad = torch.autograd.grad(loss, self.theta)
            #update params
            theta_prime.append( self.theta - self.alpha * grad[0] )

        del loss

        #perform meta-update
        m_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(len(self.tasks)):
            theta = theta_prime[i]
            batch = self.tasks[i]
            y_hat = self.constraint(theta, self.f(batch)) # gather predictions to single dimension
            m_loss = m_loss + self.criteon( y_hat, self.y ) # updating meta-loss
 
        #zero gradient before running backward pass
        self.meta_optim.zero_grad()

        #backward pass
        m_loss.backward(retain_graph=True)

        #clip gradients
        nn.utils.clip_grad_norm_([self.theta], max_norm)
        
        #one-step gradient descent
        self.meta_optim.step()
    
    def constraint(self, theta, labels):
        """ Compute dot product of X and parameters theta """
        # N = batch size ; K = batch size
        y = labels.to(device) # K x N
        dot = torch.matmul( y, theta ) # (K x N) • (N x 1) --> (K x 1)
        dot.requires_grad_() #bug fix to retain computational graph
        return dot.to(device)
    
    def get_label_weights(self):
        weights = self.theta.detach().numpy()
        return weights