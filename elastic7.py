# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 13:54:46 2024

@author: vrh
"""


import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                     # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split
import numpy as np
import time
from pyDOE import lhs         #Latin Hypercube Sampling
import scipy.io
import numpy as np
import math
import matplotlib.pyplot as plt
from IPython import get_ipython


torch.set_default_dtype(torch.float)
#PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if device == 'cuda': 
    print(torch.cuda.get_device_name()) 
    
lamd = 1
mu = 0.5
Q = 4



class Swish(nn.Module):
		def __init__(self, inplace=True):
			super(Swish, self).__init__()
			self.inplace = inplace

		def forward(self, x):
			return    torch.tanh(x)
    

# 		def __init__(self, inplace=True):
# 			super(Swish, self).__init__()
# 			self.inplace = inplace

# 		def forward(self, x):
# 			if self.inplace:
# 				x.mul_(torch.sigmoid(x))
# 				return x
# 			else:
# 				return x * torch.sigmoid(x) 

class simplePINN(nn.Module):
    def __init__(self):
        super().__init__()
        # self.flatten=nn.Flatten()
        self.sequential=nn.Sequential(
            nn.Linear(in_features=2, out_features=30,bias=True),
            Swish(),
            nn.Linear(in_features=30, out_features=30,bias=True),
            Swish(),
            nn.Linear(in_features=30, out_features=30,bias=True),
            Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            # nn.Linear(in_features=30, out_features=30,bias=True),
            # Swish(),
            nn.Linear(in_features=30, out_features=5,bias=True)
            
            )
      
          
    def forward(self,x):
            output = self.sequential(x)
            return output
        

modelv1=simplePINN().to(device)
def init_normal(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        print(m)
        #nn.init.xavier_normal_(m.weight)


    modelv1.apply(init_normal)

# #%%
def LossPDE(Collo_x,Collo_y, n1=0, n2=0):
    
    
    u_exact, v_exact, sxx_exact, syy_exact, sxy_exact=analytical_soln(lamd, mu, Q, Collo_x,Collo_y)
    x_bodyf,y_bodyf=bodyf(lamd, mu, Q, Collo_x,Collo_y)
    
    x_bodyf=torch.Tensor(Collo_x[:,None] ).to(device)
    y_bodyf=torch.Tensor(Collo_y[:,None] ).to(device)
    
    Collo_x =torch.Tensor(Collo_x[:,None] ).to(device)
    Collo_y =torch.Tensor(Collo_y [:,None]).to(device)
   
    u_exact=torch.Tensor(u_exact[:,None] ).to(device)
    v_exact=torch.Tensor(v_exact[:,None] ).to(device)
    sxx_exact=torch.Tensor(sxx_exact[:,None] ).to(device)
    syy_exact=torch.Tensor(syy_exact[:,None] ).to(device)
    sxy_exact=torch.Tensor(sxy_exact[:,None] ).to(device)
    
    # print(v_exact.shape)
    
    Collo_x.requires_grad = True
    Collo_y.requires_grad = True
    sample1 = torch.cat((Collo_x, Collo_y),  1)
   
    pred = modelv1(sample1)

    u = pred[:,0:1]
    v = pred[:,1:2]
    s11 = pred[:,2:3]
    s22 = pred[:,3:4]
    s12 = pred[:,4:5]
    

    # tractions
   


    s11_1= torch.autograd.grad(s11, Collo_x, grad_outputs=torch.ones_like(s11), create_graph=True, only_inputs=True)[0]
    s12_2= torch.autograd.grad( s12, Collo_y, grad_outputs=torch.ones_like( s12), create_graph=True, only_inputs=True)[0] 
    s22_2= torch.autograd.grad( s22, Collo_y,  grad_outputs=torch.ones_like( s22), create_graph=True, only_inputs=True)[0]
    s12_1= torch.autograd.grad( s12, Collo_x,  grad_outputs=torch.ones_like( s12), create_graph=True, only_inputs=True)[0]  # Plane stress problem
    
   
    # Plane stress problem
    u_x= torch.autograd.grad(u, Collo_x,  grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    u_y= torch.autograd.grad(u, Collo_y,  grad_outputs=torch.ones_like(u), create_graph=True, only_inputs=True)[0]
    
    
    v_x= torch.autograd.grad(v, Collo_x,  grad_outputs=torch.ones_like(v), create_graph=True, only_inputs=True)[0]
    v_y= torch.autograd.grad(v, Collo_y,  grad_outputs=torch.ones_like(v), create_graph=True, only_inputs=True)[0]
    
    e_xx = u_x
    e_yy = v_y
    e_xy = (u_y + v_x)/2.0
    e_kk = e_xx + e_yy
    
    f_s11r = lamd*e_kk + 2*mu*e_xx - s11
    f_s22r = lamd*e_kk + 2*mu*e_yy - s22
    f_s12r = 2*mu*e_xy - s12
    
    f_s1r = s11_1 + s12_2
    f_s2r = s12_1 + s22_2
    loss_f = nn.MSELoss()
    
    loss_f_s11r = loss_f(f_s11r, torch.zeros_like(f_s11r))
    
    loss_f_s22r = loss_f(f_s22r, torch.zeros_like(f_s22r))
    loss_f_s12r = loss_f(f_s12r, torch.zeros_like(f_s12r))
    
    lossf_x=loss_f( f_s1r+x_bodyf, torch.zeros_like(f_s1r))
    lossf_y=loss_f(f_s2r+y_bodyf, torch.zeros_like(f_s2r))
    
    loss_u=loss_f(u_exact,u)
    loss_v=loss_f(v_exact,v)
    loss_sxx=loss_f(lamd*e_kk + 2*mu*e_xx,sxx_exact)
    loss_syy=loss_f(lamd*e_kk + 2*mu*e_yy ,syy_exact)
    loss_sxy=loss_f(2*mu*e_xy,sxy_exact)
    
    
    
    
    return loss_f_s11r, loss_f_s22r, loss_f_s12r,lossf_x,lossf_y,loss_u,loss_v,loss_sxx,loss_syy,loss_sxy
    
def Loss_Dir_BC(x_bottom,y_bottom,tx_bottom,ty_bottom):
     x_bottom =torch.Tensor(x_bottom[:,None] ).to(device)
     y_bottom =torch.Tensor(y_bottom [:,None]).to(device)
     
     x_bottom.requires_grad = True
     y_bottom.requires_grad = True
     sample1 = torch.cat((x_bottom, y_bottom),  dim=1)
     
     pred = modelv1(sample1)
    
     u = pred[:,0:1]
     v = pred[:,1:2]
    
     loss_f = nn.MSELoss()
     loss_Dir_u=loss_f(u, torch.zeros_like(u))
     loss_Dir_v=loss_f(v, torch.zeros_like(v))
     
     return loss_Dir_u,loss_Dir_v
 
def Loss_Neum_BC(x_Neum,y_Neum,tx_Neum,ty_Neum, nx_Neum, ny_Neum):
    
    tx_Neum =torch.Tensor(tx_Neum[:,None] ).to(device)
    ty_Neum =torch.Tensor(ty_Neum [:,None]).to(device)
    x_Neum =torch.Tensor(x_Neum[:,None] ).to(device)
    y_Neum =torch.Tensor(y_Neum [:,None]).to(device)
    nx_Neum =torch.Tensor(nx_Neum[:,None] ).to(device)
    ny_Neum =torch.Tensor(ny_Neum [:,None]).to(device)
    
    
    x_Neum.requires_grad = True
    y_Neum.requires_grad = True
    sample1 = torch.cat((x_Neum, y_Neum),  dim=1)
   
    pred = modelv1(sample1)

    u = pred[:,0:1]
    v = pred[:,1:2]
    s11 = pred[:,2:3]
    s22 = pred[:,3:4]
    s12 = pred[:,4:5]
    # u, v, s11, s22, s12 = pred.T

    # tractions
    
  # tractions
    tx = torch.mul(s11,nx_Neum)+torch.mul(s12,ny_Neum)
    ty = torch.mul(s12,nx_Neum)+torch.mul(s22,ny_Neum)
    
    
    loss_f = nn.MSELoss()
    loss_tx = loss_f(tx, tx_Neum)
    loss_ty = loss_f(ty, ty_Neum)
    
    # loss_f_s22r = loss_f(f_s22r, torch.zeros_like(f_s22r))
    # loss_f_s12r = loss_f(f_s12r, torch.zeros_like(f_s12r))
    # lossf_x=loss_f(f_s1r+x_bodyf, torch.zeros_like(f_s1r))
    # lossf_y=loss_f(f_s2r+y_bodyf, torch.zeros_like(f_s2r))
    
    return loss_tx , loss_ty



 
    
 
    


#%%
# number of domain training samples
num_dom_train_samples =80
# number of boundary training samples
num_b_train_samples = 40

# collocation points in the domain
XY_c1 = np.linspace(0.0, 1.0, num=num_dom_train_samples)               # x = 0 ~ +1
XY_c2 = np.linspace(0.0, 1.0, num=num_dom_train_samples)               # y = 0 ~ +1
Collo_x, Collo_y = np.meshgrid(XY_c1, XY_c2)
Collo_x, Collo_y = ((Collo_x.flatten(),Collo_y.flatten()))




def Boundary_Condition1(num_b_train_samples):   # left
    #Boundary Condition x = 0 and 0 =< y =<1
    
    x_left = np.zeros(num_b_train_samples)                               # x = 0
    y_left= np.linspace(0, 1, num=num_b_train_samples)                   # y = 0 ~ +1
    nx_left = -1.*np.ones(num_b_train_samples)
    ny_left=np.zeros(num_b_train_samples)
    tx_left=np.zeros(num_b_train_samples)
    ty_left=-mu*np.pi*(np.cos(np.pi*y_left)+np.power(y_left,4)*Q/4)
    return x_left,y_left,nx_left,ny_left,tx_left,ty_left


def Boundary_Condition2(num_b_train_samples): #top   
    #Boundary Condition 0 =< x =<1 and y=1
    
    x_top = np.linspace(0.0, 1.0, num=num_b_train_samples)                               
    y_top=  np.ones(num_b_train_samples)                    
    nx_top = np.zeros(num_b_train_samples)
    ny_top=np.ones(num_b_train_samples)
    tx_top=mu*np.pi*(-np.cos(2*np.pi*x_top)+np.cos(np.pi*y_top)*Q/4)
    ty_top=(lamd+2*mu)*Q*np.sin(np.pi*x_top)
    return x_top,y_top,nx_top,ny_top,tx_top,ty_top    


def Boundary_Condition3(num_b_train_samples):  #right  
    #Boundary Condition x=1 and 0<y<1
    
    x_right =  np.ones(num_b_train_samples)                               
    y_right=  np.linspace(0.0, 1.0, num=num_b_train_samples)                    
    nx_right = np.ones(num_b_train_samples)
    ny_right=np.zeros(num_b_train_samples)
    tx_right=np.zeros(num_b_train_samples)
    ty_right=mu*np.pi*(np.cos(np.pi*y_right)-np.power(y_right,4)*Q/4)
    return x_right,y_right,nx_right,ny_right,tx_right,ty_right 

def Boundary_Condition4(num_b_train_samples):    #bottom
    #Boundary Condition 0<x<1 and y=0
    
    x_bottom=  np.linspace(0.0, 1.0, num=num_b_train_samples)                              
    y_bottom=  np.zeros(num_b_train_samples)                     

    tx_bottom=np.zeros(num_b_train_samples)
    ty_bottom=np.zeros(num_b_train_samples)
    return x_bottom,y_bottom,tx_bottom,ty_bottom 

def bodyf(lamd, mu, Q, x, y):
    # body force
    Pi = np.pi
    b1=lamd*(4*Pi**2*np.multiply(np.cos(2*Pi*x),np.sin(Pi*y))-Q*Pi*np.multiply(np.cos(Pi*x),np.power(y,3)))\
        +mu*(9.*Pi**2*np.multiply(np.cos(2*Pi*x),np.sin(Pi*y))-Q*Pi*np.multiply(np.cos(Pi*x),np.power(y,3)))
    b2=lamd*(-3*Q*np.multiply(np.sin(Pi*x),np.power(y,2))+2*Pi**2.*np.multiply(np.sin(2*Pi*x),np.cos(Pi*y)))\
        +mu*(-6.*Q*np.multiply(np.sin(Pi*x),np.power(y,2))+2*Pi**2.*np.multiply(np.sin(2*Pi*x),np.cos(Pi*y))\
        +Q*Pi**2.*np.multiply(np.sin(Pi*x),np.power(y,4))/4.)
    return b1, b2    

def analytical_soln(lamd, mu, Q, xx, yy):
    # analytical solution
    Pi = np.pi
    u = np.multiply(np.cos(2*Pi*xx),np.sin(Pi*yy))
    v = Q*np.multiply(np.sin(Pi*xx),np.power(yy,4))/4
    sxx = lamd*(Q*np.multiply(np.sin(np.pi*xx),np.power(yy,3))-2*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy)))\
        -4*mu*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy))
    syy = lamd*(Q*np.multiply(np.sin(np.pi*xx),np.power(yy,3))-2*Pi*np.multiply(np.sin(2*Pi*xx),np.sin(Pi*yy)))\
        +2*mu*Q*np.multiply(np.sin(Pi*xx),np.power(yy,3))
    sxy = mu*(np.multiply(np.cos(np.pi*xx),np.power(yy,4))*Pi*Q/4+Pi*np.multiply(np.cos(2*Pi*xx),np.cos(Pi*yy)))

    return u, v, sxx, syy, sxy


x_left,y_left,nx_left,ny_left,tx_left,ty_left=Boundary_Condition1(num_b_train_samples)
x_top,y_top,nx_top,ny_top,tx_top,ty_top=Boundary_Condition2(num_b_train_samples)
x_right,y_right,nx_right,ny_right,tx_right,ty_right =Boundary_Condition3(num_b_train_samples)
x_bottom,y_bottom,tx_bottom,ty_bottom=Boundary_Condition4(num_b_train_samples)

x_Neum=np.hstack((x_left,x_top,x_right))
y_Neum=np.hstack((y_left,y_top,y_right))

tx_Neum=np.hstack((tx_left,tx_top,tx_right))
ty_Neum=np.hstack((ty_left,ty_top,ty_right))

nx_Neum=np.hstack((nx_left,nx_top,nx_right))
ny_Neum=np.hstack((ny_left,ny_top,ny_right))



epochs=20000
learning_rate=10**-2

optimizer2 = torch.optim.LBFGS(modelv1.parameters(), lr=1,  history_size=100, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32)


optimizer1 = optim.AdamW(modelv1.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-40)
#optimizer1 = torch.optim.Adam(modelv1.parameters(), lr=0.001,betas=(0.9, 0.999), eps=1e-20, weight_decay=0.001, amsgrad=False)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
# max_iter = 16600
def train(optimizer1,optimizer2):
    for epoch in range(epochs):
      #for batch_idx, (x, t) in enumerate(dataloader):
            def closure():
                if epoch < 4600:
                    
                    optimizer1.zero_grad()
                else:
                    optimizer2.zero_grad()
    
                loss_f_s11r, loss_f_s22r, loss_f_s12r,lossf_x,lossf_y,loss_u,loss_v,loss_sxx,loss_syy,loss_sxy  = LossPDE(Collo_x,Collo_y, nx_Neum, ny_Neum)
                
                loss_gov = loss_f_s11r+ loss_f_s22r+ loss_f_s12r+ lossf_x+lossf_y +loss_u+loss_v+loss_sxx+loss_syy+loss_sxy
                loss_Dir_u,loss_Dir_v=Loss_Dir_BC(x_bottom,y_bottom,tx_bottom,ty_bottom)
                loss_Dir=loss_Dir_u+loss_Dir_v
                loss_tx , loss_ty=Loss_Neum_BC(x_Neum,y_Neum,tx_Neum,ty_Neum, nx_Neum, ny_Neum) 
                loss_Neum= loss_ty+loss_tx 
                loss= loss_gov+loss_Dir
                loss.backward()
                # losses.append(loss.item())
                # bc_losses.append(loss_BC.item())
                # ic_losses.append(loss_IC.item())
                return loss
    
            # Update parameters W, B
            if epoch < 4600:
                optimizer1.step(closure)
            else:
                optimizer2.step(closure)
    
            # Reset gradients
            modelv1.zero_grad()
    
        # Print loss information every 100 epochs
            if epoch % 100 == 0:
                loss_f_s11r, loss_f_s22r, loss_f_s12r,lossf_x,lossf_y,loss_u,loss_v,loss_sxx,loss_syy,loss_sxy  = LossPDE(Collo_x,Collo_y, nx_Neum, ny_Neum)
                
                loss_gov = loss_f_s11r+ loss_f_s22r+ loss_f_s12r lossf_x+lossf_y +loss_u+loss_v+loss_sxx+loss_syy+loss_sxy
                loss_Dir_u,loss_Dir_v=Loss_Dir_BC(x_bottom,y_bottom,tx_bottom,ty_bottom)
                loss_Dir=loss_Dir_u+loss_Dir_v
                loss_tx , loss_ty=Loss_Neum_BC(x_Neum,y_Neum,tx_Neum,ty_Neum, nx_Neum, ny_Neum) 
                loss_Neum= loss_ty+loss_tx 
                loss=loss_gov+loss_Dir+ loss_Neum
               
                print('Train Epoch: {} \tLoss: {:.10f} '.format(epoch, loss.item()))



train(optimizer1,optimizer2)




