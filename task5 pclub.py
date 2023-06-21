#!/usr/bin/env python
# coding: utf-8

# In[133]:


class MultiLayerPer:
    
    def __init__(self, sizes, n_layers,  n_input, n_output,  activation, d_activation, opt,momentum, seed = 123):
        #super().__init__()
        self.sizes = sizes
        self.n_layers = n_layers
        self.n_input = n_input
        self.n_output = n_output
        self.activation = activation
        self.d_activation = d_activation
        self.weights_ = list()
        self.biases_ = list()
        self.weights_V = list()
        self.biases_V = list()
        self.opt = opt
        self.momentum =  momentum
        
        
        self.m_w = list()
        self.v_w = list()
        self.m_b = list()
        self.v_b = list()
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        
        w_temp = np.random.RandomState(seed)
        self.weights_.append(w_temp.normal(loc=0.0, scale=0.1, size=(sizes[0], n_input)))
        self.biases_.append(np.zeros(sizes[0]))
        self.weights_V.append(np.zeros((sizes[0], n_input)))
        self.biases_V.append(np.zeros(sizes[0]))
        
        self.m_w.append(np.zeros((sizes[0], n_input)))
        self.v_w.append(np.zeros((sizes[0], n_input)))
        self.m_b.append(np.zeros(sizes[0]))
        self.v_b.append(np.zeros(sizes[0]))
        
        
        
        for i in range (0,n_layers-1):
            self.weights_.append(w_temp.normal(loc=0.0, scale=0.1, size=(sizes[i+1], sizes[i])))
            self.biases_.append(np.zeros(sizes[i+1]))
            self.weights_V.append(np.zeros((sizes[i+1], sizes[i])))
            self.biases_V.append(np.zeros(sizes[i+1]))
            
            self.m_w.append(np.zeros((sizes[i+1], sizes[i])))
            self.v_w.append(np.zeros((sizes[i+1], sizes[i])))
            self.v_b.append(np.zeros(sizes[i+1]))
            self.m_b.append(np.zeros(sizes[i+1]))
            
            
        
        self.weights_.append(w_temp.normal(loc=0.0, scale=0.1, size=(n_output, sizes[n_layers-1])))
        self.biases_.append(np.zeros(n_output))
        self.weights_V.append(np.zeros((n_output, sizes[n_layers-1])))
        self.biases_V.append(np.zeros(n_output))
        
        self.m_w.append(np.zeros((n_output, sizes[n_layers-1])))
        self.m_b.append(np.zeros(n_output))
        self.v_w.append(np.zeros((n_output, sizes[n_layers-1])))
        self.v_b.append(np.zeros(n_output))
        
        
        
    def forwardprop(self, X_train):
        x = X_train
        output_layer = list()
        activation_layer = list()
        for i in range (0,len(self.weights_)):
            output_layer.append(np.dot(x, self.weights_[i].T) + self.biases_[i])
            activation_layer.append(self.activation(output_layer[i]))
            
            x = activation_layer[i]
            
        return output_layer, activation_layer
                                    
    def backprop(self, X_train, y_train):
        self.t+=1
        output_layer, activation_layer = self.forwardprop(X_train)
        y_onehot = int_to_onehot(y_train, self.n_output)
        
        grad_w = list()
        grad_b = list()
        loss_output = 2.*(activation_layer[-1] - y_onehot) / y_train.shape[0]
        delta_out = loss_output * self.d_activation(activation_layer[-1])
        
        if(self.opt == 'adam'):
               
                
                dW = (np.dot(delta_out.T, activation_layer[-2]))
                db = (np.sum(delta_out, axis=0))
                self.m_w[-1] = self.beta1 * self.m_w[-1] + (1 - self.beta1) * dW
                self.m_b[-1] = self.beta1 * self.m_b[-1] + (1 - self.beta1) * db
                self.v_w[-1] = self.beta2 * self.v_w[-1] + (1 - self.beta2) * (dW**2)
                self.v_b[-1] = self.beta2 * self.v_b[-1] + (1 - self.beta2) * (db**2)
                
                m_W = self.m_w[-1] / (1 - self.beta1 ** self.t)
                m_b = self.m_b[-1] / (1 - self.beta1 ** self.t)
                
                v_W = self.v_w[-1] / (1 - self.beta2 ** self.t)
                v_b = self.v_b[-1] / (1 - self.beta2 ** self.t)
                
                grad_w.append(m_W / (np.sqrt(v_W) + self.epsilon))
                grad_b.append(m_b / (np.sqrt(v_b) + self.epsilon))
        else :                            
                grad_w.append(np.dot(delta_out.T, activation_layer[-2]))
                grad_b.append(np.sum(delta_out, axis=0))
        
        for i in range (1,len(output_layer)-1):
             
            if(self.opt=='nag'):
                lookahead_W = self.weights_[-i] + self.momentum * self.weights_V[-i]
                d_loss = np.dot(delta_out, lookahead_W)
            else:
                d_loss = np.dot(delta_out, self.weights_[-i])
 
          
            d_ah = self.d_activation(activation_layer[-i-1]) # sigmoid derivative
   
            d_z = activation_layer[-i-2]
            
            delta_out = d_loss*d_ah
            
            if(self.opt == 'adam'):
               
                
                dW = np.dot((d_loss * d_ah).T, d_z)
                db = np.sum((d_loss * d_ah), axis=0)
                self.m_w[-i-1] = self.beta1 * self.m_w[-i-1] + (1 - self.beta1) * dW
                self.m_b[-i-1] = self.beta1 * self.m_b[-i-1] + (1 - self.beta1) * db
                self.v_w[-i-1] = self.beta2 * self.v_w[-i-1] + (1 - self.beta2) * (dW**2)
                self.v_b[-i-1] = self.beta2 * self.v_b[-i-1] + (1 - self.beta2) * (db**2)
                
                m_W = self.m_w[-i-1] / (1 - self.beta1 ** self.t)
                m_b = self.m_b[-i-1] / (1 - self.beta1 ** self.t)
                
                v_W = self.v_w[-i-1] / (1 - self.beta2 ** self.t)
                v_b = self.v_b[-i-1] / (1 - self.beta2 ** self.t)
                
                grad_w.append(m_W / (np.sqrt(v_W) + self.epsilon))
                grad_b.append(m_b / (np.sqrt(v_b) + self.epsilon))
           
            else:
                
                grad_w.append(np.dot((d_loss * d_ah).T, d_z))
                grad_b.append(np.sum((d_loss * d_ah), axis=0))
            
        
        if(self.opt=='nag'):
                lookahead_W = self.weights_[-self.n_layers] + self.momentum * self.weights_V[-self.n_layers]
                d_loss = np.dot(delta_out, lookahead_W)
        else:
                d_loss = np.dot(delta_out, self.weights_[-self.n_layers])
 
  
        d_ah = self.d_activation(activation_layer[-self.n_layers-1]) # sigmoid derivative
  
        d_z = X_train
    
        if(self.opt == 'adam'):
               
                
                dW = np.dot((d_loss * d_ah).T, d_z)
                db = np.sum((d_loss * d_ah), axis=0)
                self.m_w[-self.n_layers-1] = self.beta1 * self.m_w[-self.n_layers-1] + (1 - self.beta1) * dW
                self.m_b[-self.n_layers-1] = self.beta1 * self.m_b[-self.n_layers-1] + (1 - self.beta1) * db
                self.v_w[-self.n_layers-1] = self.beta2 * self.v_w[-self.n_layers-1] + (1 - self.beta2) * (dW**2)
                self.v_b[-self.n_layers-1] = self.beta2 * self.v_b[-self.n_layers-1] + (1 - self.beta2) * (db**2)
                
                m_W = self.m_w[-self.n_layers-1] / (1 - self.beta1 ** self.t)
                m_b = self.m_b[-self.n_layers-1] / (1 - self.beta1 ** self.t)
                
                v_W = self.v_w[-self.n_layers-1] / (1 - self.beta2 ** self.t)
                v_b = self.v_b[-self.n_layers-1] / (1 - self.beta2 ** self.t)
                
                grad_w.append(m_W / (np.sqrt(v_W) + self.epsilon))
                grad_b.append(m_b / (np.sqrt(v_b) + self.epsilon))
 
        else : 
                grad_w.append(np.dot((d_loss * d_ah).T, d_z))
                grad_b.append(np.sum((d_loss * d_ah), axis=0))
        
        return grad_w, grad_b
        
     
   
        
        
            
        
            
            
        
        
        
        
        
        
        
        
            
        


# In[137]:


def minibatch_generator(X, y, minibatch_size):
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
            batch_idx = indices[start_idx:start_idx + minibatch_size]
            yield X[batch_idx], y[batch_idx]

def compute_loss_and_acc(model,X_train,y_train,  minibatch_size, p, num_labels = 10):
    mse, correct_pred, num_examples = 0.,0,0
    minibatch_gen = minibatch_generator(X_train,y_train, minibatch_size)
    
    net_loss = 0
    for i, (features, targets) in enumerate(minibatch_gen):
        __ , outputs = model.forwardprop(features)
        predicted_output = np.argmax(outputs[-1],axis=1)
        onehot_target = int_to_onehot(targets, num_labels=num_labels)
        if(p==0):
            loss = np.mean((onehot_target - outputs[-1])**2)
        else :
            outputs[-1] = softMax(outputs[-1])
            loss = -np.mean(onehot_target*(np.log(outputs[-1])))
        
        correct_pred += (predicted_output==targets).sum()
        num_examples +=targets.shape[0]
        net_loss += loss
    net_loss = net_loss/i
    acc = correct_pred/num_examples
    return net_loss,acc

def train(model, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, minibatch_size, anneal,loss,momentum,opt):
    if(loss == 'sq'):
        p1 = 0
    else :
        p1 = 1
    
    epoch_loss = []
    epoch_valid_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    e = 0
    while e<num_epochs:
    # iterate over minibatches
        minibatch_gen = minibatch_generator(X_train, y_train, minibatch_size)
        for X_train_mini, y_train_mini in minibatch_gen:
        #### Compute outputs ####
       
        #### Compute gradients ####
            grad_w, grad_b = model.backprop(X_train_mini , y_train_mini)

        #### Update weights ####
            for i in range(len(model.weights_)):
                if(opt == 'adam'):
                    model.weights_[-i-1] -=learning_rate * grad_w[i]
                    model.biases_[-i-1] -= learning_rate * grad_b[i]
                    
                else:
                    
                
                    model.weights_V[-i-1] = momentum * model.weights_V[-i-1] -learning_rate * grad_w[i]
                    model.biases_V[-i-1] = momentum * model.biases_V[-i-1] - learning_rate * grad_b[i]
                    model.weights_[-i-1] += model.weights_V[-i-1]
                    model.biases_[-i-1] +=model.biases_V[-i-1]
                
                
                

        ### Epoch Logging ####
        train_loss, train_acc = compute_loss_and_acc( model, X_train, y_train, minibatch_size, p = p1)
        valid_loss, valid_acc = compute_loss_and_acc( model, X_valid, y_valid, minibatch_size, p = p1)
        train_acc, valid_acc = train_acc*100, valid_acc*100
        train_error = 100 - train_acc
        valid_error = 100 -valid_acc
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_loss)
        epoch_valid_loss.append(valid_loss)
        print(f'Epoch: {e+1:03d} 'f'| Loss: {valid_loss:.2f} 'f'| Error: {valid_error:.2f}% ' f'| lr: {learning_rate:.4f}')
        if(anneal == True):
            if(e>0):
                if(valid_loss < epoch_valid_loss[-2]):
                    learning_rate = learning_rate/2
                    continue
        e = e + 1
       
    return epoch_loss, epoch_train_acc, epoch_valid_acc, epoch_valid_loss

def int_to_onehot(y, num_labels):
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
         ary[i, val] = 1
    return ary

def sigmoid(z):
    return 1. / (1. + np.exp(-z))

def d_sigmoid(z):
    return z*(1.-z)

def tanh(z):
    return np.tanh(z)

def dTanh(z):
    return 1/(np.cosh(z)**2)

def softMax(X):
    e = np.exp(X)
    p = e/np.sum(e, axis=0)
    return p


# In[138]:


import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import cv2
import random as rd

images = glob.glob('D:/cifar-10/train/*.png')
train_labels = pd.read_csv('D:/cifar-10/trainLabels.csv')


# In[4]:


LABELS = ['frog', 'truck', 'deer', 'automobile', 'bird', 'horse', 'ship', 'cat', 'dog', 'airplane']
X_train = []
y_train = []

X_val = []
y_val = []


# In[5]:


for img in images:
    prob = rd.random()
    label = train_labels.iloc[int(img[18:-4])-1]['label']
    img_arr = cv2.imread(img)
    img_arr = cv2.resize(img_arr, (32, 32))
    if prob > 0.8:
        X_val.append(list(img_arr))
        y_val.append(LABELS.index(label))
    else:
        X_train.append(list(img_arr))
        y_train.append(LABELS.index(label))


# In[6]:


X_train = np.array(X_train, dtype=np.float32) / 255

y_train = np.array(y_train)

X_val = np.array(X_val, dtype=np.float32) / 255
y_val = np.array(y_val)


# In[7]:


X_train1 = X_train.reshape(X_train.shape[0],-1)
X_train1.shape
X_val1 = X_val.reshape(X_val.shape[0],-1)


# In[128]:



if(args.activation == 'tanh'):
    derivative = dTanh
    activation = tanh
    
if(args.activation == 'sigmoid'):
    derivative = d_sigmoid
    activation = sigmoid

model = MultiLayerPer(args.sizes,args.num_hidden,3072,10, activation, derivative, args.opt, args.momentum)


# In[134]:


epoch_loss, epoch_train_acc, epoch_valid_acc, epoch_valid_loss = train(model, X_train1, y_train, X_val1, y_val, 50 ,args.lr,args.minibatch_size,args.anneal, args.opt,args.momentum,args.opt)


# In[136]:


import argparse

def comma_separated_list(string):
    
    items = [item.strip() for item in string.split(',')]
    return items
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=int, required=True)
parser.add_argument('--momentum', type=int, required=True)
parser.add_argument('--num_hidden', type=int, required=True)
parser.add_argument('--sizes', type=comma_separated_list, required=True)
parser.add_argument('--activation', type=str, required=True)
parser.add_argument('--loss', type=str, required=True)
parser.add_argument('--opt', type=str, required=True)
parser.add_argument('--batch_size', type=str, required=True)
parser.add_argument('--anneal', type=bool, required=True)
parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--expt_dir', type=str, required=True)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--test', type=str, required=True)
args = parser.parse_args()


# In[ ]:




