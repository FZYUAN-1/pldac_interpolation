#%%
import numpy as np 
import matplotlib.pyplot as plt
import dataSource as ds
import plotly.graph_objects as go
from plotly.offline import iplot,plot
import torch
import pandas as pd
mydevice = torch.device("cpu")

#%% [markdown]
## Parametrage
#%%
from sklearn.model_selection import train_test_split

df = ds.importData()


batch_size, train_size, step = 2, 0.5, 200
Len = 30
X_train, X_test, y_train, y_test = ds.create_data_xy_truncated(df, train_size, step, step, nb_truncated=Len)


#%%

plt.scatter(X_train[:, :,0], X_train[:,:, 1], s = 1, c = 'y')
#plt.scatter(y_test[:, :,0], y_test[:,:, 1], s = 1, c = 'y')
plt.show()


#%%
#dataset
from torch.utils.data import Dataset

class timeseries(Dataset):
    def __init__(self,x,y):
        
        self.x = torch.tensor(x, dtype = torch.float32).to(mydevice)
        self.y = torch.tensor(y, dtype = torch.float32).to(mydevice)
        self.len = x.shape[0]

    def __getitem__(self,idx):
        return self.x[idx],self.y[idx]
  
    def __len__(self):
        return self.len


dataset = timeseries(X_train,y_train)
test_dataset = timeseries(X_test, y_test)

#dataloader
from torch.utils.data import DataLoader 
train_loader = DataLoader(dataset,shuffle=True,batch_size=batch_size, drop_last = True)

# verify train_loader, especially check if batches are all consecutives
'''
torch.set_printoptions(sci_mode=False)
l = list(train_loader)
print(l[0], l[1])
'''

#%%
# rnn
#neural network
from torch import nn

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(Model, self).__init__()

        # Defining some parameters
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        #Defining the layers
        # RNN Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, state = None):
        """As we divided data into batches, to respect time order, every forward should take previous state and, produces a new state"""

        #Initializing hidden state for first input using method defined below
        #hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        new_state = None
        batch_size = x.shape[0]
        if state:
            out, new_state = self.lstm(x, state)    
        else:
            out, _ = self.lstm(x)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim) # (batch*len, hidden_dim)

        out = self.fc(out)
        
        if new_state:
            return out.reshape(batch_size, -1, 2), new_state
        else:
            return out.reshape(batch_size, -1, 2)
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = ( 
            torch.zeros((self.n_layers, batch_size, self.hidden_dim), device=mydevice), 
            torch.zeros((self.n_layers, batch_size, self.hidden_dim), device=mydevice) )
        return hidden

def grad_clipping(net, theta): 
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

#%%
# Instantiate the model with hyperparameters

def run_train(input_size, output_size, lr, num_hiddens, num_layer, model, criterion, optimizer, epochs):

    l_loss = []
    for i in range(epochs):
        state = model.init_hidden(batch_size) # init state, dim (n_layers, batch_size, hidden_dim)
        for j, data in enumerate(train_loader):
            #print(state)
            y_pred = model(data[0])
            #print(y_pred.shape, data[0].shape, data[1].shape)
            loss = criterion(y_pred,data[1])
            
            # detach hidden state from graph to prevent repeatitif calculation
            #for s in state:
            #    s.detach_()

            optimizer.zero_grad()
            loss.backward() 
            #grad_clipping(model, 1)
            optimizer.step()
        if i%100 == 0:
            l_loss.append(loss)
            print(i,"th iteration : ",loss)

    plt.plot(range(0, epochs, 100), l_loss)
    plt.legend('LSTM Loss')
    plt.show()
    #print([e.cpu().detach() for e in l_loss])
    return np.mean([e.cpu().detach() for e in l_loss])

#%%
# test set actual vs predicted

def noX_pred(input, num_future, model):
    """
    take last output as next input; init hidden state = zeros ; take last hidden state as next hidden state.

    input : prefix data points (X_test when scoring) dim (batch_size, len, 2)
    num_future : number of points we want to predict, (0 when scoring)

    """

    batch_size = input.shape[0]

    state = model.init_hidden(batch_size=batch_size)
    #print(input.shape)
    output = [input[:,0].view(batch_size,1,2) ]

    for i in range(input.shape[1] + num_future):
        #if i == 0:
        #    yhat, state = model(input[0].view(-1, batch_size, 2), state)
        #else:
        yhat, state = model(output[-1], state)
        # yhate: dim(batch, len, 2)
        print(yhat.shape,yhat)
        assert yhat.shape == (batch_size, 1, 2)
        #print(output)
        if i < len(input) - 1:
            #if not output:
            #    output = torch.Tensor(input[i].cpu()).to(mydevice)
            #else:
            output.append(input[:,i+1].view(batch_size,1,2))
        else:
            #output = torch.cat([output, yhat.detach()])
            output.append(yhat.detach())

    #for e in output:
    #    print(e)
    
    visual([e.cpu().numpy()[0,0,0] for e in output], [e.cpu().numpy()[0,0,1] for e in output])
    return output[len(input):]


def visual(x, y):

    #txt0 = [f'Point n°{t}' for t in range(X_train.shape[0])]
    txt = [f"Point n°{t}" for t in range(len(x))]
    #print(output)
    trace_0 = go.Scatter(x=x, y=y, name="LSTM", text=txt)
    #trace_1 = go.Scatter(x=test_dataset.y.cpu()[:, 0], y=test_dataset.y.cpu()[:, 1], name='Test', text = txt0)
    
    data = [trace_0]

    layout = go.Layout(
        title=f'Targets et Predictions',
        xaxis = dict(
            title='Latitude',
            ticklen = 5,
            showgrid = True,
            zeroline = False
        ),
        yaxis = dict(
            title='Logitude',
            ticklen=5,
            showgrid=True,
            zeroline=False,
        )
    )

    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename = 'lstm')


# %%
# gridsearchcv

input_size = output_size = 2 # len(["Latitude","Longitude"])

# params optimization


min_loss = float('inf')
epochs = 500
opt_model = None

for num_layer in [1, 2, 3]:
    for lr in [1e-4, 1e-2]:
        for num_hiddens in [50, 100]:

            criterion = nn.MSELoss()
            model = Model(input_size=input_size, output_size=output_size, hidden_dim=num_hiddens, n_layers=num_layer).to(mydevice)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss = run_train(input_size,output_size,lr,num_hiddens,num_layer,model,criterion,optimizer,epochs)
            
            #noX_pred(test_dataset, model)
            output = noX_pred(test_dataset.x, 40, model)
            #print('Loss in {}th ')
            #output = pred(y_train[-2:].reshape((-1,2)), all_test.shape[0], model)
            #loss_test = criterion(torch.Tensor(output), all_test)
            #print('Loss with all_test', loss_test)
            #if loss < min_loss:
            #    min_loss = loss
            #    opt_model = model


# %%

# %%
