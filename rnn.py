#%%
import numpy as np 
import matplotlib.pyplot as plt
import dataSource as ds
import plotly.graph_objects as go
from plotly.offline import iplot,plot
import torch

mydevice = torch.device("cpu")

#%% [markdown]
## Parametrage
#%%
from sklearn.model_selection import train_test_split
trip = 12

df = ds.importData()

def treat(df, batch_size, train_size, step):

    df = df.sort_values(by="GpsTime")
    
    df = df[df["Trip"] == trip]

    data = df[["Latitude","Longitude"]]
    
    step = step//200

    N = len(data)
    train_df, test_df = data[:int(N*train_size)], data[int(N*train_size):]
    X_train = ds.echantillon(train_df[:-step], step).to_numpy()
    y_train = ds.echantillon(train_df[step:], step).to_numpy()

    X_test = ds.echantillon(test_df[:-step], step).to_numpy()
    y_test = ds.echantillon(test_df[step:], step).to_numpy()
    
    return X_train, y_train, X_test, y_test#, all_test

batch_size, train_size,step = 32, 0.5, 200

#X_train, y_train, all_test = treat(df, batch_size, train_size, step)

X_train, y_train, X_test, y_test = treat(df, batch_size, train_size, step)

plt.scatter(X_train[:, 0], X_train[:, 1], s = 1, c = 'y')
plt.scatter(y_train[:, 0], y_train[:, 1], s = 1, c = 'r')
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
train_loader = DataLoader(dataset,shuffle=False,batch_size=batch_size, drop_last = True)

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
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_size)
    
    def forward(self, x, state):
        """As we divided data into batches, to respect time order, every forward should take previous state and, produces a new state"""

        #Initializing hidden state for first input using method defined below
        #hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, new_state = self.lstm(x, state)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        
        return out, new_state
    
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
            y_pred, state = model(data[0].view(1, batch_size, input_size), state)
            
            loss = criterion(y_pred,data[1])
            
            # detach hidden state from graph to prevent repeatitif calculation
            for s in state:
                s.detach_()

            optimizer.zero_grad()
            loss.backward() 
            grad_clipping(model, 1)
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

    input : prefix data points (X_test when scoring)
    num_future : number of points we want to predict, (0 when scoring)

    """
    batch_size=1

    state = model.init_hidden(batch_size=batch_size)
    output = [input[0]]

    for i in range(len(input) + num_future):
        #if i == 0:
        #    yhat, state = model(input[0].view(-1, batch_size, 2), state)
        #else:
        yhat, state = model(output[-1].view(-1, batch_size, 2), state)


        if i < len(input) - 1:
            #if not output:
            #    output = torch.Tensor(input[i].cpu()).to(mydevice)
            #else:
            output.append(input[i+1])
        else:
            #output = torch.cat([output, yhat.detach()])
            output.append(yhat[0].detach())

    #print(output)
    visual([e.cpu().numpy()[0] for e in output], [e.cpu().numpy()[1] for e in output])
    return output[len(input):]


def X_pred(input, model):

    batch_size=input

    state = model.init_hidden(batch_size=batch_size)
    
    output, _ = model(input.x.view(-1, batch_size, 2), state)

    print(input.x.view(-1, batch_size, 2)[0])

    data_predict = output.detach().cpu().numpy()

    print(data_predict[0])
    visual(data_predict[:,0], data_predict[:, 1])

    return data_predict


def visual(x, y):

    txt0 = [f'Point n°{t}' for t in range(X_train.shape[0])]
    txt = [f"Point n°{t}" for t in range(len(x))]
    txt2 = [f"Point n°{t}" for t in range(len(dataset.y))]
    #print(output)
    trace_0 = go.Scatter(x=x, y=y, name="LSTM", text=txt)
    trace_1 = go.Scatter(x=test_dataset.y.cpu()[:, 0], y=test_dataset.y.cpu()[:, 1], name='Test', text = txt0)
    trace_2 = go.Scatter(x=dataset.y.cpu()[:, 0], y=dataset.y.cpu()[:, 1], name='Train', text = txt2)
    data = [trace_0,trace_1, trace_2]
    layout = go.Layout(
        title=f'Targets et Predictions de Trip {trip}',
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
epochs = 1500
opt_model = None

for num_layer in [1, 2, 3]:
    for lr in [1e-6, 1e-4, 1e-2]:
        for num_hiddens in [50, 100]:

            criterion = nn.MSELoss()
            model = Model(input_size=input_size, output_size=output_size, hidden_dim=num_hiddens, n_layers=num_layer).to(mydevice)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            loss = run_train(input_size,output_size,lr,num_hiddens,num_layer,model,criterion,optimizer,epochs)

            X_pred(test_dataset, model)
            #print('Loss in {}th ')
            #output = pred(y_train[-2:].reshape((-1,2)), all_test.shape[0], model)
            #loss_test = criterion(torch.Tensor(output), all_test)
            #print('Loss with all_test', loss_test)
            #if loss < min_loss:
            #    min_loss = loss
            #    opt_model = model


# %%
