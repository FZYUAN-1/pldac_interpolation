#%%
import numpy as np 
import torch
import matplotlib.pyplot as plt
import dataSource as ds
import plotly.graph_objects as go
from plotly.offline import iplot,plot
#%%
#! pip install plotly
#%% [markdown]
## Parametrage
df = ds.importData()

#%%
from sklearn.model_selection import train_test_split

trips = np.unique(df["Trip"])

batch_size,train_size,step = 2, 0.8, 200

X_train, X_test, y_train, y_test = ds.create_data_xy(df[["Trip","Latitude","Longitude"]],train_size,step,step)

X_train = X_train.to_numpy()
X_test = X_test.to_numpy()
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


plt.scatter(X_train[:, 1], X_train[:, 2], s = 1, c = 'b')
plt.scatter(y_train[:, 1], y_train[:, 2], s = 1, c = 'r')
plt.show()


#%%
#dataset
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
class timeseries(Dataset):
    def __init__(self,x,y):
        self.x = torch.tensor(x,dtype=torch.float32)
        self.y = torch.tensor(y,dtype=torch.float32)
        self.len = len(np.unique(x))

    def __getitem__(self,trip):
        #print(trip)
        return self.x[self.x[:,0]==trip][:,1:], self.y[self.y[:,0]==trip][:,1:]
  
    def __len__(self):
        return self.len

def collate_fn(batch):
    x,y = zip(*batch)
    max_length = max([len(x[i]) for i in range(len(x))])
    features_x = torch.zeros((len(x), max_length, 2))
    features_y = torch.zeros((len(y), max_length, 2))

    for i in range(len(x)):
        length = x[i].size(0)
        features_x[i] = torch.cat((x[i], torch.zeros((max_length-length, 2))))
        features_y[i] = torch.cat((y[i], torch.zeros((max_length-length, 2))))

    return features_x, features_y

dataset = timeseries(X_train,y_train)
#dataloader
from torch.utils.data import DataLoader 
train_loader = DataLoader(dataset,sampler=SubsetRandomSampler(list(trips)), collate_fn=collate_fn, batch_size=2, drop_last = True)

# verify train_loader, especially check if batches are all consecutives
'''
torch.set_printoptions(sci_mode=False)
l = list(train_loader)
print(l[0], l[1])
'''
#%%
t = SubsetRandomSampler(trips)
for x,y in train_loader:
    print(x)
#%%
# rnn
#neural network
device = torch.device("cuda")

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
    
    def forward(self, x):
        """As we divided data into batches, to respect time order, every forward should take previous state and, produces a new state"""

        #Initializing hidden state for first input using method defined below
        #hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, new_state = self.lstm(x)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = self.fc(out)
        
        return out
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
         # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = (torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(device), torch.zeros((self.n_layers, batch_size, self.hidden_dim)).to(device))
        return hidden

#%%
# Instantiate the model with hyperparameters

input_size = output_size = 2 # len(["Latitude","Longitude"])
# Define hyperparameters

lr = 1e-2
num_hiddens = 256 # n. hidden units
num_layer = 1   # 1 single rnn_layer
                # multiple layers are possible but we start with a single one..

model = Model(input_size=input_size, output_size=output_size, hidden_dim=num_hiddens, n_layers=num_layer)

# We'll also set the model to the device that we defined earlier (default is CPU)
model.to(device)


# Define Loss, Optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#%%
# raw output
#init_state = model.init_hidden(batch_size)
#model(list(train_loader)[0], init_state)

# %%
# training 

def grad_clipping(net, theta): 
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


epochs = 300
batch_size = 2
state = model.init_hidden(batch_size) # init state, dim (n_layers, batch_size, hidden_dim)

for i in range(epochs):
    for x,y in train_loader:
        #print(data.shape, data)
        #print(data[0].view(1, batch_size, input_size).shape)

        y_hat = model(x.view(-1, batch_size, input_size).to(device))
        
        loss = criterion(y_hat,y.view(-1, batch_size, output_size).to(device))

        optimizer.zero_grad()
        loss.backward() # each successive batch will take more time than the previous one because it will have to back-propagate all the way through to the start of the first batch.
        #grad_clipping(model, 1)
        optimizer.step()

    if i%50 == 0:
        print(i,"th iteration : ",loss)
#%%
print(X_test[:,0])

#%%
# test set actual vs predicted

def pred(input, num_future, model):
    """
    take last output as next input; init hidden state = zeros ; take last hidden state as next hidden state.
    
    input : prefix data points (X_test when scoring)
    num_future : number of points we want to predict, (0 when scoring)
    
    """

    #state = model.init_hidden(batch_size=1)

    output = [] # first point
    for i in range(len(input) + num_future):
        if i == 0:
            yhat = model(input[0].view(-1, 1, 2))    
        else:
            yhat = model(output[-1].view(-1, 1, 2))


        if i <= len(input) - 1:
            print(input[i].view(1,2))
            output.append(input[i].view(1,2))
        else:
            output.append(yhat.detach().view(1,2))
    

    return output

output = pred(torch.Tensor(X_test[:20,1:]).to(device), X_train.shape[0], model)
txt0 = [f'Point n°{t}' for t in range(X_train.shape[0])]
txt = [f"Point n°{t}" for t in range(len(output))]
trace_0 = go.Scatter(x=[e[0,0].item() for e in output], y=[e[0,1].item() for e in output], name="LSTM", text=txt)
trace_1 = go.Scatter(x=y_test[X_test[:,0]==186][:, 1], y=y_test[X_test[:,0]==186][:, 2], name='Xtest', text = txt0)
data = [trace_0,trace_1]
layout = go.Layout(
    title=f'Targets et Predictions ',
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
#plt.plot([e[0][0].item() for e in output], [e[0][1].item() for e in output], label='predicted{}'.format((j)))
#%%
print([e[0,0].shape for e in output])
#%%

test_set = timeseries(X_test,y_test)
#test_loader = DataLoader(test_set,shuffle=False,batch_size=batch_size, drop_last = True)

#state_pred = state

plt.scatter(test_set.y[:,0],test_set.y[:,1], s = 1,label='original')

for j, data in enumerate(test_set.x):
    
    y_pred, state_pred = model(data[0].view(1,batch_size,input_size), state_pred) # state was well trained after iteration
    
    tmp = y_pred.detach().numpy()
    
    plt.scatter(tmp[:,0], tmp[:,1], s = 1, label='predicted{}'.format((j)))


    plt.legend()
# %%
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge

# %%
! pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 --user -f https://download.pytorch.org/whl/torch_stable.html
# %%
torch.tensor([1,2,3])
torch.cuda.is_available()
# %%
