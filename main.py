import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import pandas as pd
import matplotlib.pyplot as plt

losses = [] # list to store the losses over time
avg_losses = [] # list to store the average losses every 5 iterations
avg_loss = 0 # variable to accumulate the losses every 5 iterations
n = 10 # calculate the average loss every 5 iterations
num_iterations = 0 # counter for the number of iterations

# load data from csv file
df = pd.read_csv('fw_stats.csv')

full_db = df

df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')

# create empty lists for inputs and targets
train_inputs = []
train_targets = []

# loop through the rolling window of 10 values
for i in range(10, len(df)):
    # get the 10 previous values as the input
    input_vals = df.iloc[i-10:i, df.columns.get_loc('PTS')].values

    home = 1
    if full_db.iloc[i]['Home/Away'] == '@':
        home = 0
    input_vals = np.append(input_vals, home)
    train_inputs.append(input_vals)
    # get the current value as the target

    target_val = df.iloc[i, df.columns.get_loc('PTS')]
    train_targets.append(target_val)

# convert lists to numpy arrays
train_inputs = np.array(train_inputs)
train_targets = np.array(train_targets)

print(train_inputs)
print(train_targets)




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # input parameters equal to
        # --> last 10 games pts performance (10 nodes)
        # --> home or away (1 node)
        self.fc1 = nn.Linear(11, 15)
        self.fc2 = nn.Linear(15, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 1)
        self.relu = nn.ReLU()

        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()

        # ^ use if you want to reset the weights

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        return x
# Define the dataset
class MyDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]



model = MyModel()
model.load_state_dict(torch.load('my_model_weights.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
train_dataset = MyDataset(train_inputs, train_targets)
train_dataloader = DataLoader(train_dataset, batch_size=1)


print()
print()
print( "---------- Training ----------")
print()
print()
print()

# Train the model
for epoch in range(5): # 5 iterations of data
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(data)
        target = target.unsqueeze(1) # Add a new dimension to the target tensor
        loss = criterion(output.squeeze(), target.squeeze())
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        num_iterations += 1

        # calculate the average loss every 5 iterations
        if num_iterations % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            avg_losses.append(avg_loss)
        print('Epoch: {}, Batch: {}, Loss: {:.4f}'.format(epoch, batch_idx, loss.item()))
        print('Predictions:', output.squeeze().detach().numpy().mean())
        print('Ground Truth:', target.squeeze().detach().numpy())



print()
print()
print()
print("---------- Losses Graph ----------")
print()
print()
print()
print()


# plot the average losses over time
average_losses_per_5 = []
index = 0
for i in range(int(len(avg_losses)/5) - 1):
    sum = 0
    for j in range(5):
        sum += avg_losses[index]
        index += 1
    average_losses_per_5.append(sum/5)


plt.bar(range(len(average_losses_per_5)), average_losses_per_5, align='center')
plt.xticks(range(len(average_losses_per_5)))
plt.title('Average Training Loss (Every 5 Iterations)')
plt.xlabel('Iteration / 5')
plt.ylabel('Loss')
plt.show()

# make a prediction on the next game which is toady
model.eval()
new_data = [16, 20, 21, 20, 16, 19, 25, 20, 16, 17,  1]
new_data_tensor = torch.tensor(new_data, dtype=torch.float32)

with torch.no_grad():
    predictions = model.forward(new_data_tensor)
    print("Prediction: " , str(predictions.mean().item()))

torch.save(model.state_dict(), 'my_model_weights.pth')