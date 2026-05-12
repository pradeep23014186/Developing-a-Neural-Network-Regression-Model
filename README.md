# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Regression is a supervised learning technique used to predict continuous numerical values based on input data. In this problem, the goal is to develop a neural network model that learns the relationship between a numeric input and a numeric output from a dataset, and then uses this learned relationship to make predictions on new data. A neural network regression model consists of an input layer, one or more hidden layers, and an output layer. The input is processed through the network using weighted connections and activation functions like ReLU, and the final output layer produces a continuous value using a linear activation function. The model learns by adjusting its weights to minimize the difference between predicted and actual values. Before training, the data is normalized using techniques like Min-Max Scaling to improve performance. The model is trained using a loss function such as Mean Squared Error (MSE) and an optimizer like Adam. After training, the model is evaluated using test data, and its performance can be visualized using plots like the loss curve.

## Neural Network Model
<img width="884" height="578" alt="nn_model" src="https://github.com/user-attachments/assets/8ca89f46-8877-4527-927f-31c79dbb12c1" />


## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: Pradeep Kumar G

### Register Number: 212223230150

```python
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

dataset1 = pd.read_csv('/content/drive/MyDrive/Deep_Learning/Experiment_1/dataset-DL.csv')
X = dataset1[['Input']].values
y = dataset1[['Output']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}
  def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the Model, Loss Function, and Optimizer
lig = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(lig.parameters(), lr=0.001)

def train_model(lig, X_train, y_train, criterion, optimizer, epochs=2000):
  for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(lig(X_train), y_train)
        loss.backward()
        optimizer.step()
        lig.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

train_model(lig, X_train_tensor, y_train_tensor, criterion, optimizer)

with torch.no_grad():
    test_loss = criterion(lig(X_test_tensor), y_test_tensor)
    print(f'Test Loss: {test_loss.item():.6f}')

loss_df = pd.DataFrame(lig.history)

import matplotlib.pyplot as plt
loss_df.plot()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss during Training")
plt.show()

X_n1_1 = torch.tensor([[9]], dtype=torch.float32)
prediction = lig(torch.tensor(scaler.transform(X_n1_1), dtype=torch.float32)).item()
print(f'Prediction: {prediction}')
```

## Dataset Information
<img width="142" height="354" alt="image" src="https://github.com/user-attachments/assets/25671088-2f6a-44d6-9600-990a8aa402df" />


## OUTPUT

### Training Loss Vs Iteration Plot
<img width="696" height="459" alt="image" src="https://github.com/user-attachments/assets/cd592b7c-d948-4c3e-abad-af6a18470f38" />


### New Sample Data Prediction
<img width="700" height="102" alt="image" src="https://github.com/user-attachments/assets/e7dacd11-34b7-4329-b7a8-e7532f76aff2" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
