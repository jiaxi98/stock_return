import torch 
import torch.nn as nn


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu1 = nn.ReLU()
        #self.fc2 = nn.Linear(hidden_size, hidden_size)  
        #self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, output_size) 
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        #out = self.fc2(out)
        #out = self.relu2(out)
        out = self.fc3(out)
        return out
    

# currently, we directly use the same data form as the feed forward, but 
# there may exist more efficient way to do this as we r doing time series modeling

'''
# loading training and testing data for stock price prediction
training_data = torch.load('../data/training_data.pt')
training_label = torch.load('../data/training_label.pt')
testing_data = torch.load('../data/testing_data.pt')
testing_label = torch.load('../data/testing_label.pt')'''

# loading training and testing data for stock return prediction
data = torch.load('../data/stock_return.pt')
training_data = data[sample_num//5:,:-1]
training_label = data[sample_num//5:,-1:]
testing_data = data[:sample_num//5,:-1]
testing_label = data[:sample_num//5,-1:]


# Hyper-parameters 
input_size = training_data.shape[1]
hidden_size = 100
output_size = testing_label.shape[1]
num_epochs = 500
batch_size = 10000
learning_rate = 0.0001
trainsample_num = training_data.shape[0]
testsample_num = testing_data.shape[0]


model = NeuralNet(input_size, hidden_size, output_size).to(device)
#model = torch.load('trained_model/FNN_model.ckpt')
print("********************* Numerical report for FNN model *********************")

# Loss and optimizer
criterion = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    error = 0
    relative_error = 0
    for i in range(0, trainsample_num, batch_size):
        input = training_data[i:i+batch_size, :].to(device)
        label = training_label[i:i+batch_size, :].to(device)

        
        # Forward pass
        outputs = model(input)
        loss = criterion(outputs, label)
        error = error + loss.item()
        relative_error = relative_error + loss.item()/torch.mean(label**2)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    if epoch % 100 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}' 
            .format(epoch+1, num_epochs, error))
        print ('Epoch [{}/{}], Relative error: {:.4f}' 
            .format(epoch+1, num_epochs, relative_error))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
batch_size = 10
with torch.no_grad():
    error = 0
    for i in range(0, testsample_num, batch_size):
        input = testing_data[i:i+batch_size, :].to(device)
        label = testing_label[i:i+batch_size, :].to(device)

        
        # Forward pass
        outputs = model(input)
        loss = criterion(outputs, label)
        error = error + loss.item()
        relative_error = relative_error + loss.item()/torch.mean(label**2)
    

    print('Accuracy of the network on the {} test time stamp: {}'.format(testsample_num, error*batch_size/testsample_num)) 

# Save the model checkpoint
model.log = []
model.log.append('500 epochs using Adam optimizer with learning rate 0.0001')
torch.save(model.state_dict(), 'model/FNN_model.ckpt')