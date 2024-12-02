import torch.optim as optim
from LSTMmodel import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from dataset import *

# data_dir = r"C:\LabAssignment\Final\data"
data_dir = r"C:\Users\yeeli\OneDrive\Documents\GitHub\MepiaPipe_hand_tracking" #update this if move "skeleton_data" folder into other folder
full_dataset = HandSkeleton(data_dir)

train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=22)
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

num_classes = 64
output_size = 128
lstm_hidden_size = 128
lstm_num_layers = 2
dropout = 0.2
learning_rate = 0.001
batch_size = 32
num_epochs = 50

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
# Create model
device = 'cuda'
model = CNNLSTMModel(num_classes=num_classes, 
                     output_size=output_size, 
                     lstm_hidden_size=lstm_hidden_size, 
                     lstm_num_layers=lstm_num_layers, 
                     dropout=dropout)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for face_point, hand_point, labels, _ in train_loader:
        # Move data to device
        face_point, hand_point, labels = face_point.to(device), hand_point.to(device), labels.to(device)

        # Forward pass
        outputs = model(face_point, hand_point)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for face_point, hand_point, labels, _ in val_loader:
            # Move data to device
            face_point, hand_point, labels = face_point.to(device), hand_point.to(device), labels.to(device)

            # Forward pass
            outputs = model(face_point, hand_point)
            loss = criterion(outputs, labels)

            # Track metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val

    # Print epoch results
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if epoch%10==9:
        #save last network
        print('saving model...')
        torch.save(model.state_dict(), 'hand_skeleton_model.pth')

#Test
#test_dataset = HandSkeleton("path/to/test/data", train=False)
#test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
