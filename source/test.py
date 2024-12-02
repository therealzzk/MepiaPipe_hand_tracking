import argparse
import torch.optim as optim
from LSTMmodel import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from dataset import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()
# data_dir = r"C:\LabAssignment\Final\data"
data_dir = "" #update this if move "skeleton_data" folder into other folder
full_dataset = HandSkeleton(data_dir)

train_indices, val_indices = train_test_split(range(len(full_dataset)), test_size=0.2, random_state=22)
train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

num_classes = 64
output_size = 128
lstm_hidden_size = 128
lstm_num_layers = 2
dropout = 0.2
learning_rate = 0.0001
batch_size = 32
num_epochs = 70
device = 'cuda'
model = CNNLSTMModel(num_classes=num_classes, 
                    output_size=output_size, 
                    lstm_hidden_size=lstm_hidden_size, 
                    lstm_num_layers=lstm_num_layers, 
                    dropout=dropout)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
    

#Test
#create skeleton


#utils.getSkeleton("042_test.mp4") #replace the video name to get skeleton of the video
utils.getSkeleton("042_test_video01.mp4") 

test_dataset = HandSkeleton("", train=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
model.load_state_dict(torch.load("hand_skeleton_model_xiaohe_1.pth"))
model.eval()

for face_point, hand_point, labels in test_loader:
    # print(f'face_point: {face_point}')
    # print(f'hand_point: {hand_point}')
    print(f'GT_labels: {labels}')
    face_point, hand_point, labels = face_point.to(device), hand_point.to(device), labels.to(device)
    outputs = model(face_point, hand_point)
    # loss = criterion(outputs, labels)

    pred_output = outputs.detach().cpu().numpy()
    # print(pred_output)
    _, predicted = torch.max(outputs, 1)
    print(f'pred_labels: {predicted[0]}')
    

print("End test")