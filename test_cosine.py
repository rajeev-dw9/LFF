import torch
from torch.utils.data import DataLoader
from module.util import get_model
from data.util import get_dataset
from tqdm import tqdm




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/home/rajeev/rrd/LFF/dataset"
# colored_mnist = "ColoredMNIST-Skewed0.01-Severity4"
corrupted_cifar = "CorruptedCIFAR10-Type1-Skewed0.01-Severity4"
model_path = "/home/rajeev/rrd/LFF/log_cifar/corrupted_cifar/result/CorruptedCIFAR10-Type1-Skewed0.01-Severity4/cifar_model_cosine__0.001__L41C2.th"


valid_dataset = get_dataset(
    # colored_mnist,
    corrupted_cifar,
    data_dir=data_dir,
    dataset_split="eval",
    transform_split="eval",
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

state_dict = torch.load(model_path)

# Initialize the model with the saved state dictionary
# model = get_model("MLP", 10).to(device)
model = get_model("ResNet20", 10).to(device)
model.load_state_dict(state_dict['state_dict'])
model.eval()


# Evaluate the model on the test data
# correct = 0
# total = 0
# with torch.no_grad():
#     for data_tuple in tqdm(valid_loader, leave=False):
#         data, label = data_tuple
#         data = data.to(device)
#         label = label[:, 0].to(device)  # Flatten the label tensor
#         output = model(data)
#         _, predicted = torch.max(output.data, 1)
#         total += label.size(0) #10000
#         correct += (predicted == label).sum().item()

# accuracy = 100 * correct / total
# print('Accuracy on test data: {:.2f}%'.format(accuracy))

# BC testing
correct = torch.tensor(0).to(device)
total = torch.tensor(0).to(device)

with torch.no_grad():
    for data_tuple in tqdm(valid_loader, leave=False):
        data, label = data_tuple
        data = data.to(device)
        label = label.to(device)

        label1 = label[:,0]
        label2 = label[:,1]

        output = model(data)
        _, predicted = torch.max(output.data, 1)
        mask = (label1 != label2)  # Create a mask for non-matching labels
        total += mask.sum()
        correct += (predicted[mask] == label1[mask]).sum()
accuracy = 100 * correct / total
print('Accuracy on test data with non-matching labels: {:.2f}%'.format(accuracy))
