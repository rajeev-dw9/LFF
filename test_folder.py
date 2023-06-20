import os
import torch
from torch.utils.data import DataLoader
from module.util import get_model
from data.util import get_dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "/home/rajeev/rrd/LFF/DATA"
colored_mnist = "ColoredMNIST-Skewed0.01-Severity4"
folder_path = "/home/rajeev/rrd/LFF/LOGS/log_mlp_kl_LFF_ortho_on_D/colored_mnist/result/ColoredMNIST-Skewed0.01-Severity4/D"

valid_dataset = get_dataset(
    colored_mnist,
    # corrupted_cifar,
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

# Iterate over all .th files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".th"):
        model_path = os.path.join(folder_path, file_name)
        print("Testing model:", model_path)

        state_dict = torch.load(model_path)
        model = get_model("MLP", 10).to(device)
        model.load_state_dict(state_dict['state_dict'])
        model.eval()

        # BC testing
        correct = torch.tensor(0).to(device)
        total = torch.tensor(0).to(device)

        with torch.no_grad():
            for data_tuple in tqdm(valid_loader, leave=False):
                data, label = data_tuple
                data = data.to(device)
                label = label.to(device)

                label1 = label[:, 0]
                label2 = label[:, 1]

                output, _ = model(data)
                _, predicted = torch.max(output.data, 1)
                mask = (label1 != label2)  # Create a mask for non-matching labels
                total += mask.sum()
                correct += (predicted[mask] == label1[mask]).sum()

        accuracy = 100 * correct / total
        print('Accuracy on test data with non-matching labels: {:.2f}%'.format(accuracy))
        print()  # Add a newline for separation between models
