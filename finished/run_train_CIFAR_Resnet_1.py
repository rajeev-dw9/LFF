import subprocess

lambda_ortho_values = [0.001, 0.002, 0.01, 0.1]

for lambda_ortho in lambda_ortho_values:
    command = f"python3 train_CIFAR_Resnet_1.py with server_user corrupted_cifar10 type1 skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command}" + "\n")
    subprocess.run(command, shell=True)
