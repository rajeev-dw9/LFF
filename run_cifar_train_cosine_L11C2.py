import subprocess

lambda_ortho_values = [0.001, 0.002, 0.01]

for lambda_ortho in lambda_ortho_values:
    command = f"python cifar_train_cosine_L11C2.py with server_user corrupted_cifar10 type1 skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command, shell=True)
