import subprocess

lambda_ortho_values = [0.001, 0.002, 0.01, 1, 2]

for lambda_ortho in lambda_ortho_values:
    command = f"python train_cosine_024.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command, shell=True)