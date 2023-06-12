import subprocess

lambda_ortho_values = [0.01]

for lambda_ortho in lambda_ortho_values:
    command1 = f"python train_T1.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command1, shell=True)
    command2 = f"python train_T2.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command2, shell=True)
    command3 = f"python train_T3.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command3, shell=True)
    command4 = f"python train_T4.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    subprocess.run(command4, shell=True)
