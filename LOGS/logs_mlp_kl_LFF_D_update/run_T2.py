import subprocess

# lambda_ortho_values = [0.001, 0.01, 0.1, 1]
lambda_ortho_values = [0.01]

for lambda_ortho in lambda_ortho_values:
    command1 = f"python train_lff_kl_0.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command1}")
    subprocess.run(command1, shell=True)
    command2 = f"python train_lff_kl_2.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command2}")
    subprocess.run(command2, shell=True)
    command3 = f"python train_lff_kl_4.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command3}")
    subprocess.run(command3, shell=True)
    command4 = f"python train_lff_kl_02.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command4}")
    subprocess.run(command4, shell=True)
    command5 = f"python train_lff_kl_04.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command5}")
    subprocess.run(command5, shell=True)
    command6 = f"python train_lff_kl_24.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command6}")
    subprocess.run(command6, shell=True)
    command7 = f"python train_lff_kl_024.py with server_user colored_mnist skewed3 severity4 lambda_ortho={lambda_ortho}"
    print(f"Running command: {command7}")
    subprocess.run(command7, shell=True)
