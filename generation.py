import os
import subprocess
import sys
import numpy as np
from matplotlib import pyplot as plt
# Model Used: https://github.com/dsaragih/diffuse-gen?tab=readme-ov-file

# Paths setup
data_dir = './data_for_generation/'
output_path = './image_samples/'
model_path = 'model200000.pt'

diseases = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']
image_counts = {
    'Atelectasis': 2521,
    'Effusion': 2318,
    'Infiltration': 2964,
    'No finding': 6103,
    'Nodule': 1633,
    'Pneumothorax': 1302
}
max_images = max(image_counts.values())

def generate_images_for_disease(disease, num_to_generate):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")

    output_disease_path = os.path.join(output_path, disease)
    os.makedirs(output_disease_path, exist_ok=True)

    command = [
        sys.executable, 'main.py',
        '--data_dir', data_dir,
        '--output_path', output_disease_path,
        '--model_path', model_path,
        '--diff_iter', '100',
        '--timestep_respacing', '200',
        '--skip_timesteps', '80',
        '--model_output_size', '256',
        '--num_samples', str(num_to_generate),
        '--batch_size', '1',
        '--use_noise_aug_all',
        '--use_colormatch',
        '-fti', '-sty', '-inp', '-spi'
    ]

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error generating images for {disease}: {e}", file=sys.stderr)

for disease in diseases:
    num_to_generate = max_images - image_counts[disease]
    if num_to_generate > 0:
        print(f"Generating {num_to_generate} images for {disease}...")
        try:
            generate_images_for_disease(disease, num_to_generate)
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            break