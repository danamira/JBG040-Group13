import torch
import matplotlib.pyplot as plt

diseases = ['Atelectasis', 'Effusion', 'Infiltration', 'No finding', 'Nodule', 'Pneumothorax']

Atelectasis_images = torch.load("data_for_generation/Atelectasis.pt")
Effusion_images = torch.load("data_for_generation/Effusion.pt")
Infiltration_images = torch.load("data_for_generation/Infiltration.pt")
No_finding_images = torch.load("data_for_generation/No finding.pt")
Nodule_images = torch.load("data_for_generation/Nodule.pt")
Pneumothorax_images = torch.load("data_for_generation/Pneumothorax.pt")

plt.figure(figsize=(5, 5))


i = 1
while input('continue? ') not in ['n''no']:
    plt.imshow(Infiltration_images[i].squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()
    i+=1