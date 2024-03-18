from image_dataset import ImageDataset
from pathlib import Path
import matplotlib.pyplot as plt


trainData = ImageDataset(Path("data/X_train.npy"), Path("data/Y_train.npy"))



i = 1


while not input('continue? ') in ['n''no']:
    x=(trainData.__getitem__(i))
    print(x)
    plt.imshow(x[0].permute(1,2,0),cmap='gray')
    plt.show()
    i+=1

