# Data-Challenge-1
This repository contains the code for the TU/e course JBG040 Data Challenge 1.
Please read this document carefully as it has been filled out with important information.
Note: The "Code structure" section has been modified compared to the original README.md.

## Code structure
The code is structured into multiple files, based on their functionality. 
Each `.py` file contains a different part of the code. 

- To download the data: run the `ImageDataset.py` file. The script will create a directory `/dc1/data/` and download the training and test data with corresponding labels to this directory. 
    - You will only have to run this script once usually, at the beginning of your project.

- To run the whole training/evaluation pipeline: run `/dc1/train.py`. This script is prepared to do the followings:
    - Load your train and test data (Make sure its downloaded beforehand!)
    - Initializes the neural network chosen in `model.py`. The structure of the network is defined in the `net.py` file for the template model and in the `resnet.py` for ResNet.
    - Initialize loss functions and optimizers. If you want to change the loss function/optimizer, do it here.
    - Define number of training epochs and batch size.
    - Check and enable GPU acceleration for training (if you have CUDA or Apple Silicon enabled device).
    - Train the neural network and perform evaluation on test set at the end of each epoch.
    - Provide plots about the training losses both during training in the command line and as a png (saved in the `/dc1/artifacts/` subdirectory).
    - Save your trained model's weights in the `/dc1/model_weights/` subdirectory so that you can reload them later.
    - Save your experiment description, which can be defined at the beginning of the file, in `/dc1/results/CNN-template/optimizers/`.

- To calculate the accuracy, f1 score, precision and recall of the trained model: run `/dc1/use_model.py`. Make sure to specify the following:
    - `model_file_name`: the name of the .txt file with model weights.

- To use the model for the classification of an image: run `main.py`. Make sure to specify the following:
    - `modelPath`: the path to the .txt file with model weights.
    - `image_path`: the path to the image you want to classify - the image should be uploaded in `/input/`. An example image is included.

Note that the Neural Network structure is defined in the `net.py` file, so if you want to modify the network itself, you can do so in that script.
The loss functions and optimizers are all defined in `/dc1/train.py`.

## GitHub setup instructions
1. Click the green *<> Code* button at the upper right corner of the repository.
2. Make sure that the tab *Local* is selected and click *Download ZIP*.
3. Go to the GitHub homepage and create a new repository.
4. Make sure that the repository is set to **private** and give it the name **JBG040-GroupXX**, where XX is your group number.
5. Press *uploading an existing file* and upload the extracted files from Data-Challenge-1-template-main.zip to your repository. Note that for the initial commit you should commit directly to the main branch
6. Invite your **group members, tutor and teachers** by going to *Settings > Collaborators > Add people*.
7. Open PyCharm and make sure that your GitHub account is linked.*
8. In the welcome screen of PyCharm, click *Get from VCs > GitHub* and select your repository and click on clone.
9. After the repository is cloned, you can now create a virtual environment using the requirements.txt.

*For information on how to install PyCharm and link Github to your PyCharm, we refer to the additional resources page on Canvas.


## Environment setup instructions
We recommend to set up a virtual Python environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt.` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

## Submission instructions
After each sprint, you are expected to submit your code. This will **not** be done in Canvas, instead you will be creating a release of your current repository. 
A release is essentially a snapshot of your repository taken at a specific time. 
Your future modifications are not going to affect this release.
**Note that you are not allowed to update your old releases after the deadline.**
For more information on releases, see the [GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) page.

1. Make sure that your code is running without issues and that **everything is pushed to the main branch**.
2. Head over to your repository and click on *Releases* (located at the right-hand side).
3. Click on the green button *Create a new release*.*
4. Click on *Choose a tag*.
5. Fill in the textbox with **SprintX** where X is the current sprint number and press *Create new tag: SprintX*. 
6. Make sure that *Target: main* or *Target: master* (depending on your main/master branch) is selected, so that the code release will be based on your main branch. 
7. Fill in the title of the release with **Group XX Sprint X** where XX is your group number and X is the current sprint number.
8. Click the *Publish release* button to create a release for your sprint.
9. **Verify** that your release has been succesfully created by heading over to your repository and press the *Releases* button once again. There you should be able to see your newly created release.

*After the first release, you should click *Draft a new release* instead of *Create a new release*

## Mypy
The template is created with support for full typehints. This enables the use of a powerful tool called `mypy`. Code with typehinting can be statically checked using this tool. It is recommended to use this tool as it can increase confidence in the correctness of the code before testing it. Note that usage of this tool and typehints in general is entirely up to the students and not enforced in any way. To execute the tool, simply run `mypy .`. For more information see https://mypy.readthedocs.io/en/latest/faq.html

## Argparse
Argparse functionality is included in the main.py file. This means the file can be run from the command line while passing arguments to the main function. Right now, there are arguments included for the number of epochs (nb_epochs), batch size (batch_size), and whether to create balanced batches (balanced_batches). You are free to add or remove arguments as you see fit.

To make use of this functionality, first open the command prompt and change to the directory containing the main.py file.
For example, if you're main file is in C:\Data-Challenge-1-template-main\dc1\, 
type `cd C:\Data-Challenge-1-template-main\dc1\` into the command prompt and press enter.

Then, main.py can be run by, for example, typing `python main.py --nb_epochs 10 --batch_size 25`.
This would run the script with 10 epochs, a batch size of 25, and balanced batches, which is also the current default.
If you would want to run the script with 20 epochs, a batch size of 5, and batches that are not balanced, 
you would type `main.py --nb_epochs 20 --batch_size 5 --no-balanced_batches`.
