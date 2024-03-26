# Paper overview
## How does the transformer work? Model overview
- Image is divided into patches
- Patches act as tokens for the transformer (patches are 16x16 in the paper) (pinkK
- Alongside the patches you feed positional embeddings to the model (purple). Among them, there is an extra learnable class embedding. Thus, the number of tokens is nr_patches + 1
- Only the learnable class embedding is passed further to the MLP Head (orange)


# Implementation
## Defining hyperparameters
- learning_rate: learning rate for the training
- num_classes: how many classes to classify into: 6
- patch_size: how big should the patches be: original paper-> 16
- img_size: how big is the image? - we can change the image size to whatever we wantL: ex 224
- in_channels: number of channels of the image: 1 for us
- num_heads: how many attention heads to use
- dropout: dropout for the training
- hidden_dimension: hidden dimension of the MLP part of the Encoder. In paper, 768, 1024, 1280 are all used
- adam_weight_decay: ??? e.g. 0
- adam_betas: ??? e.g. (0.9,0.999)
- activation: activation function (paper uses GeLU) - used in the encoder layers
- embed_dim: (patch_size**2)*in_channels (dimension of the embedding for a patch)
- num_patches: (img_size // patch_size)**2 (number of patches)

## Implementation remarks (implementation taken from https://www.youtube.com/watch?v=Vonyoz6Yt9c&t=460s&ab_channel=UygarKurt and compared with the existing implementation in torchvision)
- class embeddings can be initialised with random numbers as opposed to zeros, maybe this improves performance? (see 14:30)
- same goes for positional embeddings (see 14:30)