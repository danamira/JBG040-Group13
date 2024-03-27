from tqdm import tqdm
import torch
from dc1.net import Net
from dc1.batch_sampler import BatchSampler
from typing import Callable, List


def train_model(
        model: Net,
        train_sampler: BatchSampler,
        optimizer: torch.optim.Optimizer,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    losses = []
    model.train()
    for batch in tqdm(train_sampler):
        x, y = batch
        x, y = x.to(device), y.to(device)
        output = model(x)  # Using model directly calls forward

        # Check if the model's output is a tuple (for models returning feature maps)
        # and unpack accordingly
        if isinstance(output, tuple):
            predictions, _ = output
        else:
            predictions = output

        loss = loss_function(predictions, y)
        losses.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses



def test_model(
        model: Net,
        test_sampler: BatchSampler,
        loss_function: Callable[..., torch.Tensor],
        device: str,
) -> List[torch.Tensor]:
    losses = []
    model.eval()
    with torch.no_grad():
        for (x, y) in tqdm(test_sampler):
            x, y = x.to(device), y.to(device)
            output = model(x)

            # Similarly, handle the tuple output for models that return additional data
            if isinstance(output, tuple):
                prediction, _ = output
            else:
                prediction = output

            loss = loss_function(prediction, y)
            losses.append(loss)
    return losses