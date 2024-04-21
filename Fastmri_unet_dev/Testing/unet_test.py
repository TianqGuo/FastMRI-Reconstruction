import torch
import torch.nn
import numpy as np

from collections import defaultdict


def test(args, model, test_loader):
    '''

    Args:
        model: pretrained model used to reconstruct MR image
        test_loader: test data loader

    Returns:
        reconstruction: Dictionary of mapping from file name to reconstruction image
    '''
    
    model.eval()
    reconstructions = defaultdict(list)              # defaultdict could hold a key-value pair

    with torch.no_grad():
        for i, (input, _, _, fnames, slices) in enumerate(test_loader):

            input = input.to(args.device)
            pred = model(input)

            N = len(fnames)                     # batch size N
            for i in range(N):
                reconstructions[fnames[i]].append((slices[i].numpy(), pred[i].numpy()))         # torch tensor needs to be converted to nd.array

    for fname, slice_pred in reconstructions.items():
        reconstructions[fname] = np.stack(
            [pred for _, pred in sorted(slice_pred)]
        )

    return reconstructions