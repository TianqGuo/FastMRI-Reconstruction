import torch


def test(args, model, test_loader):
    '''

    Args:
        model: pretrained model used to reconstruct MR image
        test_loader: test data loader

    Returns: MRI reconstruction
    '''

    model.eval()


    with torch.no_grad():
        for ... in test_loader:



    return reconstructions