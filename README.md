# PyTorch-SSD [in progress]
[Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) in PyTorch.

## Use pretrained VGG16 model
I do not recommend training SSD from scratch. Use pretrained VGG model helps a lot to achieve lower losses.

I use the pretrained VGG16 model from [here](https://github.com/jcjohnson/pytorch-vgg), thanks to Justin Johnson.

## Credit
This implementation is heavily inspired by:
- [Hakuyume/chainer-ssd](https://github.com/Hakuyume/chainer-ssd)  
- [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)  

Check them out.
