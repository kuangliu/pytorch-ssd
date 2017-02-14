# PyTorch-SSD [in progress]
[Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) with PyTorch.  

## Dataset organization
We describe the dataset as `<image, cls, loc>`.
- `image`: image pixels.
- `cls`: list of object classes in the image, with length `N`.
- `loc`: tensor of object localization, sized `[N,4]`.

## Credit
This implementation is heavily inspired by [Hakuyume/chainer-ssd](https://github.com/Hakuyume/chainer-ssd). Go and check it out.
