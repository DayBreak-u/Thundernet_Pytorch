# PSROIAlign with multi-batch training support - PyTorch
**Position-Sensitive-Region-of-Interests-Alignment** has been widely used throughout numerous well known deep object detectors, s.t. [R-FCN](https://arxiv.org/pdf/1605.06409.pdf), [LightHead R-CNN](https://arxiv.org/pdf/1711.07264.pdf), etc. However there are not that much implementations support ***multi-batch training*** in the world of PyTorch. With just one image per GPU, models would hardly be aware of the statistical information of the training data especially in cases that rarely one or two GPUs at hand.

This CUDA based implementation fully supports multi-batch training, and can be easily integrated into your PyTorch object detectors.


## Prerequisite
```
python3
pytorch >= 1.0 with CUDA support
```


## Build the module
```bash
sh build.sh
```


## Use Case
```python
import torch
import torch.nn as nn
from model.roi_layers import PSROIAlign
```

```python
class PSROIAlignExample(nn.Module):
    """
    :spatial_scale: stride of the backbone
    :roi_size:      output size of the pooled feature
    :sample_ratio:  sample ratio of bilinear interpolation
    :pooled_dim:    output channel of the pooled feature
    """
    def __init__(self,
                 spatial_scale=1./16.,
                 roi_size=7,
                 sample_ratio=2,
                 pooled_dim=10):

        super(PSROIAlignExample, self).__init__()
        self.psroialign = PSROIAlign(spatial_scale=spatial_scale,
                                     roi_size=roi_size,
                                     sampling_ratio=sample_ratio,
                                     pooled_dim=pooled_dim)

    def forward(self, feat, rois):
        return self.psroialign(feat, rois)
```

#### Feature Map to be pooled
```python
batch_size = 4
feat_height = 30
feat_width = 40
roi_size = 7
oup_dim = 10

feature = torch.randn((batch_size,
                       roi_size * roi_size * oup_dim,
                       feat_height,
                       feat_width),
                       requires_grad=True).cuda()
```

#### RoIs should be formatted as **(batch_index, x1, y1, x2, y2)**
```python
rois = torch.tensor([
    [0, 1., 1., 5., 5.],
    [0, 3., 3., 9., 9.],
    [1, 5., 5., 10., 10.],
    [1, 7., 7., 12., 12.]
]).cuda()
```

#### Essential Job
```python
psroialign_pooled_feat = psroialign_example(feature, rois)
```


Play with ***example.py*** to get more details.


## License
[MIT](LICENSE)