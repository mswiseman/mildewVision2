Testing of segmentation methods has been limited due to: 

* difficulty with annotation
* exceptional performance already achieved with classification 
* greater inference time (~5x) as compared to our classification model

Nonetheless, we did generate ~300 or so segmentation labels for hop powdery mildew which we split in test/train/val datasets. This directory shows some of the example output. The IoU and DICE performance is modest. Further enhancement of model performance would likely mostly come from additional training data, but altering dilation, hole-filing, and various thresholds could also potentially help. Since our classification model performed exceptionally well as-is, we decided to abandon this route. 

Metrics from best model: 

```
[2026-02-04 00:46:04,618] - [root] - [INFO] - Loss of the network on the 216 train images: 0.281020
[2026-02-04 00:46:04,619] - [root] - [INFO] - Pixel accuracy of the network on the 216 train images: 97.258%
[2026-02-04 00:46:04,619] - [root] - [INFO] - IOU accuracy of the network on the 216 train images: 85.065%
[2026-02-04 00:46:04,619] - [root] - [INFO] - Dice accuracy of the network on the 216 train images: 91.930%
[2026-02-04 00:46:09,988] - [root] - [INFO] - Loss of the network on the 52 val images: 1.303747
[2026-02-04 00:46:09,988] - [root] - [INFO] - Pixel accuracy of the network on the 52 val images: 91.251%
[2026-02-04 00:46:09,989] - [root] - [INFO] - IOU accuracy of the network on the 52 val images: 54.743%
[2026-02-04 00:46:09,989] - [root] - [INFO] - Dice accuracy of the network on the 52 val images: 70.754%
```
