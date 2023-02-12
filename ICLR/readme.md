## Running Examples
Use REASSURE to repair a pre-trained MNIST model:
```python3
python ./Experiments/MNIST/PNN_MNIST.py
```
```
-------------------------------------------------- repair num = 10 --------------------------------------------------
Test Acc before repair: 98.0%
Working on 10 cores.
Time cost: 259.69484424591064
Repair success rate: 100%
Average inf Diff(on patch area): tensor(0.1503)
Average L2 Diff(on patch area): tensor(0.2071)
Average inf Diff(on all): tensor(0.0001)
Average L2 Diff(on all): tensor(0.0002)
Test Acc after repair: 98.1%
-------------------------------------------------- repair num = 20 --------------------------------------------------
Test Acc before repair: 98.0%
Working on 10 cores.
Time cost: 433.57593607902527
Repair success rate: 100%
Average inf Diff(on patch area): tensor(0.1257)
Average L2 Diff(on patch area): tensor(0.1728)
Average inf Diff(on all): tensor(0.0002)
Average L2 Diff(on all): tensor(0.0003)
Test Acc after repair: 98.2%
-------------------------------------------------- repair num = 50 --------------------------------------------------
Test Acc before repair: 98.0%
Working on 10 cores.
Time cost: 1184.2179553508759
Repair success rate: 100%
Average inf Diff(on patch area): tensor(0.1420)
Average L2 Diff(on patch area): tensor(0.1953)
Average inf Diff(on all): tensor(0.0006)
Average L2 Diff(on all): tensor(0.0009)
Test Acc after repair: 98.5%
-------------------------------------------------- repair num = 100 --------------------------------------------------
Test Acc before repair: 98.0%
Working on 10 cores.
Time cost: 2117.45366024971
Repair success rate: 100%
Average inf Diff(on patch area): tensor(0.1264)
Average L2 Diff(on patch area): tensor(0.1731)
Average inf Diff(on all): tensor(0.0012)
Average L2 Diff(on all): tensor(0.0017)
Test Acc after repair: 99.0%

```

Use REASSURE to repair a pre-trained watermarked model:
```python3
python ./Experiments/Watermark/MNIST.py
```

Use REASSURE to repair a pre-trained ImageNet model:
```python3
python ./Experiments/ImageNet/PNN_AlexNet.py
```

Use REASSURE to repair a pre-trained HCAS model:
```python3
python ./Experiments/HCAS/PNN_HCAS_area_repair.py
```

