# Yolov3.jittor

pytorch source: https://github.com/ultralytics/yolov3

This is reimplemented in jittor


This is the comparsion of speed (yolov3.cfg)

Due to GPU memory limitations, we use batch-size=8 when we train this model.

|  framework   | train(batch-size=8) | test(batch-size=16) | benchmark(batch-size=16)
|  ----  | ----  | ---- | ---- |
| pytorch  | 35.9 ms | 9.0ms | 4.1ms|
| jittor | 32.2ms | 8.5ms | 3.8 ms |
| Speedup | 1.11 | 1.06| 1.08|