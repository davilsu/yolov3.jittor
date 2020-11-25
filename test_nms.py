import torchvision
import torch 
import jittor as jt 
from utils.nms import nms

import numpy as np 

def test(num_box=100):
    jt.flags.use_cuda=0
    xy = np.random.randint(224,size=(num_box,2))
    hw = np.random.randint(224,size=(num_box,2))
    x2y2 = xy+hw
    boxes = np.concatenate([xy,x2y2],1).astype(np.float32)
    scores = np.random.rand(num_box).astype(np.float32)
    iou_thres = 0.3

    torch_result = torchvision.ops.boxes.nms(torch.tensor(boxes), torch.tensor(scores), iou_thres)
    jittor_result = nms(jt.array(boxes),jt.array(scores),iou_thres)
    t = np.sort(torch_result.numpy())
    j = np.sort(jittor_result.numpy())
    assert np.allclose(t,j)

def test_cuda(num_box=100):
    jt.flags.use_cuda=1
    xy = np.random.randint(224,size=(num_box,2))
    hw = np.random.randint(224,size=(num_box,2))
    x2y2 = xy+hw
    boxes = np.concatenate([xy,x2y2],1).astype(np.float32)
    scores = np.random.rand(num_box).astype(np.float32)
    iou_thres = 0.3

    torch_result = torchvision.ops.boxes.nms(torch.tensor(boxes).cuda(), torch.tensor(scores).cuda(), iou_thres)
    jittor_result = nms(jt.array(boxes),jt.array(scores),iou_thres)
    t = np.sort(torch_result.cpu().numpy())
    j = np.sort(jittor_result.numpy())
    assert np.allclose(t,j)


for i in range(100,10000):
    print(i)
    test(i)
    test_cuda(i)