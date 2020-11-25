import argparse
import json
import torch
import numpy as np

# from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

from jittor_utils import auto_diff


def test(cfg,
         data,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    # Initialize/load model and set device
    if model is None:
        is_training = False
        # device = jittor_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_parameters(torch.load(weights, map_location="cpu")['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        # model.to(device)

        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)
    else:  # called by train.py
        is_training = True
        # device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = jt.array(np.linspace(0.5, 0.95, 10))  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataloader = LoadImagesAndLabels(path, imgsz, batch_size, 
                                         rect=True, 
                                         single_cls=opt.single_cls, 
                                         pad=0.5, 
                                         num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]))

    seen = 0
    model.eval()
    _ = model(jt.zeros((1, 3, imgsz, imgsz))) if jt.flags.use_cuda else None  # run once
    coco91class = coco80_to_coco91_class()
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = jt.zeros((3,))
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader)):
        imgs = imgs.float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = jt.array([width, height, width, height])

        # Disable gradients
        with jt.no_grad():
            # Run model
            t = jittor_utils.time_synchronized()
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs
            t0 += jittor_utils.time_synchronized() - t

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = jittor_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += jittor_utils.time_synchronized() - t
        
        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[(targets[:, 0] == si).where()[0], 1:]
            nl = labels.shape[0]
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((jt.zeros((0, niou)).bool().numpy(), jt.zeros((0,)).numpy(), jt.zeros((0,)).numpy(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})
            
            # Assign all predictions as incorrect
            correct = jt.zeros((pred.shape[0], niou)).bool()
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                clss = jt.unique(tcls_tensor)
                for idx in range(clss.shape[0]):
                    cls = clss[idx]
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # target indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # prediction indices
                    
                    # print(ti,pi)
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        i,ious = box_iou(pred[pi, :4], tbox[ti]).argmax(1)  # best ious, indices

                        # Append detections
                        tmp_iou = (ious > iouv[0]).nonzero()
                        for idj in range(tmp_iou.shape[0]):
                            j = tmp_iou[idj]

                            d = ti[i[j]].item()  # detected target

                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.numpy(), pred[:, 4].numpy(), pred[:, 5].numpy(), tcls))

        # Plot images
        if batch_i < 1 and False:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

    print(s)
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    for s in stats:
        print('stats:',s.sum(),s.shape)
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = jt.zeros((1,))

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            cocoGt = COCO(glob.glob('data/coco/annotations/instances_val2014.json')[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(e)
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.numpy() / len(dataloader)).tolist()), maps


if __name__ == '__main__':
    jt.flags.use_cuda=1
    # jt.flags.merge_loop_mismatch_threshold = 1


    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.pt', help='weights path')
    parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)
    
    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')
