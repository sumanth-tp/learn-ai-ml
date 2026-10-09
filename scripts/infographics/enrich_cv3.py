from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'cv-enrich'


def bars(board, x, y, rows, scale, gap=38, label_w=190, bar_w=250, fmt='{:.3f}'):
    for index, (name, value, colour) in enumerate(rows):
        yy = y + index * gap
        board.text(x, yy + 13, name, 14, None, '400', anchor='start')
        board.bar(x + label_w, yy, bar_w, value / scale, color=colour, h=16, label=fmt.format(value))


def classical_segmentation():
    board = Board(1160, 560, 'Which classical method survives a dimming belt?', 'Synthetic belt, 14 tablets, noise 10, five scenes per condition, IoU against the true mask')
    board.card(30, 96, 540, 40, 'IoU when the light fades to 45% across the belt', [], 'blue', size=14)
    bars(board, 40, 156, [
        ('fixed T=128', 0.317, 'red'),
        ('Otsu', 0.941, 'green'),
        ('adaptive 51', 0.977, 'green'),
        ('k-means gray', 0.933, 'green'),
        ('k-means gray+xy', 0.143, 'red'),
        ('GrabCut rectangle', 1.000, 'teal'),
    ], 1.0, bar_w=250)
    board.card(30, 410, 540, 120, 'Under even light the simplest method wins', [
        'fixed T=128, Otsu, k-means gray, GrabCut: IoU 1.000',
        'adaptive 51 is slightly worse: 0.987',
    ], 'green', size=13)
    board.card(610, 96, 520, 150, 'Feature scale decides what similar means', [
        'k-means with position weight 0.5',
        'IoU 0.129 on even light, 0.143 dimmed',
        'clusters follow image halves, not tablets',
    ], 'red', size=13)
    board.card(610, 276, 520, 254, 'Watershed does not rescue a rough mask', [
        'true tablets: 14 (two touching pairs)',
        'clean masks split to 14.0 regions',
        'dimmed belt: Otsu 19.0, grey k-means 19.2',
        'adaptive 15.2, GrabCut 14.0',
    ], 'yellow', size=13)
    return board


def mask_metrics():
    board = Board(1160, 600, 'What each mask metric hides', 'Rare class is 0.83% of a 128 by 128 label map; four predictions with a known error')
    board.card(30, 96, 560, 40, 'Pixel accuracy (grey) against rare-class IoU (red)', [], 'blue', size=14)
    rows = [('all background', 0.898, 0.000), ('rare class missed', 0.992, 0.000), ('rare shifted 2 px', 0.992, 0.360), ('5% random noise', 0.965, 0.301)]
    for index, (name, acc, iou) in enumerate(rows):
        yy = 160 + index * 62
        board.text(40, yy + 13, name, 14, None, '400', anchor='start')
        board.bar(200, yy, 340, acc, color='grey', h=16, label=f'{acc:.3f}')
        board.bar(200, yy + 24, 340, iou, color='red', h=16, label=f'{iou:.3f}')
    board.card(30, 420, 560, 150, 'The ranking flips', [
        'accuracy: missed 0.992 beats noise 0.965',
        'mIoU: noise 0.698 beats missed 0.664',
        'a 2 px shift scores the same accuracy as a miss',
    ], 'yellow', size=13)
    board.table(630, 96, [120, 170, 130], [
        ['bar width', 'IoU, 1 px shift', 'Dice'],
        ['1', '0.000', '0.000'],
        ['3', '0.500', '0.667'],
        ['9', '0.800', '0.889'],
        ['17', '0.889', '0.941'],
        ['33', '0.941', '0.970'],
    ], size=14)
    board.card(630, 340, 500, 230, 'The mean is not one number', [
        '20 squares of random size, shifted 2 px',
        'per-image mean IoU 0.705',
        'pooled IoU 0.807',
        'mean Dice 0.816, Dice from mean IoU 0.827',
    ], 'purple', size=13)
    return board


def detection_ap():
    board = Board(1160, 560, 'Suppression matters; its threshold barely does', 'SSDlite320 MobileNetV3, COCO weights, 60 Penn-Fudan images, 129 pedestrians, AP at IoU 0.5')
    board.card(30, 96, 540, 40, 'Average precision (envelope rule) by NMS setting', [], 'blue', size=14)
    bars(board, 40, 156, [
        ('no suppression', 0.428, 'red'),
        ('NMS 0.3', 0.920, 'green'),
        ('NMS 0.5', 0.919, 'green'),
        ('NMS 0.7', 0.915, 'green'),
    ], 1.0, bar_w=250)
    board.card(30, 330, 540, 200, 'False positives change far more than AP', [
        'no suppression: 7,338',
        'NMS 0.7: 3,203   NMS 0.5: 1,803   NMS 0.3: 946',
        'extra false positives rank low, so the area barely moves',
        'missed pedestrians: 4 at 0.3, 2 at 0.7',
    ], 'yellow', size=13)
    board.card(610, 96, 520, 170, 'Tighter matching costs about 0.06 AP', [
        'NMS 0.5: AP 0.919 at IoU 0.5, 0.857 at IoU 0.75',
        'boxes find people but are not tight enough',
    ], 'orange', size=13)
    board.card(610, 296, 520, 234, 'Two AP rules, two numbers', [
        'envelope (scratch) against step (scikit-learn)',
        'NMS 0.3, IoU 0.75: 0.859 against 0.870',
        'no suppression: 0.428 against 0.424',
        'my IoU and NMS match torchvision exactly on 400 random boxes',
    ], 'purple', size=13)
    return board


def tracking_idf1():
    board = Board(1160, 580, 'MOTA hides what IDF1 shows', 'Six crossing objects, 100 frames, a pole hides them for about 7 frames; means of five seeds, lifetime 12, no dropout')
    board.card(30, 96, 540, 40, 'IoU-only tracker against Kalman tracker', [], 'blue', size=14)
    for index, (name, mota, idf1) in enumerate([('IoU only', 0.854, 0.485), ('Kalman + IoU', 0.869, 0.907)]):
        yy = 160 + index * 80
        board.text(40, yy + 13, name, 14, None, '400', anchor='start')
        board.bar(190, yy, 250, mota, color='grey', h=16, label=f'MOTA {mota:.3f}')
        board.bar(190, yy + 26, 250, idf1, color='green' if index else 'red', h=16, label=f'IDF1 {idf1:.3f}')
    board.card(30, 340, 540, 190, 'Why MOTA barely moves', [
        'misses 46.2 and false positives 31.6 dominate',
        'one identity switch is one error, however long it lasts',
        'switches: 9.8 against 0.8',
    ], 'yellow', size=13)
    board.table(610, 96, [200, 90, 90, 90], [
        ['identity switches', 'drop 0', 'drop 0.2', 'drop 0.4'],
        ['IoU only, life 12', '9.8', '15.2', '35.6'],
        ['Kalman, life 3', '6.4', '7.6', '17.6'],
        ['Kalman, life 12', '0.8', '1.2', '3.0'],
    ], size=13)
    board.card(610, 280, 520, 120, 'Dropout is what MOTA measures', [
        'Kalman, life 12: MOTA 0.869 to 0.490, IDF1 0.907 to 0.661',
        'misses 46.2 to 268.4',
    ], 'red', size=13)
    board.card(610, 420, 520, 110, 'A longer lifetime needs prediction', [
        'IoU only: switches 8.2 to 9.8 at life 3 to 12',
        'Kalman: 6.4 to 0.8',
    ], 'green', size=13)
    return board


def edge_quantisation():
    board = Board(1160, 580, 'Smaller is not the same as faster or right', 'One Apple M3 Pro CPU thread, batch 1, 64 crops of six photographs, agreement is top-1 match with float32')
    board.table(30, 96, [250, 90, 90, 130], [
        ['variant', 'MB', 'ms', 'top-1 match'],
        ['ResNet-18 PyTorch fp32', '46.8', '13.2', '1.000'],
        ['ResNet-18 ONNX fp32', '46.7', '43.4', '1.000'],
        ['ResNet-18 ONNX int8 static', '11.7', '9.3', '0.875'],
        ['MobileNetV3-S PyTorch fp32', '10.3', '9.8', '1.000'],
        ['MobileNetV3-S ONNX fp32', '10.2', '4.8', '1.000'],
        ['MobileNetV3-S int8 linear', '5.4', '4.7', '1.000'],
        ['MobileNetV3-S int8 static', '2.7', '1.3', '0.000'],
    ], size=14)
    board.card(630, 96, 500, 150, 'Size follows the arithmetic', [
        '11.69 M parameters x 4 bytes = 46.76 MB',
        'x 1 byte = 11.69 MB, measured 11.7',
    ], 'green', size=13)
    board.card(630, 276, 500, 130, 'Runtime change is not a speed-up', [
        'ResNet-18: ONNX fp32 43.4 ms, PyTorch 13.2 ms',
        'MobileNetV3-S: ONNX 4.8 ms, PyTorch 9.8 ms',
    ], 'yellow', size=13)
    board.card(630, 436, 500, 110, 'PyTorch dynamic int8 covers linear layers only', [
        'ResNet-18 file 46.8 to 45.3 MB, no faster',
    ], 'purple', size=13)
    board.card(30, 400, 570, 146, 'Fastest and wrong', [
        'MobileNetV3-S static int8: 3.7x faster than ONNX fp32',
        'top-1 agreement 0.000 on 64 crops',
        'cause not diagnosed',
    ], 'red', size=13)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [
        ('v3-classical-segmentation', classical_segmentation),
        ('v3-mask-metrics', mask_metrics),
        ('v3-detection-ap', detection_ap),
        ('v3-tracking-idf1', tracking_idf1),
        ('v3-edge-quantisation', edge_quantisation),
    ]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
