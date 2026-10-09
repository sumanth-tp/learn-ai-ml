from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'cv-enrich'


def bars(board, x, y, rows, gap=34, label_w=190, bar_w=230):
    for index, (name, value, colour) in enumerate(rows):
        yy = y + index * gap
        board.text(x, yy + 13, name, 13, None, '400', anchor='start')
        board.bar(x + label_w, yy, bar_w, value, color=colour, h=16, label=f'{value:.3f}')


def harris_hog():
    board = Board(1180, 600, 'Harris repeats until the image is noisy or rescaled', 'Camera image, 400 reference corners, match within 2 px; digits, 1,078 train and 719 test images')
    board.card(30, 95, 560, 36, 'Share of reference corners found again', [], 'blue', size=13)
    bars(board, 40, 145, [
        ('rotate 90 (exact)', 1.000, 'green'),
        ('rotate 5 degrees', 0.795, 'teal'),
        ('rotate 45 degrees', 0.745, 'teal'),
        ('noise 5 grey levels', 0.559, 'orange'),
        ('noise 20 grey levels', 0.258, 'red'),
        ('scale 0.9', 0.665, 'orange'),
        ('scale 0.5', 0.140, 'red'),
        ('scale 2.0', 0.326, 'red'),
    ])
    board.card(30, 440, 560, 130, 'Rotation loss is resampling, not geometry', [
        '90 degrees only permutes pixels: 1.000',
        '5 degrees already loses 20%',
        'noise of 2% of the range halves it',
    ], 'yellow', size=13)
    board.text(880, 108, 'Digit accuracy, linear SVM', 15, 'purple', '700')
    board.table(620, 125, [130, 70, 70, 90, 70, 70], [
        ['features', 'clean', 'shift\n1 px', 'contrast\nx0.3', 'noise\n0.1', 'noise\n0.2'],
        ['raw pixels', '0.971', '0.446', '0.690', '0.960', '0.908'],
        ['HoG cell 2', '0.950', '0.592', '0.950', '0.894', '0.734'],
        ['HoG cell 4', '0.840', '0.599', '0.840', '0.789', '0.630'],
    ], header_color='purple', size=13)
    board.card(620, 330, 530, 240, 'What the table says', [
        'HoG ignores a contrast change: 0.950 stays 0.950',
        'raw pixels fall from 0.971 to 0.690',
        'but gradients amplify noise: 0.734 against 0.908',
        'raw pixels win on clean digits',
        'neither survives a 1 px shift',
    ], 'purple', size=13)
    return board


def sift_ratio():
    board = Board(1180, 600, 'The ratio test trades a few good matches for clean ones', 'Camera image rotated 30 degrees, scaled 0.7 and dimmed; a match is correct within 3 px of the known mapping')
    board.table(30, 100, [150, 100, 100, 110, 120], [
        ['ratio threshold', 'kept', 'correct', 'precision', 'trials for 99%'],
        ['0.8', '244', '215', '0.88', '5'],
        ['0.9', '346', '221', '0.64', '26'],
        ['1.0 (no test)', '791', '226', '0.29', '689'],
    ], header_color='teal', size=14)
    board.card(30, 300, 580, 270, 'Reading the left table', [
        'the test removes 547 matches',
        'only 11 of them were correct',
        'the homography fits equally well without it',
        'worst corner error 0.42 px at 0.8',
        'worst corner error 0.22 px with no test',
        'the saving is in RANSAC trials: 689 down to 5',
    ], 'teal', size=13)
    board.text(900, 108, 'Precision by threshold', 15, 'blue', '700')
    board.table(650, 125, [170, 70, 70, 70, 70, 70], [
        ['case', '0.6', '0.7', '0.8', '0.9', '1.0'],
        ['brightness', '1.00', '0.99', '0.97', '0.83', '0.50'],
        ['rotate 30', '0.99', '0.98', '0.97', '0.88', '0.60'],
        ['scale 0.5', '0.96', '0.89', '0.81', '0.55', '0.23'],
        ['all three', '0.98', '0.96', '0.88', '0.64', '0.29'],
        ['brick, all three', '0.92', '0.88', '0.79', '0.62', '0.36'],
    ], header_color='blue', size=13)
    board.card(650, 400, 500, 170, 'Repeated texture is harder', [
        'brick at 0.8: precision 0.79, 285 of 316 correct kept',
        'camera at 0.8: precision 0.88, 215 of 226 kept',
    ], 'yellow', size=13)
    return board


def ransac_trials():
    board = Board(1180, 620, 'The 17 trials are right and still not enough', 'Line with 50% outliers, 1,000 points; homography with 300 correspondences and 1 px noise')
    board.table(30, 100, [90, 110, 150, 150, 130], [
        ['trials', 'formula', 'all-inlier\nsample found', 'line recovered', 'least\nsquares'],
        ['8', '0.8999', '0.8983', '0.771', '0.000'],
        ['16', '0.9900', '0.9898', '0.947', '0.000'],
        ['17', '0.9925', '0.9925', '0.953', '0.000'],
        ['34', '0.9999', '1.0000', '0.999', '0.000'],
    ], header_color='orange', size=13)
    board.card(30, 330, 630, 260, 'What the 17 does and does not promise', [
        '16 trials give 0.9898, 17 give 0.9925',
        'the formula predicts the sampling event exactly',
        'a good sample is not a good line:',
        '17 trials recover the line 0.953 of the time',
        'about 34 trials were needed for 0.999',
    ], 'orange', size=13)
    board.text(900, 108, 'Median homography error (px)', 15, 'blue', '700')
    board.table(700, 125, [100, 90, 90, 90, 90], [
        ['outliers', 'least\nsquares', 'RANSAC', 'LMEDS', 'RHO'],
        ['0%', '0.15', '0.27', '0.15', '0.21'],
        ['10%', '19.35', '0.29', '0.16', '0.22'],
        ['30%', '58.64', '0.27', '0.17', '0.31'],
        ['50%', '184.09', '0.27', '4.13', '0.35'],
        ['70%', '190.72', '0.32', '188.69', '0.45'],
    ], header_color='blue', size=13)
    board.card(700, 400, 450, 190, 'Four-point model, 99% target', [
        'w=0.5: formula 72 trials, success 88%',
        'w=0.3: formula 567 trials, success 86%',
        'double the trials: 96% and 97%',
    ], 'yellow', size=13)
    return board


def classification():
    board = Board(1180, 620, 'No model wins everywhere', 'Fashion-MNIST, 10 classes, 3 random draws of the training set, 2,000 test images')
    board.text(300, 108, 'Clean test accuracy by training size', 15, 'blue', '700')
    board.table(30, 125, [90, 130, 130, 110, 130], [
        ['images', 'raw +\nlogistic', 'HoG +\nSVM', 'small\nCNN', 'ResNet18\n+ logistic'],
        ['50', '0.660', '0.667', '0.654', '0.646'],
        ['100', '0.731', '0.739', '0.700', '0.727'],
        ['200', '0.745', '0.770', '0.768', '0.753'],
        ['500', '0.779', '0.791', '0.797', '0.787'],
    ], header_color='blue', size=13)
    board.card(30, 370, 590, 210, 'On clean data the four are within 2 points', [
        'at 500 images: 0.779 to 0.797',
        'the 20,490-parameter CNN matches ResNet18',
        'features, which have 512 inputs from 11.2M weights',
        'raw pixels with logistic regression is a real baseline',
    ], 'blue', size=13)
    board.text(900, 108, 'Trained on 500 images, tested on changed images', 15, 'red', '700')
    board.table(650, 125, [130, 100, 90, 90, 100], [
        ['test set', 'raw', 'HoG', 'CNN', 'ResNet18'],
        ['clean', '0.779', '0.791', '0.797', '0.787'],
        ['shift 3 px', '0.287', '0.348', '0.288', '0.717'],
        ['contrast x0.4', '0.365', '0.791', '0.455', '0.729'],
        ['noise 0.2', '0.758', '0.339', '0.741', '0.289'],
    ], header_color='red', size=13)
    board.card(650, 370, 500, 210, 'A different winner for each change', [
        'shift: pretrained features',
        'contrast: HoG, unchanged at 0.791',
        'noise: raw pixels and the CNN',
        'only the clean column looks alike',
    ], 'red', size=13)
    return board


def bovw():
    board = Board(1180, 620, 'Bigger vocabularies help until they stop being vocabularies', '12 scenes, 36 gallery and 60 query views, 3 seeds, top-1 retrieval accuracy')
    board.table(30, 100, [210, 60, 60, 60, 60, 70, 70], [
        ['codebook, weighting', '4', '16', '64', '256', '1024', '4096'],
        ['gallery, tf', '0.37', '0.64', '0.64', '0.73', '0.79', '1.00'],
        ['gallery, tf-idf', '0.37', '0.64', '0.66', '0.78', '0.87', '1.00'],
        ['gallery+queries, tf-idf', '0.35', '0.58', '0.69', '0.77', '0.89', '0.99'],
        ['unrelated photos, tf-idf', '0.36', '0.48', '0.59', '0.62', '0.62', '0.64'],
    ], header_color='green', size=13)
    board.card(30, 330, 600, 250, 'Reading the table', [
        'tf-idf adds 0.08 at 1024 words',
        'adding the queries to k-means adds 0.02',
        'a codebook from unrelated photos stalls at 0.64',
        '4096 words for about 5,000 descriptors is a lookup table',
    ], 'green', size=13)
    board.text(900, 108, 'Direct SIFT voting, top-1', 15, 'purple', '700')
    bars(board, 660, 135, [('ratio 0.6', 1.000, 'green'), ('ratio 0.7', 0.889, 'teal'), ('ratio 0.8', 0.700, 'orange')], label_w=100, bar_w=250)
    board.card(670, 230, 480, 320, 'The threshold moves more than the vocabulary', [
        'one ratio threshold swings accuracy',
        'from 1.000 to 0.700',
        'no codebook of 64 to 1024 words',
        'reaches the 1.000 of ratio 0.6',
        'direct matching needs every gallery',
        'image at query time; the histogram',
        'index does not',
    ], 'purple', size=13)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [('v2-harris-hog', harris_hog), ('v2-sift-ratio', sift_ratio), ('v2-ransac-trials', ransac_trials), ('v2-classification-shifts', classification), ('v2-bovw-vocabulary', bovw)]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
