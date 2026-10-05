from pathlib import Path
from board import Board

OUT = Path(__file__).resolve().parents[2] / 'static/img/dist'


def regression():
    b = Board(1120, 620, 'One objective, two aggregation choices', 'Synthetic linear and logistic regression; unequal shards of 3 and 9 rows')
    left = b.card(35, 115, 280, 150, 'Worker 0', ['3 rows', 'mean gradient g0', 'weight 3 / 12'], 'blue', size=16)
    right = b.card(805, 115, 280, 150, 'Worker 1', ['9 rows', 'mean gradient g1', 'weight 9 / 12'], 'orange', size=16)
    middle = b.card(390, 130, 340, 130, 'Weighted reduction', ['(3 g0 + 9 g1) / 12', 'same model version'], 'green', size=16)
    b.arrow(left.right(), middle.left())
    b.arrow(right.left(), middle.right())
    b.table(85, 325, [300, 300, 350], [['loss', 'weighted error', 'plain mean error'], ['linear', '0.000000000000', '0.411543'], ['logistic', '0.000000000000', '0.040139']], 'teal', size=15, row_h=48)
    b.card(85, 505, 950, 80, 'Lecture check: mean of [2, 4, 6, 8] = 5', ['A plain mean is correct when the shard means carry equal sample weight.'], 'purple', size=15)
    return b


def stale():
    b = Board(1120, 580, 'A correct gradient can arrive too late', 'Synthetic objective f(w) = 0.5 w²; initial w = 1; learning rate 0.4; 20 accepted updates')
    b.card(40, 120, 450, 130, 'Fresh update', ['evaluate at current w', 'w(next) = w - 0.4 w'], 'blue', size=17)
    b.card(630, 120, 450, 130, 'Delayed update', ['evaluate at older w', 'delay measured in accepted updates'], 'orange', size=17)
    b.table(110, 300, [200, 330, 350], [['delay', 'final weight', 'final loss'], ['0', '0.000037', '0.000000'], ['2', '0.002573', '0.000003'], ['3', '-0.661440', '0.218751']], 'red', size=16, row_h=45)
    b.text(560, 525, 'No claim about wall-clock speed or a universal safe delay.', size=17, color='grey')
    return b


def compression():
    b = Board(1120, 620, 'Compress the message, remember the remainder', 'Fixed eight-coordinate gradient; six transmissions; signed 8-bit magnitude rounding')
    cards = [b.card(30+i*275, 130, 235, 130, title, lines, colour, size=15) for i,(title,lines,colour) in enumerate([('Correct', ['gradient + residual'], 'blue'), ('Quantise', ['scale to integer range', 'round, then decode'], 'orange'), ('Transmit', ['decoded gradient', 'apply update'], 'green'), ('Remember', ['corrected - decoded', 'carry next time'], 'purple')])]
    for a,c in zip(cards,cards[1:]): b.arrow(a.right(),c.left())
    b.table(85, 325, [530, 420], [['checked quantity', 'value'], ['one-step maximum error', '0.007087'], ['six-step residual norm', '0.013588'], ['fp32 / 8-bit / plus scale bytes', '32 / 8 / 12'], ['sum sent + residual = desired sum', 'error 0.000000000000']], 'teal', size=15, row_h=43)
    b.text(560, 580, 'Payload arithmetic is not a measured network speedup.', size=17, color='grey')
    return b


def local():
    b = Board(1120, 640, 'Fewer rounds can change the optimiser', 'Equal-weight quadratic clients: curvature [1, 4], minima [-1, 2], central optimum 1.4')
    b.card(35, 115, 495, 125, 'Train locally', ['24 local steps per worker, learning rate 0.1', 'average models after each period'], 'blue', size=16)
    b.card(590, 115, 495, 125, 'Illustrative time model', ['10 ms per local step, 40 ms per sync', 'time = 240 + 40 × rounds'], 'orange', size=16)
    b.table(55, 290, [140, 150, 240, 250, 240], [['period', 'rounds', 'global weight', 'excess loss', 'modelled ms'], ['1', '24', '1.398595', '0.000002', '1200'], ['2', '12', '1.311143', '0.009869', '720'], ['4', '6', '1.146146', '0.080552', '480'], ['8', '3', '0.889560', '0.325686', '360']], 'teal', size=15, row_h=48)
    b.text(560, 595, 'Choose by time to an accepted quality target, not by rounds alone.', size=16, color='purple')
    return b


if __name__ == '__main__':
    for name, make in [('distributed-regression-weighted', regression), ('distributed-regression-staleness', stale), ('advanced-distributed-deep-learning-compression', compression), ('advanced-sgd-local-steps', local)]:
        print(make().save(OUT / (name+'.svg')))
