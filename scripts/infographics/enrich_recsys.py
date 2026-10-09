from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'recsys-enrich'


def bars(board, x, y, rows, scale, gap=40, label_w=215, bar_w=250, fmt='{:.3f}'):
    for index, (name, value, colour) in enumerate(rows):
        yy = y + index * gap
        board.text(x, yy + 13, name, 14, None, '400', anchor='start')
        board.bar(x + label_w, yy, bar_w, value / scale, color=colour, h=16, label=fmt.format(value))


def explicit_vs_implicit():
    board = Board(1160, 520, 'Same data, different objective, different list', 'MovieLens 100K, last 20% of each user held out; one run, seed 0')
    board.card(30, 100, 560, 66, 'Precision@10 of items users later rated 4 or 5', [], 'blue', size=14)
    bars(board, 50, 190, [
        ('popularity', 0.079, 'grey'),
        ('raw item mean rating', 0.001, 'red'),
        ('explicit ALS', 0.049, 'orange'),
        ('implicit ALS', 0.100, 'green'),
    ], 0.12, bar_w=170)
    board.card(30, 360, 560, 130, 'The rating model is more accurate and worse at ranking', [
        'RMSE: explicit ALS 0.999, item-mean 1.074',
        'precision@10: explicit ALS 0.049 vs popularity 0.079',
        'implicit ALS covers 0.432 of the catalogue',
    ], 'yellow', size=13)
    board.card(630, 100, 500, 175, 'Position bias in a simulated log', [
        '200 items, 10 slots, examination 1 / position',
        'naive click rate vs true appeal: 0.787',
        'divided by exposure: 0.931',
        'true top 20 found: 12 naive, 15 corrected',
    ], 'purple', size=13)
    board.card(630, 305, 500, 185, 'Exposure also hides items', [
        '100 of 200 items were never shown',
        '3 of the true top 20 among them',
        'no correction can score an item with zero impressions',
    ], 'red', size=13)
    return board


def cf_baselines():
    board = Board(1160, 540, 'Neighbours and factors against popularity', 'MovieLens 100K, per-user time split, precision@10 on items rated 4 or 5')
    rows = [('popularity', 0.079, 'grey'), ('user-kNN k=50', 0.126, 'green'), ('item-kNN k=50', 0.121, 'green'),
            ('SVD d=10', 0.126, 'green'), ('SVD d=50', 0.120, 'teal'), ('SVD d=200', 0.076, 'red')]
    bars(board, 40, 110, rows, 0.15, bar_w=260)
    board.card(40, 370, 500, 140, 'More factors is not better', [
        'd=200 reproduces the training matrix',
        'precision falls to 0.076, below popularity',
        'but coverage rises to 0.483',
    ], 'yellow', size=13)
    board.card(640, 110, 490, 225, 'Gain over popularity by history length', [
        'under 25 train items: 0.039 to 0.065',
        '25 to 59: 0.057 to 0.098',
        '60 or more: 0.111 to 0.173',
        '(popularity to SVD d=10, 196, 286, 425 users)',
    ], 'blue', size=13)
    board.card(640, 365, 490, 145, 'Catalogue coverage of the top 10 lists', [
        'popularity 0.043, user-kNN 0.178',
        'item-kNN 0.259, SVD d=10 0.213',
    ], 'purple', size=13)
    return board


def funnel_depth():
    board = Board(1160, 560, 'Deeper candidate lists raise recall, not the final slate', 'Two-tower retrieval then a boosted-tree ranker, MovieLens 100K, one run')
    board.card(30, 100, 540, 160, 'Retrieval recall at depth 100', [
        'popularity 0.376',
        'two-tower, plain in-batch negatives 0.302',
        'two-tower, logQ corrected 0.519',
    ], 'blue', size=14)
    board.table(30, 290, [90, 200, 220], [
        ['depth', 'candidate recall', 'P@10 after ranker'],
        ['10', '0.124', '0.099'],
        ['25', '0.233', '0.112'],
        ['50', '0.356', '0.115'],
        ['100', '0.519', '0.113'],
        ['200', '0.698', '0.109'],
        ['400', '0.866', '0.099'],
    ], size=14)
    board.card(620, 100, 510, 160, 'Approximate search at 100,000 items', [
        'IVF nprobe=16: recall 0.934',
        'HNSW efSearch=128: recall 0.728',
        'exact search is the reference',
    ], 'purple', size=14)
    board.card(620, 290, 510, 200, 'Genre re-ranking trades relevance', [
        'lambda 0.00: P@10 0.115, 9.57 genres',
        'lambda 0.02: P@10 0.114, 10.78 genres',
        'lambda 0.05: P@10 0.107, 12.04 genres',
        'lambda 0.10: P@10 0.100, 12.76 genres',
    ], 'green', size=14)
    return board


def loop_and_ips():
    board = Board(1160, 560, 'Feedback loops narrow the catalogue; IPS needs overlap', 'Simulated users, 40 rounds and 200 repeated logs; means over seeds')
    board.card(30, 100, 540, 40, 'Distinct items shown in round 40 (of 300)', [], 'blue', size=14)
    bars(board, 40, 165, [('popularity', 5, 'red'), ('greedy MF', 38, 'orange'), ('MF + 10% exploration', 203, 'green')], 300, gap=44, label_w=230, bar_w=240, fmt='{:.0f}')
    board.card(30, 310, 540, 190, 'Exposure concentration', [
        'top 10 items share of exposure:',
        'popularity 0.976, greedy MF 0.393',
        'exploration 0.392',
        'Gini: 0.963, 0.826, 0.757',
    ], 'yellow', size=14)
    board.table(610, 100, [230, 100, 100, 90], [
        ['estimator, log temp 0.03', 'mean', 'bias', 'std'],
        ['truth 0.6516', '', '', ''],
        ['naive logged mean', '0.2701', '-0.3815', '0.0091'],
        ['IPS', '0.4667', '-0.1849', '1.0274'],
        ['IPS clipped at 10', '0.2183', '-0.4333', '0.0165'],
        ['self-normalised IPS', '0.7254', '+0.0738', '0.1341'],
        ['direct method', '0.1454', '-0.5062', '0.0111'],
    ], size=13)
    board.card(610, 400, 520, 100, 'At temperature 1.0 plain IPS is nearly unbiased', ['bias -0.0047, std 0.0549; clipping adds bias -0.0829'], 'green', size=13)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [('explicit-vs-implicit', explicit_vs_implicit), ('cf-baselines', cf_baselines), ('funnel-depth', funnel_depth), ('loop-and-ips', loop_and_ips)]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
