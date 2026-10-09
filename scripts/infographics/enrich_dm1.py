from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'dm-enrich'
W = 1160


def stack(board, x, y, w, cards, gap=16):
    for title, lines, colour, size in cards:
        box = board.card(x, y, w, None, title, lines, colour, size=size)
        y += box.h + gap
    return y


def bars(board, x, y, rows, scale, label_w=250, bar_w=230, gap=36, fmt='{:.3f}'):
    for index, (name, value, colour) in enumerate(rows):
        yy = y + index * gap
        board.text(x, yy + 13, name, 13, None, '400', anchor='start')
        board.bar(x + label_w, yy, bar_w, value / scale, color=colour, h=16, label=fmt.format(value))
    return y + len(rows) * gap


def build(maker):
    probe = maker(1600)
    return maker(int(probe[1]) + 24)[0]


def representations_measured(height):
    board = Board(W, height, 'Measured: CSV against Parquet on 2,000,000 rows', 'Synthetic table, 20 columns, one run; sizes are exact, timings move between runs')
    y = 110
    board.text(30, y, 'File size in MB (bar = CSV, scale 673)', 14, None, '700', anchor='start')
    y = bars(board, 30, y + 14, [('20 columns, CSV', 673, 'orange'), ('20 columns, Parquet', 285, 'green'),
                                  ('5 compressible, CSV', 84, 'orange'), ('5 compressible, Parquet', 19, 'green')], 673, fmt='{:.0f}')
    y = stack(board, 30, y + 10, 540, [
        ('Ratio is a property of the data', ['20 columns of which 15 are random floats: 2.36x', '5 realistic columns: 4.42x', 'the chapter example assumed 5x'], 'yellow', 13),
    ])
    top = 110
    top = stack(board, 610, top, 520, [
        ('Two of 20 columns, group by', ['CSV 0.380 s, Parquet 0.004 s: 107x', 'four runs gave 72x to 109x', 'most of the gap is CSV parsing'], 'blue', 13),
        ('Inside Parquet only', ['16 columns summed 0.052 s', '2 columns 0.004 s: 14.6x'], 'teal', 13),
        ('Sorted against shuffled, one-day filter', ['row groups that can match: 1 of 8 against 8 of 8', 'query time 0.0031 s against 0.0050 s', 'small at this size, larger on big data'], 'purple', 13),
    ])
    return board, max(y, top)


def quality_rule_recall(height):
    board = Board(W, height, 'What each validation rule caught', '20,000 synthetic orders, 200 injected defects of each of 5 types, pandera 0.34.0')
    board.table(30, 110, [150, 90, 110, 90], [
        ['rule', 'flagged', 'precision', 'recall'],
        ['not_nullable', '200', '1.000', '1.000'],
        ['ge(0)', '200', '1.000', '1.000'],
        ['le(1000)', '180', '1.000', '0.900'],
        ['isin', '200', '1.000', '1.000'],
        ['unique', '396', '0.500', '0.990'],
    ], size=14)
    y = stack(board, 30, 340, 440, [
        ('Range rule misses legal-looking errors', ['pounds stored as pence: 20 of 200 stayed under 1000', 'log-outlier rule: 186 of 200, 240 false alarms'], 'red', 13),
    ])
    top = stack(board, 620, 110, 510, [
        ('Uniqueness flags every copy', ['396 rows flagged for 200 duplicated keys', 'precision 0.500, recall 0.990'], 'yellow', 13),
        ('Coercing to int hides truncation', ['mean 40.061 becomes 39.567', '99.1% of rows changed, no error'], 'orange', 13),
        ('Parsing with errors=coerce', ['3.1% become NaN', 'sum 777,725 against 801,225'], 'purple', 13),
    ])
    return board, max(y, top)


def manifest_vs_listing(height):
    board = Board(W, height, 'A table is its manifest, not its directory', 'Three commits and one unfinished write, Parquet files and JSON manifests')
    y = stack(board, 30, 110, 540, [
        ('Read through the latest manifest', ['v_3 lists a_fixed and b', '5 rows, total 155'], 'green', 15),
        ('Snapshots stay readable', ['v_1: 3 rows, 60', 'v_2: 5 rows, 150', 'v_3: 5 rows, 155'], 'blue', 13),
    ])
    top = stack(board, 610, 110, 520, [
        ('Read the directory listing', ['a (3 rows, 60) superseded, a_fixed (3, 65), b (2, 90)', 'c_half_written (1, 999) never committed', '9 rows, total 1,214: 7.8x too large'], 'red', 13),
        ('Schema changes', ['added column: old rows read as null', 'text in a number column: ArrowInvalid at read time'], 'yellow', 13),
        ('Small files', ['300 files of 100 rows: 0.0099 s', 'one compacted file: 0.0003 s, 29x (one run)'], 'purple', 13),
    ])
    return board, max(y, top)


def retry_duplicates(height):
    board = Board(W, height, 'Retries duplicate rows unless the write is idempotent', '100 days of 50 events, DuckDB sink, acknowledgements lost after the write')
    board.text(30, 112, 'Duplicate rows, ack lost 10% (scale 2,100)', 14, None, '700', anchor='start')
    y = bars(board, 30, 126, [('append', 800, 'red'), ('upsert', 0, 'green'), ('replace-day', 0, 'green')], 2100, label_w=150, fmt='{:.0f}')
    board.text(30, y + 14, 'Duplicate rows, ack lost 30%', 14, None, '700', anchor='start')
    y = bars(board, 30, y + 28, [('append', 2100, 'red'), ('upsert', 0, 'green'), ('replace-day', 0, 'green')], 2100, label_w=150, fmt='{:.0f}')
    y = stack(board, 30, y + 10, 540, [
        ('Revenue is overstated', ['append total 350,753, upsert total 247,090', 'overstated by 42.0%, every task green'], 'orange', 13),
    ])
    top = stack(board, 610, 110, 520, [
        ('Little\'s Law at 80% load', ['arrival x mean stay = 4.2 in flight', 'counted directly: 4.4'], 'green', 13),
        ('Little\'s Law at 120% load', ['mean stay 30.9 s, 56.8 s at the end', 'formula 3,117, counted 2,580.5', 'averages are not stable, so it fails'], 'red', 13),
    ])
    return board, max(y, top)


def alert_noise(height):
    board = Board(W, height, 'Three alert rules on a lagging feed', '20 simulated months, SLO 99% of minutes under 15 min lag, pages grouped after 120 quiet minutes')
    board.table(30, 110, [250, 140, 130, 170, 150, 160], [
        ['rule', 'pages/month', 'idle pages', 'outages caught', 'median delay', 'creep caught'],
        ['any bad minute', '66.3', '94%', '100%', '0 min', '100%'],
        ['5 bad minutes in a row', '4.0', '1%', '100%', '4 min', '30%'],
        ['burn rate 14.4 / 6', '5.7', '17%', '100%', '8 min', '100%'],
    ], size=14)
    y = stack(board, 30, 330, 540, [
        ('Error budget', ['99% of 43,200 minutes allows 432 bad minutes', 'burn rate = bad share / 1% allowed'], 'blue', 14),
    ])
    top = stack(board, 610, 330, 520, [
        ('Neither rule wins everywhere', ['run rule: quiet and quick on outages', 'but catches only 30% of creeping faults', 'burn rate catches all, with more idle pages'], 'yellow', 14),
    ])
    return board, max(y, top)


def leakage_and_skew(height):
    board = Board(W, height, 'Leakage inflates scores, skew hides in AUC', 'scikit-learn 1.9.1; noise features, synthetic customers and the breast cancer data')
    board.text(30, 112, 'AUC (bar = AUC out of 1.0)', 14, None, '700', anchor='start')
    y = bars(board, 30, 126, [('noise, selected on all rows', 0.946, 'red'), ('noise, selected in folds', 0.526, 'green'),
                               ('random row split', 0.908, 'red'), ('split by customer', 0.570, 'green'),
                               ('scaler on all rows', 0.9953, 'blue'), ('scaler in folds', 0.9952, 'blue')], 1.0, label_w=230, bar_w=210, fmt='{:.3f}')
    y = stack(board, 30, y + 10, 540, [
        ('Label-aware leaks are the large ones', ['the scaler leak changed AUC by 0.0001', 'selection on all rows lifted noise to 0.946'], 'yellow', 13),
    ])
    board.text(610, 112, 'Four-feature model served with a mistake', 14, None, '700', anchor='start')
    t = board.table(610, 126, [250, 120, 150], [
        ['serving', 'AUC', 'accuracy'],
        ['as trained', '0.9831', '0.930'],
        ['unit x0.1', '0.9724', '0.626'],
        ['zero filled', '0.9591', '0.690'],
        ['frozen at median', '0.9591', '0.877'],
        ['columns swapped', '0.9642', '0.819'],
    ], size=13)
    top = stack(board, 610, t.y + t.h + 16, 520, [
        ('Redundancy protects a wide model', ['30 features: AUC 0.9953 to 0.9961, accuracy 0.947 at worst', 'check accuracy at the threshold, not only AUC'], 'green', 13),
    ])
    return board, max(y, top)


def late_data_ingestion(height):
    board = Board(W, height, 'Late data and incremental loads', '280,000 events over 14 days, 3.0% arrive more than an hour late, synthetic')
    t = board.table(30, 110, [270, 140, 130], [
        ['strategy', 'rows read', 'kept'],
        ['full reload', '2,652,720', '100.00%'],
        ['arrival-time watermark', '280,000', '100.00%'],
        ['event-time watermark', '273,226', '97.58%'],
        ['event-time, 6 h lookback', '347,549', '98.11%'],
        ['event-time, 24 h lookback', '571,219', '99.30%'],
        ['event-time, 72 h lookback', '1,111,705', '100.00%'],
    ], size=13)
    y = stack(board, 30, t.y + t.h + 16, 540, [
        ('First publication is 2.29% short for every strategy', ['those events had not reached the source yet', 'publish as provisional and restate later'], 'yellow', 13),
    ])
    top = stack(board, 610, 110, 520, [
        ('A 0.25% sample of a 0.3% rare label', ['5,000 of 2,000,000: 15.0 rare rows on average', 'range 3 to 25, under 10 in 9% of draws', 'rate 95% of draws between 0.0016 and 0.0046'], 'blue', 13),
        ('Stratified with weights', ['500 rare rows, raw rate 0.100', 'weighted rate 0.0030'], 'green', 13),
        ('First 5,000 rows of a time-ordered table', ['mean of a drifting field 0.003 against 0.500'], 'red', 13),
    ])
    return board, max(y, top)


def psi_vs_ks(height):
    board = Board(W, height, 'PSI and KS answer different questions', '200 batches per cell, 10 deciles from a 20,000-row reference, normal data')
    t = board.table(30, 110, [210, 90, 120, 130, 120], [
        ['shift', 'n', 'PSI>0.1', 'PSI>0.25', 'KS p<0.05'],
        ['none', '200', '4%', '0%', '6%'],
        ['mean 0.25', '200', '52%', '0%', '80%'],
        ['mean 0.5', '2000', '100%', '30%', '100%'],
        ['mean 0.5', '20000', '100%', '3%', '100%'],
        ['wider 1.3x', '20000', '2%', '0%', '100%'],
        ['2% outliers at 6', '2000', '0%', '0%', '25%'],
        ['2% outliers at 6', '20000', '0%', '0%', '100%'],
    ], size=13)
    y = t.y + t.h + 16
    top = stack(board, 700, 110, 430, [
        ('No change, batch of 100', ['median PSI 0.087, 99th percentile 0.260', 'above 0.1 in 38.0% of batches'], 'red', 13),
        ('20 unchanged features, KS at 0.05', ['64% of days raise an alert', 'threshold 0.05 / 20: 2% of days'], 'orange', 13),
        ('PSI is about the shift squared', ['0.5 standard deviations: PSI near 0.235', 'noise floor about 9 x (1/n + 1/m)'], 'green', 13),
    ])
    return board, max(y, top)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [('representations-measured', representations_measured), ('quality-rule-recall', quality_rule_recall),
                        ('manifest-vs-listing', manifest_vs_listing), ('retry-duplicates', retry_duplicates),
                        ('alert-noise', alert_noise), ('leakage-and-skew', leakage_and_skew),
                        ('late-data-ingestion', late_data_ingestion), ('psi-vs-ks', psi_vs_ks)]:
        build(maker).save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
