from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'timeseries'


def temporal_foundations():
    board = Board(1160, 505, 'Time order is part of the data', 'Forecast only from information that existed at the origin')
    board.card(30, 130, 340, 155, 'Read the pattern', ['level · trend · season', 'calendar and missing timestamps', 'changing variance or regime'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Cut time once', ['train t0–t5', 'hold out t6,t7 = 11,13', 'period-four guess = 11,13'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Prevent leakage', ['fit transforms on past only', 'know covariates by origin', 'score each horizon separately'], 'green', size=17)
    board.card(210, 360, 740, 90, 'A random shuffle changes the question', ['future neighbours and regime clues can enter the training set'], 'yellow', size=17)
    return board


def classical_forecasts():
    board = Board(1160, 505, 'Classical forecasts make their structure explicit', 'A strong baseline comes before extra parameters')
    board.card(30, 130, 340, 155, 'Baselines', ['last value: 13', 'sample mean: 11.5', 'seasonal last cycle if relevant'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Smoothed level', ['data 10,12,11,13', 'alpha 0.5', 'next level forecast 12'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Structured models', ['ETS level/trend/season', 'ARIMA differences and errors', 'SARIMAX with future covariates'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Compare at the same origins', ['residual fit is not a held-out forecast score'], 'yellow', size=17)
    return board


def lagged_learning():
    board = Board(1160, 505, 'Turn past observations into a training row', 'Every feature must have a timestamp earlier than its target')
    board.card(30, 130, 340, 155, 'One row for t3', ['history 10,12,11', 'lag1=11 · lag2=12', 'prior-three mean=11'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Fit a tabular model', ['trees handle nonlinear rules', 'one model across many series', 'known future covariates only'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Forecast horizon', ['recursive: feed predictions back', 'direct: separate horizon output', 'backtest each origin'], 'green', size=17)
    board.card(210, 360, 740, 90, 'A rolling window must end before its target', ['an unshifted rolling mean can leak the answer into its own row'], 'yellow', size=17)
    return board


def pretrained_forecasts():
    board = Board(1160, 505, 'Pretrained models still need a valid time contract', 'Patching and covariates do not excuse future-target leakage')
    board.card(30, 130, 340, 155, 'Observed context', ['10,12 | 11,13 | 10,12', 'three width-two patches', 'normalise from past only'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Hidden horizon', ['two future targets masked', 'planned events may be known', 'unknown future traffic is not'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Compare fairly', ['TimesFM-3 · Chronos-2', 'zero-shot is a protocol', 'naive and local baselines remain'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Model family is not a deployment result', ['measure local accuracy, intervals, compute and data availability'], 'yellow', size=17)
    return board


def forecast_evaluation():
    board = Board(1160, 505, 'Evaluate forecasts as decisions made in time', 'Backtesting, scale and uncertainty answer different questions')
    board.card(30, 130, 340, 155, 'Rolling origins', ['fit through each cutoff', 'predict a fixed horizon', 'compare to later observations'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Scaled error', ['train differences 2,1,2', 'naive scale 5/3', 'test MAE 1 → MASE 0.6'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Intervals and alerts', ['centre 13 ± radius 2', 'illustration [11,15]', 'check coverage and false alerts'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Production needs a feedback loop', ['late data, revisions, holidays and drift alter the forecast contract'], 'yellow', size=17)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [
        ('temporal-foundations', temporal_foundations),
        ('classical-forecasts', classical_forecasts),
        ('lagged-learning', lagged_learning),
        ('pretrained-forecasts', pretrained_forecasts),
        ('forecast-evaluation', forecast_evaluation),
    ]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
