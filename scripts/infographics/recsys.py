from pathlib import Path

from board import Board


OUT = Path(__file__).resolve().parents[2] / 'static' / 'img' / 'recsys'


def feedback():
    board = Board(1160, 505, 'Recommendation starts with observed behaviour', 'A missing interaction is not a recorded dislike')
    board.card(30, 130, 340, 155, 'Explicit feedback', ['rating or stated preference', 'known scale and context', 'sparse and selected users'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Implicit feedback', ['view · click · purchase', 'count two means preference one', 'alpha two gives confidence five'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Cold start', ['new user: no history', 'new item: no exposure', 'use content and exploration'], 'green', size=17)
    board.card(210, 360, 740, 90, 'An impression records opportunity', ['without exposure logs, an absent click has several possible meanings'], 'yellow', size=17)
    return board


def factors():
    board = Board(1160, 505, 'Compress an interaction matrix into factors', 'Dot products score pairs, but hand-set factors are only an illustration')
    board.card(30, 130, 340, 155, 'Sparse matrix', ['users × items', 'observed ratings or events', 'missing entries need a policy'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Two latent dimensions', ['user = (1,1)', 'item A = (2,1) → score 3', 'item B = (0,2) → score 2'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Training objective', ['fit observed signals', 'regularise sparse factors', 'weight implicit confidence'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Cold entities have no fitted factor', ['metadata or a controlled popularity fallback must supply a path'], 'yellow', size=17)
    return board


def retrieval():
    board = Board(1160, 505, 'Retrieve broadly, then score deeply', 'The ranker cannot rescue a relevant item absent from candidates')
    board.card(30, 130, 340, 155, 'Query tower', ['user history and context', 'one query embedding', 'computed per request'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Item tower and index', ['precomputed item embeddings', 'ANN retrieves a shortlist', 'toy top two: A,B'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Rank and rerank', ['richer user-item features', 'eligibility and diversity', 'serve a small slate'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Candidate recall sets an upper bound', ['toy top-two finds one of two labelled relevant items: recall 0.5'], 'yellow', size=17)
    return board


def evaluation():
    board = Board(1160, 505, 'Measure a slate and its feedback loop', 'Offline relevance and online user outcomes answer different questions')
    board.card(30, 130, 340, 155, 'Offline replay', ['temporal split · exposure set', 'Recall@K and NDCG@K', 'slices for new/rare items'], 'blue', size=17)
    board.card(410, 130, 340, 155, 'Online experiment', ['randomised assignment', 'retention and satisfaction', 'latency and safety guardrails'], 'purple', size=17)
    board.card(790, 130, 340, 155, 'Feedback loop', ['position affects clicks', 'recommendations change exposure', 'log impressions and policy'], 'green', size=17)
    board.card(210, 360, 740, 90, 'Illustrative slate', ['A,B share topic X; bonus 0.2 gives A,C and two topics'], 'yellow', size=17)
    return board


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in [('feedback-objective', feedback), ('matrix-factors', factors), ('retrieval-ranking', retrieval), ('evaluation-loops', evaluation)]:
        maker().save(OUT / f'{name}.svg')


if __name__ == '__main__':
    main()
