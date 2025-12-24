

import math

DEFAULT_K = 20

def expected_score(rating_a, rating_b):
    """Expected score of A vs B."""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

def update_elo(rating_a, rating_b, score_a, k=DEFAULT_K):
    """
    Update Elo ratings after a match.
    score_a: 1.0 = A win, 0.5 = draw, 0.0 = loss
    Returns new_rating_a, new_rating_b
    """
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    new_a = rating_a + k * (score_a - exp_a)
    new_b = rating_b + k * ((1.0 - score_a) - exp_b)
    return new_a, new_b

def batch_update(ratings, matches, k=DEFAULT_K):
    """
    Optionally process a batch of matches.
    ratings: dict {player_id: rating}
    matches: list of (player_a, player_b, outcome) where outcome is 1/0/0.5 for A
    Returns updated ratings (mutates copy).
    """
    ratings = ratings.copy()
    for a,b,out in matches:
        ra, rb = ratings.get(a,1500), ratings.get(b,1500)
        na, nb = update_elo(ra, rb, out, k=k)
        ratings[a], ratings[b] = na, nb
    return ratings

class EloTracker:
    def __init__(self, initial_elo=1000, k=32):
        self.elo = initial_elo
        self.k = k

    def update(self, win: bool):
        expected_score = 1 / (1 + 10 ** ((1000 - self.elo) / 400))
        score = 1 if win else 0
        self.elo += self.k * (score - expected_score)
        return self.elo
