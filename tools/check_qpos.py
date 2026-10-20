"""Deterministic indexing illustration, not a new scientific measurement."""
import numpy as np


def qpos(hidden, permutations):
    n, length, _ = hidden.shape
    assert permutations.shape == (length, n)
    for p in permutations:
        assert np.array_equal(np.sort(p), np.arange(n))
    return np.stack([hidden[permutations[t], t] for t in range(length)], axis=1)


def main():
    h = np.arange(5 * 4 * 3, dtype=np.int64).reshape(5, 4, 3)
    p = np.stack([np.roll(np.arange(5), t) for t in range(4)])
    q = qpos(h, p)
    for t in range(4):
        assert sorted(map(tuple, h[:, t])) == sorted(map(tuple, q[:, t]))
        for j in range(5):
            assert np.array_equal(q[j, t], h[p[t, j], t])
    assert not np.array_equal(h, q)
    invalid = p.copy(); invalid[1, 0] = invalid[1, 1]
    try:
        qpos(h, invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("Invalid donor permutation accepted")
    print("Complete-vector finite marginals and donor indexing verified; invalid permutations rejected.")


if __name__ == "__main__":
    main()
