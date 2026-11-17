"""One training record (912 screen elements, 19,535 state tokens; the next longest has 8,937) runs the 24 GB GPU out
of memory at any cap that leaves room for the GPU's other users. It stopped exec2 seed 0 at step 7,192 and seed 1 at
step 189. `--max-state-tokens` skips a sample whose state is longer than the limit; 0 (the default) skips nothing."""
from benchmarks.train import train_d1


def tok(text):
    return {"input_ids": text.split()}


def test_a_state_longer_than_the_limit_is_too_long():
    assert train_d1.too_long("a " * 11, tok, 10)
    assert not train_d1.too_long("a " * 10, tok, 10)


def test_no_limit_means_no_state_is_too_long():
    assert not train_d1.too_long("a " * 50000, tok, 0)
