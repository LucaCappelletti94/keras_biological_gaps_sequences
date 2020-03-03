from silence_tensorflow import silence_tensorflow
silence_tensorflow()
from keras_biological_gaps_sequences import BiologicalGapsSequence
import pytest

def test_biological_gaps_sequence_wrong_input():
    with pytest.raises(ValueError):
        BiologicalGapsSequence("hg38", "hg19", 1000, 1, 32)

def test_biological_gaps_sequence():
    sequence = BiologicalGapsSequence("hg19", "hg38", 1000, 1, 32)
    x, y = sequence[0]
    assert sequence.samples_number == 19
    assert len(sequence) == 1
    assert x.shape == (sequence.samples_number, 1000, 4)
    assert y.shape == (sequence.samples_number, 1, 4)
    x1, y1 = sequence[0]
    assert (x==x1).all()
    assert (y==y1).all()
    assert sequence.batch_size == 32
    assert sequence.steps_per_epoch == 1
    sequence.batch_size = 10
    sequence.on_epoch_end()
    x, y = sequence[0]
    assert len(sequence) == 2
    assert x.shape == (10, 1000, 4)
    assert y.shape == (10, 1, 4)