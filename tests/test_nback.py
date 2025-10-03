from unittest.mock import MagicMock, patch

import pytest

from emotional_n_back.nback import NBackSequence, OneBackSequence


class TestOneBackSequence:
    def test_init(self):
        """
        Test initialization of OneBackSequence.
        """
        seq = OneBackSequence(length=10, repeat_probability=0.5, distinct_items=4)
        assert seq.length == 10
        assert seq.repeat_probability == 0.5
        assert seq.distinct_items == 4

    def test_init_invalid_length(self):
        """
        Test that OneBackSequence raises an error for invalid length.
        """
        with pytest.raises(AssertionError):
            OneBackSequence(length=0)

    def test_init_invalid_repeat_probability(self):
        """
        Test that OneBackSequence raises an error for invalid repeat_probability.
        """
        with pytest.raises(AssertionError):
            OneBackSequence(length=10, repeat_probability=1.1)
        with pytest.raises(AssertionError):
            OneBackSequence(length=10, repeat_probability=-0.1)

    def test_init_invalid_distinct_items(self):
        """
        Test that OneBackSequence raises an error for invalid distinct_items.
        """
        with pytest.raises(AssertionError):
            OneBackSequence(length=10, distinct_items=0)

    @patch("random.randint", return_value=1)
    @patch("random.random")
    @patch("random.choice")
    def test_generate(self, mock_choice, mock_random, mock_randint):
        """
        Test the generate method with mocked randomness.

        randint affects the first item. Random choice is
        used in subsequent items to avoid unintended repeats.
        """
        # Control the randomness
        mock_random.side_effect = [
            0.2,
            0.8,
            0.2,
            0.8,
            0.2,
        ]  # repeat, no repeat, repeat, ...
        mock_choice.side_effect = [2, 3, 4, 1]

        seq = OneBackSequence(length=5, repeat_probability=0.3, distinct_items=4)
        sequence, truth = seq.sequence, seq.truth

        assert sequence == [1, 1, 2, 2, 3]
        assert truth == [False, True, False, True, False]

    def test_len(self):
        """
        Test the __len__ method.
        """
        seq = OneBackSequence(length=15)
        assert len(seq) == 15

    def test_iter(self):
        """
        Test the __iter__ method.
        """
        seq = OneBackSequence(length=5)
        items = list(seq)
        assert len(items) == 5
        for item, truth in items:
            assert isinstance(item, int)
            assert isinstance(truth, bool)

    def test_next(self):
        """
        Test the __next__ method.
        """
        seq = OneBackSequence(length=3)
        _ = next(seq)
        _ = next(seq)
        _ = next(seq)
        with pytest.raises(StopIteration):
            next(seq)


class TestNBackSequence:
    def test_init(self):
        """
        Test initialization of NBackSequence.
        """
        seq = NBackSequence(length=20, n=2, distinct_items=4)
        assert seq.length == 20
        assert seq.n == 2
        assert seq.partial_sequence_length == 10
        assert len(seq.partial_sequences) == 2

    def test_len(self):
        """
        Test the __len__ method of NBackSequence.
        """
        seq = NBackSequence(length=25, n=3)
        assert len(seq) == 25

    @patch("emotional_n_back.nback.OneBackSequence")
    def test_iter(self, mock_one_back_sequence):
        """
        Test the __iter__ method with mocked partial sequences.
        """
        # Create mock partial sequences
        mock_seq1 = MagicMock()
        mock_seq1.sequence = [1, 2, 3]
        mock_seq1.truth = [False, True, False]

        mock_seq2 = MagicMock()
        mock_seq2.sequence = [4, 5, 6]
        mock_seq2.truth = [True, False, True]

        # Make the mock return different sequences for each call
        mock_one_back_sequence.side_effect = [mock_seq1, mock_seq2]

        seq = NBackSequence(length=6, n=2, repeat_probability=0.5, distinct_items=2)

        expected_sequence = [
            (1, False),
            (4, True),
            (2, True),
            (5, False),
            (3, False),
            (6, True),
        ]

        generated_sequence = list(seq)

        assert generated_sequence == expected_sequence

    def test_next(self):
        """
        Test the __next__ method of NBackSequence,
        ensuring StopIteration is raised correctly.
        """
        seq = NBackSequence(length=4, n=2)
        _ = next(seq)
        _ = next(seq)
        _ = next(seq)
        _ = next(seq)
        with pytest.raises(StopIteration):
            next(seq)

    def test_iter_uneven_length(self):
        """
        Test that iteration works correctly for uneven lengths.
        """
        seq = NBackSequence(length=7, n=3)
        assert len(list(seq)) == 7
