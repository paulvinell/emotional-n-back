import math
import random


class OneBackSequence:
    """
    Dual N-Back could be considered as 2 separate N-Back tasks,
    one for each modality (visual and auditory). In turn,
    each N-Back task can be thought of as N 1-back sequences,
    shown in a "round-robin" fashion. E.g:

    1. Seq_1[0], Seq_2[0], Seq_3[0]
    2. Seq_1[1], Seq_2[1], Seq_3[1]
    3. ...
    4. Seq_1[end], Seq_2[end], Seq_3[end]
    """

    def __init__(
        self,
        length: int,
        repeat_probability: float = 0.2,
        distinct_items: int = 4,
    ):
        self.length = length
        self.repeat_probability = repeat_probability
        self.distinct_items = distinct_items

        assert length > 0, "length must be positive"
        assert 0 <= repeat_probability <= 1, (
            "repeat_probability must be between 0 and 1"
        )
        assert distinct_items > 0, "distinct_items must be positive"

        self.sequence, self.truth = self.generate()

    def generate(self) -> tuple[list[int], list[bool]]:
        """
        To generate a sequence, we iterate in pairs over the
        length of the sequence, and with some probability we
        repeat the previous item.
        """
        sequence = [0] * self.length
        sequence[0] = random.randint(1, self.distinct_items)
        truth = [False] * self.length
        for i in range(1, self.length):
            if random.random() < self.repeat_probability:
                sequence[i] = sequence[i - 1]
                truth[i] = True
            else:
                # Generate a random int that is not sequence[i-1],
                # otherwise we get unintended repeats.
                sequence[i] = random.choice(
                    [
                        x
                        for x in range(1, self.distinct_items + 1)
                        if x != sequence[i - 1]
                    ]
                )
        return sequence, truth

    def __len__(self) -> int:
        """
        Get the length of the sequence.
        """
        return self.length

    def __iter__(self):
        """
        Iterate over the sequence.
        """
        for i in range(self.length):
            yield self.sequence[i], self.truth[i]

    def __next__(self):
        """
        Get the next item in the sequence.
        """
        if not self.sequence:
            raise StopIteration
        return self.sequence.pop(0), self.truth.pop(0)


class NBackSequence:
    """
    NBackSequence is a wrapper around NBackPartialSequence
    that generates a sequence of partial sequences.
    """

    def __init__(self, length: int, n: int, **kwargs):
        self.length = length
        self.n = n
        self.partial_sequence_length = math.ceil(length / n)
        self.partial_sequences = [
            OneBackSequence(self.partial_sequence_length, **kwargs) for _ in range(n)
        ]

        # for __next__
        self.current_i = 0
        self.current_j = 0
        self.current_total = 0

    def __len__(self) -> int:
        """
        Get the length of the sequence.
        """
        return self.length

    def __iter__(self):
        """
        Iterate over the sequence.
        """
        total = 0
        for i in range(self.partial_sequence_length):
            for j in range(self.n):
                if total >= self.length:
                    return

                yield (
                    self.partial_sequences[j].sequence[i],
                    self.partial_sequences[j].truth[i],
                )
                total += 1

    def __next__(self):
        """
        Get the next item in the sequence.
        """
        if self.current_total >= self.length:
            raise StopIteration

        item = self.partial_sequences[self.current_j].sequence[self.current_i]
        truth = self.partial_sequences[self.current_j].truth[self.current_i]

        self.current_total += 1
        self.current_i += 1

        if self.current_i >= self.partial_sequence_length:
            self.current_i = 0
            self.current_j += 1

        return item, truth
