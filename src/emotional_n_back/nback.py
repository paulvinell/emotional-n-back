import random


class NBackPartialSequence:
    """
    Dual N-Back could be considered as 2 separate N-Back tasks,
    one for each modality (visual and auditory). In turn,
    each N-Back task can be thought of as N partial sequences,
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

        assert length > 0, "length must be positive"
        assert 0 <= repeat_probability <= 1, (
            "repeat_probability must be between 0 and 1"
        )
        assert distinct_items > 0, "distinct_items must be positive"

    def generate(self):
        """
        To generate a sequence, we iterate in pairs over the
        length of the sequence, and with some probability we
        repeat the previous item.
        """
        sequence = [0] * self.length
        for i in range(1, self.length):
            if random.random() < self.repeat_probability:
                sequence[i] = sequence[i - 1]
            else:
                sequence[i] = random.randint(1, self.distinct_items)
        return sequence
