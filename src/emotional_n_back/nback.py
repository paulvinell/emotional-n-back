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


class DualNBackTerminal:
    """
    Minimal terminal driver for dual n-back over two modalities (seq1, seq2).
    At each overall step, shows the two current items and asks whether
    seq1, seq2, or both are the SAME as n steps back.

    Controls:
      - Enter / "" / 0 / n / none  -> 'none'
      - 1 or s1                    -> seq1 only
      - 2 or s2                    -> seq2 only
      - 12 / 21 / b / both         -> both
      - q / quit / exit            -> stop early

    For debugging convenience, set show_truth=True to print ground truth each step.
    """

    def __init__(
        self,
        length: int,
        n: int,
        *,
        repeat_probability: float = 0.2,
        distinct_items: int = 4,
        seed: int | None = None,
        show_truth: bool = False,
    ):
        if seed is not None:
            random.seed(seed)

        self.length = length
        self.n = n
        self.show_truth = show_truth

        # Two independent modalities:
        self.seq1 = NBackSequence(
            length,
            n,
            repeat_probability=repeat_probability,
            distinct_items=distinct_items,
        )
        self.seq2 = NBackSequence(
            length,
            n,
            repeat_probability=repeat_probability,
            distinct_items=distinct_items,
        )

        # Stats
        self.history = []  # list of dicts (one per step)
        self.correct_1 = 0
        self.correct_2 = 0
        self.total_1 = 0
        self.total_2 = 0

    @staticmethod
    def _parse_answer(s: str):
        s = (s or "").strip().lower()
        if s in {"q", "quit", "exit"}:
            return "quit", False, False
        if s in {"", "0", "n", "none"}:
            return "ok", False, False
        if s in {"1", "s1"}:
            return "ok", True, False
        if s in {"2", "s2"}:
            return "ok", False, True
        if s in {"12", "21", "b", "both"}:
            return "ok", True, True
        # Unrecognized → prompt again
        return "retry", False, False

    def _feedback(self, want1: bool, truth1: bool, want2: bool, truth2: bool):
        icon = lambda good: "✓" if good else "✗"
        msg1 = f"S1 {icon(want1 == truth1)}"
        msg2 = f"S2 {icon(want2 == truth2)}"
        print(f"→ {msg1} | {msg2}")

    def _summary(self):
        acc1 = (self.correct_1 / self.total_1 * 100) if self.total_1 else 0.0
        acc2 = (self.correct_2 / self.total_2 * 100) if self.total_2 else 0.0
        print("\n=== SUMMARY ===")
        print(f"S1: {self.correct_1}/{self.total_1} correct ({acc1:.1f}%)")
        print(f"S2: {self.correct_2}/{self.total_2} correct ({acc2:.1f}%)")

    def play(self):
        print(f"Dual n-back (n={self.n}, steps={self.length})")
        print(
            "Answer after each step: [1] seq1, [2] seq2, [b] both, [Enter] none, [q] quit\n"
        )

        it1 = iter(self.seq1)
        it2 = iter(self.seq2)

        for t in range(self.length):
            v1, truth1 = next(it1)
            v2, truth2 = next(it2)

            # Display current step
            print(f"Step {t + 1}/{self.length}")
            print(f"  Seq 1: {v1}")
            print(f"  Seq 2: {v2}")
            if self.show_truth:
                print(
                    f"  [GT] S1={'same' if truth1 else 'diff'}  S2={'same' if truth2 else 'diff'}"
                )

            # Ask until valid (or quit)
            while True:
                ans = input("Same as n back? [1/2/b/Enter/q]: ")
                status, want1, want2 = self._parse_answer(ans)
                if status == "quit":
                    self._summary()
                    return
                if status == "ok":
                    break
                print("  Didn't catch that. Use 1, 2, b, Enter, or q.")

            # Update stats
            self.total_1 += 1
            self.total_2 += 1
            if want1 == truth1:
                self.correct_1 += 1
            if want2 == truth2:
                self.correct_2 += 1

            self.history.append(
                {
                    "t": t,
                    "v1": v1,
                    "truth1": truth1,
                    "guess1": want1,
                    "correct1": want1 == truth1,
                    "v2": v2,
                    "truth2": truth2,
                    "guess2": want2,
                    "correct2": want2 == truth2,
                }
            )

            # Immediate feedback
            self._feedback(want1, truth1, want2, truth2)
            print()  # spacer line

        self._summary()
