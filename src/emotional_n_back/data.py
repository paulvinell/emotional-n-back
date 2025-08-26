import random
import re

from emotional_n_back.constants import KDEF_DIR, MAV_DIR


class KDEFLoader:
    """
    Loader for KDEF images.
    """

    def __init__(self):
        self.emotions = self._load_emotions()
        self.enumerated_emotions: dict[int, str] = {
            i: e for i, e in enumerate(self.emotions)
        }

    def _load_emotions(self) -> list[str]:
        """
        Get the list of emotions from the KDEF directory
        by listing subdirectories.
        """
        return sorted([d.name for d in KDEF_DIR.iterdir() if d.is_dir()])

    def get_images_for_emotion(self, emotion: str):
        """
        Get the list of image paths for a given emotion.
        """
        emotion_dir = KDEF_DIR / emotion
        if not emotion_dir.exists() or not emotion_dir.is_dir():
            raise ValueError(
                f"Emotion directory {emotion_dir} does not exist or is not a directory."
            )
        return sorted([p for p in emotion_dir.iterdir() if p.is_file()])

    def get_random_image(self, emotion: str):
        """
        Get a random image path for a given emotion.
        """

        images = self.get_images_for_emotion(emotion)
        if not images:
            raise ValueError(f"No images found for emotion {emotion}.")
        return random.choice(images)

    def get_emotion(self, index: int):
        """
        Get the emotion corresponding to the given index.
        """
        if index not in self.enumerated_emotions:
            raise ValueError(f"Index {index} is out of range for emotions.")
        return self.enumerated_emotions[index]


class MAVLoader:
    def __init__(self):
        self.emotions = self._load_emotions()
        self.enumerated_emotions: dict[int, str] = {
            i: e for i, e in enumerate(self.emotions)
        }

    def _load_emotions(self) -> list[str]:
        """
        Get the list of emotions from the MAV directory
        by extracting unique emotion codes from filenames.
        """
        emotion_pattern = re.compile(r"^\d+_(\w+)\.wav$")
        emotions = set()
        for file in MAV_DIR.iterdir():
            if file.is_file():
                match = emotion_pattern.match(file.name)
                if match:
                    emotions.add(match.group(1))
        return sorted(emotions)

    def get_audio_for_emotion(self, emotion: str):
        """
        Get the list of audio file paths for a given emotion.
        """
        emotion_pattern = re.compile(rf"^\d+_{re.escape(emotion)}\.wav$")
        audio_files = [
            p
            for p in MAV_DIR.iterdir()
            if p.is_file() and emotion_pattern.match(p.name)
        ]
        return sorted(audio_files)

    def get_random_audio(self, emotion: str):
        """
        Get a random audio file path for a given emotion.
        """
        audio_files = self.get_audio_for_emotion(emotion)
        if not audio_files:
            raise ValueError(f"No audio files found for emotion {emotion}.")
        return random.choice(audio_files)

    def get_emotion(self, index: int):
        """
        Get the emotion corresponding to the given index.
        """
        if index not in self.enumerated_emotions:
            raise ValueError(f"Index {index} is out of range for emotions.")
        return self.enumerated_emotions[index]
