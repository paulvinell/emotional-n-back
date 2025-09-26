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


class KDEFSentimentLoader:
    """
    Loader for KDEF images, but instead of by emotion,
    it grabs by sentiment: positive, negative, neutral.

    Surprise is excluded because it can be ambiguous.
    """

    KDEF_SENTIMENT_MAP = {
        "positive": ["happy"],
        "neutral": ["neutral"],
        "negative": ["angry", "sad", "fear", "disgust"],
    }

    def __init__(self, binary: bool = False):
        self.sentiment_map = self.KDEF_SENTIMENT_MAP.copy()
        if binary:
            del self.sentiment_map["neutral"]
        self.sentiments = list(self.sentiment_map.keys())
        self.enumerated_sentiments: dict[int, str] = {
            i: s for i, s in enumerate(self.sentiments)
        }
        self._kdef_loader = KDEFLoader()

    def get_images_for_sentiment(self, sentiment: str):
        """
        Get the list of image paths for a given sentiment.
        """
        if sentiment not in self.sentiment_map:
            raise ValueError(f"Sentiment {sentiment} is not recognized.")

        emotions = self.sentiment_map[sentiment]
        images = []
        for emotion in emotions:
            images.extend(self._kdef_loader.get_images_for_emotion(emotion))

        return sorted(images)

    def get_random_image(self, sentiment: str):
        """
        Get a random image path for a given sentiment.
        """
        images = self.get_images_for_sentiment(sentiment)
        if not images:
            raise ValueError(f"No images found for sentiment {sentiment}.")
        return random.choice(images)

    def get_sentiment(self, index: int):
        """
        Get the sentiment corresponding to the given index.
        """
        if index not in self.enumerated_sentiments:
            raise ValueError(f"Index {index} is out of range for sentiments.")
        return self.enumerated_sentiments[index]


class MAVSentimentLoader:
    """
    Loader for MAV audio files, but instead of by emotion,
    it grabs by sentiment: positive, negative, neutral.

    Surprise is excluded because it can be ambiguous.
    """

    MAV_SENTIMENT_MAP = {
        "positive": ["pleasure"],
        "neutral": ["neutral"],
        "negative": ["anger", "disgust", "fear", "pain"],
    }

    def __init__(self, binary: bool = False):
        self.sentiment_map = self.MAV_SENTIMENT_MAP.copy()
        if binary:
            del self.sentiment_map["neutral"]
        self.sentiments = list(self.sentiment_map.keys())
        self.enumerated_sentiments: dict[int, str] = {
            i: s for i, s in enumerate(self.sentiments)
        }
        self._mav_loader = MAVLoader()

    def get_audio_for_sentiment(self, sentiment: str):
        """
        Get the list of audio file paths for a given sentiment.
        """
        if sentiment not in self.sentiment_map:
            raise ValueError(f"Sentiment {sentiment} is not recognized.")

        emotions = self.sentiment_map[sentiment]
        audio_files = []
        for emotion in emotions:
            audio_files.extend(self._mav_loader.get_audio_for_emotion(emotion))

        return sorted(audio_files)

    def get_random_audio(self, sentiment: str):
        """
        Get a random audio file path for a given sentiment.
        """
        audio_files = self.get_audio_for_sentiment(sentiment)
        if not audio_files:
            raise ValueError(f"No audio files found for sentiment {sentiment}.")
        return random.choice(audio_files)

    def get_sentiment(self, index: int):
        """
        Get the sentiment corresponding to the given index.
        """
        if index not in self.enumerated_sentiments:
            raise ValueError(f"Index {index} is out of range for sentiments.")
        return self.enumerated_sentiments[index]


class TESSLoader:
    """
    Loader for TESS audio files.
    """

    def __init__(self):
        self.emotions = self._load_emotions()
        self.emotion_dirs = self._load_emotion_dirs()
        self.enumerated_emotions: dict[int, str] = {
            i: e for i, e in enumerate(self.emotions)
        }

    def _load_emotions(self) -> list[str]:
        """
        Get the list of emotions from the TESS directory
        by listing subdirectories.
        """
        dirs = sorted(
            [d.name for d in (KDEF_DIR.parent / "tess").iterdir() if d.is_dir()]
        )
        emotions = [dir_name.split("_")[1] for dir_name in dirs]
        return sorted(set(emotions))

    def _load_emotion_dirs(self) -> dict[str, list[str]]:
        """
        Map emotions to their corresponding directories.
        """
        dirs = sorted([d for d in (KDEF_DIR.parent / "tess").iterdir() if d.is_dir()])
        emotion_dirs = {}
        for dir in dirs:
            emotion = dir.name.split("_")[1]
            if emotion not in emotion_dirs:
                emotion_dirs[emotion] = []
            emotion_dirs[emotion].append(dir)
        return emotion_dirs

    def get_audio_for_emotion(self, emotion: str):
        """
        Get the list of audio file paths for a given emotion.
        """
        if emotion not in self.emotion_dirs:
            raise ValueError(f"Emotion {emotion} is not recognized.")

        audio_files = []
        for dir in self.emotion_dirs[emotion]:
            audio_files.extend([p for p in dir.iterdir() if p.is_file()])

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


class TESSSentimentLoader:
    """
    Loader for TESS audio files, but instead of by emotion,
    it grabs by sentiment: positive, negative, neutral.

    Surprise is excluded because it can be ambiguous.
    """

    TESS_SENTIMENT_MAP = {
        "positive": ["happy", "pleasant_surprise"],
        "neutral": ["neutral"],
        "negative": ["sad", "anger", "disgust", "fear"],
    }

    def __init__(self, binary: bool = False):
        self.sentiment_map = self.TESS_SENTIMENT_MAP.copy()
        if binary:
            del self.sentiment_map["neutral"]
        self.sentiments = list(self.sentiment_map.keys())
        self.enumerated_sentiments: dict[int, str] = {
            i: s for i, s in enumerate(self.sentiments)
        }
        self._tess_loader = TESSLoader()

    def get_audio_for_sentiment(self, sentiment: str):
        """
        Get the list of audio file paths for a given sentiment.
        """
        if sentiment not in self.sentiment_map:
            raise ValueError(f"Sentiment {sentiment} is not recognized.")

        emotions = self.sentiment_map[sentiment]
        audio_files = []
        for emotion in emotions:
            audio_files.extend(self._tess_loader.get_audio_for_emotion(emotion))

        return sorted(audio_files)

    def get_random_audio(self, sentiment: str):
        """
        Get a random audio file path for a given sentiment.
        """
        audio_files = self.get_audio_for_sentiment(sentiment)
        if not audio_files:
            raise ValueError(f"No audio files found for sentiment {sentiment}.")
        return random.choice(audio_files)

    def get_sentiment(self, index: int):
        """
        Get the sentiment corresponding to the given index.
        """
        if index not in self.enumerated_sentiments:
            raise ValueError(f"Index {index} is out of range for sentiments.")
        return self.enumerated_sentiments[index]
