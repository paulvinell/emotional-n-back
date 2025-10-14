import queue
import threading
from dataclasses import dataclass
from typing import Dict, Optional

from emotional_n_back.utils.eeg.erp import OscErpServer


@dataclass
class ErpUpdate:
    code: str
    clean: bool | str
    amp: Optional[float]
    lat: Optional[float]
    raw: Dict  # full original payload for debugging


class ErpAdapter:
    def __init__(self, erp_component: str, fs_target: float = 256.0, **kwargs):
        self.erp_component = erp_component
        self._updates = queue.Queue()
        self.eeg_started = threading.Event()

        self._server = OscErpServer(
            fs_target=fs_target,
            on_update=self._handle_erp_update,
            components_to_calculate=[erp_component],
            eeg_started=self.eeg_started,
            **kwargs,
        )

    @property
    def effective_fs(self) -> float:
        return self._server.resampler.fs_obs

    @property
    def stream_health(self) -> dict:
        return {
            "fs_target": self._server.resampler.fs_target,
            "fs_obs": self._server.resampler.fs_obs,
            "drift_ratio": self._server.resampler.drift_ratio,
        }

    @property
    def continuous_fs_est(self) -> float:
        return self._server.resampler.fs_obs

    def _handle_erp_update(self, update: dict):
        component_data = update.get("component", {}).get(self.erp_component, {})
        erp_update = ErpUpdate(
            code=update.get("code"),
            clean=update.get("clean"),
            amp=component_data.get("amp"),
            lat=component_data.get("lat"),
            raw=update,
        )
        self._updates.put(erp_update)

    def start(self) -> None:
        self._server.start()

    def shutdown(self) -> None:
        self._server.shutdown()

    def ingest_event(self, code: str) -> None:
        self._server.ingest_event(code)

    def poll_update(
        self, code: Optional[str] = None, timeout: float = 0.0
    ) -> Optional[ErpUpdate]:
        try:
            update = self._updates.get(block=timeout > 0, timeout=timeout)
            if code is None or update.code == code:
                return update
            else:
                # This is not ideal, but it's the best we can do with a single queue
                # without losing updates. A better implementation would use a dictionary
                # of queues, one for each event code.
                self._updates.put(update)
                return None
        except queue.Empty:
            return None
