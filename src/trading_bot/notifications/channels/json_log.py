"""JSONL append-only notification log channel.

Writes structured newline-delimited JSON events to a file for auditing,
forensic verification, and downstream analytics.
"""

import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Union
from trading_bot.notifications.channels.base import BaseNotificationChannel
from trading_bot.notifications.models import BaseNotificationEvent, DeliveryStatus

logger = logging.getLogger("JsonLogChannel")


class JsonLogChannel(BaseNotificationChannel):
    """Structured append-only JSONL event log channel."""

    def __init__(
        self,
        log_path: Union[str, Path] = "logs/notifications.jsonl",
        enabled: bool = True,
        auto_mkdir: bool = True,
    ):
        self.log_path = Path(log_path)
        self._enabled = enabled
        self._lock = threading.Lock()

        if self._enabled and auto_mkdir:
            try:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as ex:
                logger.warning(f"Could not create log directory {self.log_path.parent}: {ex}")

    @property
    def name(self) -> str:
        return "json_log"

    def is_enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        """Enable JSONL log channel."""
        self._enabled = True

    def disable(self) -> None:
        """Disable JSONL log channel."""
        self._enabled = False

    def dispatch(self, event: BaseNotificationEvent) -> DeliveryStatus:
        """Append event as a JSON line to the log file."""
        if not self._enabled:
            return DeliveryStatus(success=True, channel_name=self.name)

        try:
            line = event.model_dump_json() + "\n"
            with self._lock:
                # Ensure directory exists prior to open in case it was deleted
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()

            return DeliveryStatus(success=True, channel_name=self.name)
        except Exception as ex:
            logger.error(f"Failed to append event to {self.log_path}: {ex}", exc_info=True)
            return DeliveryStatus(
                success=False,
                channel_name=self.name,
                error_message=str(ex),
            )

    def read_events(self) -> List[Dict[str, Any]]:
        """Read and parse all events from the log file."""
        if not self.log_path.exists():
            return []
        events = []
        with self._lock:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        try:
                            events.append(json.loads(stripped))
                        except json.JSONDecodeError as ex:
                            logger.warning(f"Malformed line in {self.log_path}: {ex}")
        return events

    def clear_log(self) -> None:
        """Clear the contents of the log file."""
        with self._lock:
            if self.log_path.exists():
                with open(self.log_path, "w", encoding="utf-8") as f:
                    f.truncate(0)

    def close(self) -> None:
        """Flush and close resources."""
        pass
