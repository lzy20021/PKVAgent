from PySide6.QtCore import QThread, Signal


class AsyncExecutor(QThread):
    """Run agent request without blocking UI thread."""

    progress = Signal(str)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, agent, user_input: str):
        super().__init__()
        self.agent = agent
        self.user_input = user_input

    def run(self):
        try:
            def on_event(event_type, data):
                if event_type == "thinking":
                    self.progress.emit(str(data))
                elif event_type == "tool_start":
                    name = data.get("name") if isinstance(data, dict) else "tool"
                    self.progress.emit(f"Executing: {name}")

            self.agent.set_event_callback(on_event)
            reply = self.agent.chat(self.user_input)
            self.finished.emit(reply)
        except Exception as exc:
            self.error.emit(str(exc))
