"""Entrypoint for the Reachy Mini conversation app."""

import os
import sys
import time
import asyncio
import argparse
import threading
from typing import Any, Dict, List, Optional

import gradio as gr
from fastapi import FastAPI
from fastrtc import Stream

from reachy_mini import ReachyMini, ReachyMiniApp
from reachy_mini_conversation_app.utils import parse_args, setup_logger, handle_vision_stuff


def update_chatbot(
    chatbot: List[Dict[str, Any]], response: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Append an AdditionalOutputs message to the Gradio chatbot."""
    chatbot.append(response)
    return chatbot


def main() -> None:
    args, _ = parse_args()
    run(args)


def run(
    args: argparse.Namespace,
    robot: ReachyMini = None,
    app_stop_event: Optional[threading.Event] = None,
    settings_app: Optional[FastAPI] = None,
    instance_path: Optional[str] = None,
) -> None:
    """Run the Reachy Mini conversation app."""
    from reachy_mini_conversation_app.moves import MovementManager
    from reachy_mini_conversation_app.console import LocalStream
    from reachy_mini_conversation_app.stuart_realtime import StuartRealtimeHandler
    from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
    from reachy_mini_conversation_app.audio.head_wobbler import HeadWobbler

    logger = setup_logger(args.debug)
    logger.info("Starting Reachy Mini Conversation App")

    # ------------------------------------------------------------------
    # Robot init
    # ------------------------------------------------------------------
    if robot is None:
        logger.info("Using default backend")
        robot = ReachyMini(media_backend="default")

    camera_worker, _, vision_manager = handle_vision_stuff(args, robot)

    # ------------------------------------------------------------------
    # Managers
    # ------------------------------------------------------------------
    movement_manager = MovementManager(
        current_robot=robot,
        camera_worker=camera_worker,
    )
    head_wobbler = HeadWobbler(
        set_speech_offsets=movement_manager.set_speech_offsets
    )
    deps = ToolDependencies(
        reachy_mini=robot,
        movement_manager=movement_manager,
        camera_worker=camera_worker,
        vision_manager=vision_manager,
        head_wobbler=head_wobbler,
    )

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------
    handler = StuartRealtimeHandler(
        deps, gradio_mode=args.gradio, instance_path=instance_path
    )

    # ------------------------------------------------------------------
    # UI / stream
    # ------------------------------------------------------------------
    current_file_path = os.path.dirname(os.path.abspath(__file__))

    if args.gradio:
        chatbot = gr.Chatbot(
            type="messages",
            resizable=True,
            avatar_images=(
                os.path.join(current_file_path, "images", "user_avatar.png"),
                os.path.join(current_file_path, "images", "reachymini_avatar.png"),
            ),
        )

        stream = Stream(
            handler=handler,
            mode="send-receive",
            modality="audio",
            additional_inputs=[chatbot],
            additional_outputs=[chatbot],
            additional_outputs_handler=update_chatbot,
            ui_args={"title": "Talk with Reachy Mini — Stuart AI RAG"},
        )
        stream_manager = stream.ui

        app = settings_app or FastAPI()
        app = gr.mount_gradio_app(app, stream.ui, path="/")
    else:
        stream_manager = LocalStream(
            handler,
            robot,
            settings_app=settings_app,
            instance_path=instance_path,
        )

    # ------------------------------------------------------------------
    # Start background workers
    # ------------------------------------------------------------------
    movement_manager.start()
    head_wobbler.start()
    if camera_worker:
        camera_worker.start()
    if vision_manager:
        vision_manager.start()

    # Graceful stop on external event
    def _poll_stop() -> None:
        if app_stop_event is not None:
            app_stop_event.wait()
        logger.info("Stop event received — shutting down")
        try:
            stream_manager.close()
        except Exception as exc:
            logger.error("Error closing stream manager: %s", exc)

    if app_stop_event:
        threading.Thread(target=_poll_stop, daemon=True).start()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    try:
        stream_manager.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt — shutting down")
    finally:
        movement_manager.stop()
        head_wobbler.stop()
        if camera_worker:
            camera_worker.stop()
        if vision_manager:
            vision_manager.stop()
        try:
            robot.media.close()
        except Exception as exc:
            logger.debug("Error closing media: %s", exc)
        robot.client.disconnect()
        time.sleep(1)
        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# ReachyMiniApp entry point
# ---------------------------------------------------------------------------

class ReachyMiniConversationApp(ReachyMiniApp):  # type: ignore[misc]
    """Reachy Mini Apps entry point for the conversation app."""

    custom_app_url = "http://0.0.0.0:7860/"
    dont_start_webserver = False

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        args, _ = parse_args()
        instance_path = self._get_instance_path().parent
        run(
            args,
            robot=reachy_mini,
            app_stop_event=stop_event,
            settings_app=self.settings_app,
            instance_path=instance_path,
        )


if __name__ == "__main__":
    app = ReachyMiniConversationApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
