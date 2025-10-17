"""WebSocket server for broadcasting breathing metrics to connected clients."""

import asyncio
import json
import logging
from typing import Set, Dict, Any
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class BreathingMetricsServer:
    """WebSocket server for broadcasting breathing metrics."""

    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize the WebSocket server.

        Args:
            host: Host address to bind the server
            port: Port number for the WebSocket server
        """
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.running = False
        self.server = None

    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        # Send initial connection confirmation
        await self.send_to_client(
            websocket,
            {
                "type": "connection",
                "status": "connected",
                "message": "Connected to breathing metrics server"
            }
        )

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")

    async def send_to_client(self, websocket: WebSocketServerProtocol, data: Dict[str, Any]):
        """Send data to a specific client."""
        try:
            message = json.dumps(data)
            await websocket.send(message)
        except Exception as e:
            logger.error(f"Error sending to client: {e}")

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """
        Broadcast breathing metrics to all connected clients.

        Args:
            metrics: Dictionary containing breathing metrics
                Expected keys:
                - bpm: Breaths per minute (float)
                - apnea_detected: Boolean indicating apnea detection
                - shallow_breathing: Boolean indicating shallow breathing
                - timestamp: ISO format timestamp (optional)
        """
        if not self.clients:
            return

        # Format the message
        message = {
            "type": "metrics",
            "data": {
                "bpm": metrics.get("bpm", 0.0),
                "apnea_detected": metrics.get("apnea_detected", False),
                "shallow_breathing": metrics.get("shallow_breathing", False),
                "timestamp": metrics.get("timestamp", "")
            }
        }

        # Send to all clients
        disconnected_clients = set()
        for websocket in self.clients:
            try:
                await self.send_to_client(websocket, message)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(websocket)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected_clients.add(websocket)

        # Clean up disconnected clients
        for websocket in disconnected_clients:
            await self.unregister_client(websocket)

    async def handle_client(self, websocket: WebSocketServerProtocol):
        """Handle individual client connections."""
        await self.register_client(websocket)
        try:
            # Keep connection alive and handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    # Handle client messages (ping, subscription requests, etc.)
                    if data.get("type") == "ping":
                        await self.send_to_client(websocket, {"type": "pong"})
                    elif data.get("type") == "subscribe":
                        await self.send_to_client(
                            websocket,
                            {"type": "subscribed", "status": "success"}
                        )
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received: {message}")
                except Exception as e:
                    logger.error(f"Error handling client message: {e}")
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        finally:
            await self.unregister_client(websocket)

    async def start(self):
        """Start the WebSocket server."""
        self.running = True
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")

    async def stop(self):
        """Stop the WebSocket server."""
        self.running = False
        if self.server:
            self.server.close()
            await self.server.wait_closed()
        # Close all client connections
        for websocket in self.clients.copy():
            await websocket.close()
        self.clients.clear()
        logger.info("WebSocket server stopped")


class BreathingMetricsPublisher:
    """Publishes breathing metrics at periodic intervals."""

    def __init__(self, server: BreathingMetricsServer, update_interval: float = 1.0):
        """
        Initialize the metrics publisher.

        Args:
            server: BreathingMetricsServer instance
            update_interval: Time between metric updates in seconds
        """
        self.server = server
        self.update_interval = update_interval
        self.running = False
        self._task = None

    async def publish_periodic_metrics(self, metrics_callback):
        """
        Periodically publish metrics from a callback function.

        Args:
            metrics_callback: Callable that returns current metrics dict
        """
        self.running = True
        while self.running:
            try:
                metrics = metrics_callback()
                if metrics:
                    await self.server.broadcast_metrics(metrics)
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error publishing metrics: {e}")
                await asyncio.sleep(self.update_interval)

    def start(self, metrics_callback):
        """Start publishing metrics."""
        self._task = asyncio.create_task(self.publish_periodic_metrics(metrics_callback))
        return self._task

    async def stop(self):
        """Stop publishing metrics."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def run_server(host: str = "localhost", port: int = 8765):
    """
    Run the WebSocket server.

    Example usage:
        asyncio.run(run_server())
    """
    server = BreathingMetricsServer(host, port)
    await server.start()
    try:
        # Keep server running indefinitely
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
    finally:
        await server.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the server
    asyncio.run(run_server())
