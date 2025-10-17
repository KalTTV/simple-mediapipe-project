"""
WebSocket server for broadcasting breathing state.
"""

import asyncio
import json
import websockets
from typing import Optional, Set
import threading


class WebSocketBroadcaster:
    """WebSocket server for broadcasting breathing monitor state."""
    
    def __init__(self, host: str = "localhost", port: int = 8765):
        """
        Initialize WebSocket broadcaster.
        
        Args:
            host: Server host
            port: Server port
        """
        self.host = host
        self.port = port
        self.clients: Set = set()
        self.server = None
        self.loop = None
        self.thread = None
        self.running = False
        self.latest_state = {}
        
    async def _register(self, websocket):
        """Register a new client."""
        self.clients.add(websocket)
        print(f"[WS] Client connected. Total clients: {len(self.clients)}")
        
    async def _unregister(self, websocket):
        """Unregister a client."""
        self.clients.discard(websocket)
        print(f"[WS] Client disconnected. Total clients: {len(self.clients)}")
        
    async def _handler(self, websocket, path):
        """Handle WebSocket connections."""
        await self._register(websocket)
        try:
            # Send latest state on connection
            if self.latest_state:
                await websocket.send(json.dumps(self.latest_state))
            
            # Keep connection alive
            async for message in websocket:
                pass  # We don't expect messages from clients
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self._unregister(websocket)
    
    async def _broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
    
    def _run_server(self):
        """Run WebSocket server in event loop."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        start_server = websockets.serve(self._handler, self.host, self.port)
        self.server = self.loop.run_until_complete(start_server)
        
        print(f"[WS] Server started on ws://{self.host}:{self.port}")
        self.running = True
        
        try:
            self.loop.run_forever()
        finally:
            self.running = False
    
    def start(self):
        """Start WebSocket server in background thread."""
        if self.thread and self.thread.is_alive():
            print("[WS] Server already running")
            return
        
        self.thread = threading.Thread(target=self._run_server, daemon=True)
        self.thread.start()
        
        # Wait for server to start
        import time
        for _ in range(50):  # 5 seconds max
            if self.running:
                break
            time.sleep(0.1)
    
    def stop(self):
        """Stop WebSocket server."""
        if not self.running:
            return
        
        print("[WS] Stopping server...")
        
        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        self.running = False
        print("[WS] Server stopped")
    
    def broadcast_state(self, bpm: Optional[float], apnea: bool, shallow: bool, 
                       confidence: float, timestamp: float):
        """
        Broadcast current breathing state.
        
        Args:
            bpm: Current BPM (or None)
            apnea: Apnea detected
            shallow: Shallow breathing detected
            confidence: Confidence level
            timestamp: Timestamp
        """
        state = {
            "bpm": bpm,
            "apnea": apnea,
            "shallow": shallow,
            "confidence": confidence,
            "timestamp": timestamp
        }
        
        self.latest_state = state
        
        if self.running and self.loop and self.clients:
            message = json.dumps(state)
            asyncio.run_coroutine_threadsafe(
                self._broadcast(message), 
                self.loop
            )


def main():
    """Test WebSocket server."""
    import time
    
    broadcaster = WebSocketBroadcaster()
    broadcaster.start()
    
    print("Broadcasting test data... Press Ctrl+C to stop")
    
    try:
        for i in range(100):
            bpm = 40 + 10 * (i % 3)
            broadcaster.broadcast_state(
                bpm=bpm,
                apnea=i % 20 == 0,
                shallow=i % 15 == 0,
                confidence=0.8,
                timestamp=time.time()
            )
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        broadcaster.stop()


if __name__ == "__main__":
    main()

