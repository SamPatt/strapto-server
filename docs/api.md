# StrapTo Local Server API Documentation

## Overview

The StrapTo Local Server exposes several HTTP endpoints for server control, WebRTC signaling, and model interaction. All endpoints accept and return JSON data unless otherwise specified.

## Base URL

By default, the server runs on `http://localhost:8080`. All endpoints are relative to this base URL.

## Endpoints

### Health Check

```http
GET /health
```

Simple endpoint to verify server is running.

#### Response

```json
{
    "status": "healthy"
}
```

### Server Status

```http
GET /status
```

Get detailed information about server state, including WebRTC connections and model status.

#### Response

```json
{
    "server": {
        "status": "running",
        "uptime": 3600.5  // seconds
    },
    "webrtc": {
        "active_connections": 2,
        "data_channels": 1
    },
    "model": {
        "status": "connected",
        "last_activity": "2024-02-09T15:30:45.123Z"
    }
}
```

### WebRTC Signaling

#### Submit Offer

```http
POST /webrtc/offer
```

Submit a WebRTC offer to initiate a connection.

##### Request Body

```json
{
    "sdp": "v=0\no=- 4611731400...",  // SDP string
    "type": "offer",
    "client_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

##### Response

```json
{
    "sdp": "v=0\no=- 4611731400...",  // Answer SDP
    "type": "answer"
}
```

##### Error Response

```json
{
    "error": "Missing required fields: sdp, type",
    "type": "ValidationError",
    "timestamp": "2024-02-09T15:30:45.123Z"
}
```

#### Submit ICE Candidate

```http
POST /webrtc/ice-candidate
```

Submit an ICE candidate for connection negotiation.

##### Request Body

```json
{
    "candidate": "candidate:1 1 UDP 2013266431...",
    "sdpMLineIndex": 0,
    "sdpMid": "0",
    "client_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

##### Response

```json
{
    "status": "accepted"
}
```

##### Error Response

```json
{
    "error": "Missing required fields: candidate",
    "type": "ValidationError",
    "timestamp": "2024-02-09T15:30:45.123Z"
}
```

### Model Control

#### Reset Model Connection

```http
POST /model/reset
```

Reset the model interface connection. Useful when the model becomes unresponsive.

##### Response

```json
{
    "status": "reset_complete",
    "connection_status": "connected"
}
```

##### Error Response

```json
{
    "error": "Model interface is not connected or is already resetting",
    "type": "StateError",
    "timestamp": "2024-02-09T15:30:45.123Z"
}
```

## Error Handling

All endpoints follow a consistent error response format:

```json
{
    "error": "Description of what went wrong",
    "type": "ErrorType",
    "timestamp": "2024-02-09T15:30:45.123Z"
}
```

Common error types include:
- `ValidationError`: Invalid or missing request parameters
- `StateError`: Invalid server or component state
- `ConnectionError`: Network or connection issues
- `InternalError`: Unexpected server errors

## CORS Support

When running in debug mode, the server enables CORS support with the following configuration:
- All origins allowed (`*`)
- Credentials allowed
- All headers exposed and allowed

## Rate Limiting

The server implements rate limiting using a token bucket algorithm. Limits are applied per endpoint:
- WebRTC endpoints: 10 requests per second
- Model control endpoints: 2 requests per second
- Status endpoints: 5 requests per second

When rate limit is exceeded, the server returns:
```json
{
    "error": "Rate limit exceeded. Please try again later.",
    "type": "RateLimitError",
    "timestamp": "2024-02-09T15:30:45.123Z"
}
```

## WebSocket Events

In addition to HTTP endpoints, the server emits several WebSocket events through established WebRTC data channels:

### Server → Client Events

- `model_output`: New output from the model
  ```json
  {
      "type": "text|json|image",
      "content": "Model generated content",
      "timestamp": "2024-02-09T15:30:45.123Z"
  }
  ```

- `connection_status`: WebRTC connection state changes
  ```json
  {
      "status": "connected|disconnected|failed",
      "timestamp": "2024-02-09T15:30:45.123Z"
  }
  ```

### Client → Server Events

- `model_input`: Send input to the model
  ```json
  {
      "content": "User input text",
      "type": "text|command",
      "timestamp": "2024-02-09T15:30:45.123Z"
  }
  ```

## Example Usage

### Establishing a WebRTC Connection

1. Get server status to ensure it's running:
```bash
curl http://localhost:8080/status
```

2. Submit WebRTC offer:
```bash
curl -X POST http://localhost:8080/webrtc/offer \
  -H "Content-Type: application/json" \
  -d '{
    "sdp": "v=0\no=- 4611731400...",
    "type": "offer",
    "client_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

3. Submit ICE candidates:
```bash
curl -X POST http://localhost:8080/webrtc/ice-candidate \
  -H "Content-Type: application/json" \
  -d '{
    "candidate": "candidate:1 1 UDP 2013266431...",
    "sdpMLineIndex": 0,
    "sdpMid": "0",
    "client_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

## Security Considerations

1. The server does not implement authentication by default. In production:
   - Use HTTPS
   - Implement proper authentication
   - Configure CORS appropriately
   - Use secure WebRTC configurations

2. Rate limiting helps prevent DoS attacks but additional security measures are recommended:
   - API key authentication
   - Request signing
   - IP-based blocking
