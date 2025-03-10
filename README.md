# StrapTo Local Server

*Disclaimer: This repository is currently under active development and not all features are available yet.*

StrapTo is the provider-side component of the StrapTo suite, designed to capture and stream outputs from self-hosted AI models (e.g., Ollama, LMStudio, Llama.cpp, LangChain/Langgraph, etc.) in real time. Built in Python with the [aiortc](https://aiortc.readthedocs.io/) library, the local server handles low-latency WebRTC connections and facilitates bi-directional communication—allowing consumer inputs to be fed back into your model.

## Overview

The StrapTo Local Server enables model providers to:
- Capture diverse outputs (text, JSON, images, etc.) from self-hosted AI models.
- Stream these outputs in real time to remote consumers using WebRTC.
- Receive interactive input from consumers, which can be routed back into the model.
- Manage the streaming session with controls for starting, pausing/resuming, and stopping the stream.

This repository focuses solely on the local server component. For a complete interactive streaming experience, see the other StrapTo repositories:
- **StrapTo Host:** Manages room creation, WebRTC signaling, and real-time communication (built with Node.js, Express, Socket.io, and PeerJS).
- **StrapTo Client:** A modern React-based interface for both model providers and consumers.

## Features

- **Real-Time Streaming:**  
  Transmit any type of model output (plain text, formatted JSON, images, etc.) with low latency.

- **Bi-Directional Communication:**  
  Capture consumer messages (chat, suggestions, votes) and relay them to the local model, enabling interactive sessions.

- **Flexible Integration:**  
  Designed to plug into a variety of self-hosted AI model frameworks, making it easy to adapt to your existing setup.

- **Provider Control:**  
  Offers session management functions to start, pause/resume, and stop the stream, giving you full control over the interactive experience.

- **Modular Architecture:**  
  Operates as a standalone Python service that seamlessly integrates with other StrapTo components to form a complete interactive streaming solution.

## Architecture Overview

The StrapTo Local Server is built in Python using `aiortc` to handle the real-time aspects of WebRTC communications. Its core responsibilities include:

- **Capturing Model Outputs:**  
  Interfacing directly with your self-hosted AI model to capture and stream outputs in real time.

- **Establishing WebRTC Connections:**  
  Setting up and maintaining peer-to-peer data channels for efficient, low-latency transmission of model outputs to connected consumers.

- **Routing Consumer Inputs:**  
  Receiving bi-directional messages from consumers via established signaling channels and relaying these inputs back into the model. This enables interactive, two-way sessions where feedback can influence the model’s behavior.

- **Interoperability with Other Components:**  
  While this repository focuses on the local server, it is designed to work in concert with:
  - A dedicated signaling server that handles room management and the WebRTC handshake.
  - A React-based frontend that provides an interactive interface for both the model provider and end users.

## Use Cases

- **Live Model Streaming:**  
  Broadcast live outputs from your local AI model to an audience, ideal for demonstrations, interactive sessions, or remote collaboration.

- **Interactive Feedback Loop:**  
  Enable real-time consumer interaction where inputs like chat messages or suggestions can directly influence ongoing model outputs.

- **Modular Integration:**  
  Seamlessly integrate into various self-hosted AI workflows, providing a plug-and-play solution for real-time streaming and interaction.

## Getting Started

*Coming soon...*
