# CreatorPulse - End-to-End YouTube Sentiment Analysis

## 🚀 Getting Started

This project is fully containerized using Docker. No local Python environment setup is required.

### Prerequisites

- Docker
- Docker Compose

### Installation & Running

1.  **Build and run the container:**
    From the project root directory, run the following command. This will build the Docker image and start the service.

    ```bash
    docker-compose up --build
    ```

2.  **Running in the background (detached mode):**

    ```bash
    docker-compose up -d
    ```

3.  **Stopping the application:**

    ```bash
    docker-compose down
    ```

4.  **Accessing a shell inside the container:**
    If you need to run commands inside the container's environment:

    ```bash
    docker-compose exec app bash
    ```
