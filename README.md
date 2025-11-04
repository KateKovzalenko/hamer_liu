# HaMeR-LiU: Hand Mesh tracking

## Adaption of the HAMER repository.

For more information about the original readme.md, refers to the **[Original README.DM](./docs/README_original.md)** file.

## Docker Installation

This procedure details setting up and running the HaMeR project using Docker and the NVIDIA Container Toolkit. This is the **recommended method** as it provides a reproducible, isolated environment that resolves all complex Python dependency conflicts (e.g., numpy, opencv, xtcocotools).

These instructions are tailored for **Windows 11 (Home/Pro)** with an **NVIDIA GPU**.

### 1\. Host System Prerequisites (Windows 11)

Your host system must be configured to provide GPU access to Docker containers.

**Install NVIDIA Drivers:** Ensure you have the latest NVIDIA Game Ready or Studio Drivers installed for your GPU.

**Install/Enable WSL 2:** Docker Desktop on Windows requires the Windows Subsystem for Linux (WSL) 2 for GPU passthrough. Windows 11 Home editions fully support this.

  * Open PowerShell as Administrator and run:
    ```bash
    wsl --install
    ```
  * Reboot your system if prompted.

**Install Docker Desktop:**

  * Download and install Docker Desktop for Windows.
  * During setup, ensure it is configured to use the "WSL 2 based engine." This is the default.
  * After installation, navigate to **Settings \> Resources \> WSL Integration** and ensure "Enable integration with my default WSL distro" is checked.

### 2\. Project & Asset Setup

**Clone the Repository:** Clone the `hamer` repository, including its submodules (like ViTPose).

```bash
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
```

**Download MANO Model:** The MANO model is required but cannot be redistributed due to its license.

  * Visit the [MANO website](https://mano.is.tue.mpg.de/) and register to access the downloads section.
  * Download the right hand model (`MANO_RIGHT.pkl`).
  * Create the required data directory structure and place the file there. The final path on your host machine must be:
    `hamer/_DATA/data/mano/MANO_RIGHT.pkl`

### 3\. Build and Launch the Container

The provided Docker configuration will build the environment with all pinned dependencies (PyTorch, CUDA 11.8, correct NumPy/OpenCV versions).

**Build the Image:** From the root `hamer` directory, run the following command. This will take up to one hour as it builds the `Dockerfile` specified in `docker/docker-compose.yml`.

```bash
docker compose -f ./docker/docker-compose.yml up -d --build
```

  * `--build`: Forces a new build of the image.
  * `-d`: Runs the container in detached (background) mode.

**Verify Container is Running:** Check that the `hamer-dev` container is running.

```bash
docker ps
```

You should see an entry for `hamer-dev`.

### 4\. Running the Demo

**Enter the Container:** Access the running container's shell.

```bash
docker compose -f ./docker/docker-compose.yml exec hamer-dev /bin/bash
```

**Fetch Demo Data:** Once inside the container, download the pre-trained HaMeR models.

```bash
bash fetch_demo_data.sh
```

**Run the Demo:** Execute the demo script on the example data.

```python
python demo.py \
    --img_folder example_data --out_folder demo_out \
    --batch_size=48 --side_view --save_mesh --full_frame
```

Because the `docker-compose.yml` file mounts the local directory (`../:/app` relative to the compose file, which is the project root), the `demo_out` folder will appear on your host machine's `hamer/demo_out` directory.

______________________________________________________________________________
**Temporary**

From the beginning

 Project & Asset Setup

**Clone the Repository:** Clone the `hamer` repository, including its submodules (like ViTPose).

```bash
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer


**Download MANO Model:**

The MANO model is required but cannot be redistributed due to its license.

  * Visit the [MANO website](https://mano.is.tue.mpg.de/) and register to access the downloads section.
  * Download the right hand model (`MANO_RIGHT.pkl`).
  * Create the required data directory structure and place the file there. The final path on your host machine must be:
    `hamer/_DATA/data/mano/MANO_RIGHT.pkl`

> This section is temporary.  

Contains instructions for running the Docker file for hand tracking server, located in the docker folder.  

Future plans:  
- Add Hamer processing.  
- After integrating Hamer to hand tracking, remove code related to MediaPipe.

Temporary Notice:
Until dev-ht-24-interface changes are merged into the dev branch, please use this branch to get the latest hand tracking integration updates.

# Switch to our temporary development branch
git fetch origin dev-ht-24-interface
git checkout dev-ht-24-interface
git pull origin dev-ht-24-interface

**Build the Docker Image**

docker build -t docker-hamer-server -f docker/docker_hamer_server.Dockerfile .

**Run the Docker Container**

docker run --gpus all -v ${PWD}:/app -it --rm -p 8080:5000 --name docker-hamer-server  docker-hamer-server:latest

**Run in the Terminal**

python -m server.server


**Test the Application**

You have two ways to test your running API:

### 1\. Using the Web Browser (GUI)

1.  Open your favorite web browser (like Chrome, Firefox, or Edge).
2.  Navigate to the following URL: **[http://localhost:8080](https://www.google.com/search?q=http://localhost:8080)**
3.  You should see a simple webpage with an "Upload an Image" form.
4.  Use the form to upload your image to see the JSON results.

### 2\. Using the Python Client Script

This method provides a visual confirmation by plotting the results on the image.

1.  **Install Client Dependencies**: In your terminal, install the required Python libraries for the client script.
    ```powershell
    pip install requests matplotlib Pillow
    ```
2.  **Run the Client**: Execute the `client.py` script.

    ```powershell
    python client/client.py --input "path"
    ```
