# **HaMeR-LiU: Hand Mesh Tracking**

## **Adaptation of the HaMeR Repository**

This project functions as a wrapper and adaptation of the original HaMeR repository for hand mesh tracking. For information regarding the underlying logic, refer to the [**Original README**](https://www.google.com/search?q=./docs/README_original.md).

## **1\. System Prerequisites**

The following configuration is required to establish the runtime environment. These instructions are validated for **Windows 11 (Home/Pro)** using an **NVIDIA GPU**.

### **Host Configuration**

1. **NVIDIA Drivers:** Install the latest [NVIDIA Game Ready or Studio Drivers](https://www.nvidia.com/Download/index.aspx) for your specific GPU (RTX 4070 Ti SUPER or equivalent).  
2. **WSL 2:** Docker Desktop on Windows requires the Windows Subsystem for Linux (WSL) 2 for GPU passthrough.  
   * Open PowerShell as Administrator and execute:  
     wsl \--install

   * Reboot the system if prompted.  
3. **Docker Desktop:**  
   * Download and install [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).  
   * During installation, ensure **"Use WSL 2 based engine"** is selected.  
   * Post-installation: Navigate to **Settings \> Resources \> WSL Integration** and ensure "Enable integration with my default WSL distro" is active.

## **2\. Project & Asset Setup**

### **Repository Initialization**

Clone the repository and its submodules. Switch to the active development branch immediately after cloning.

\# Clone recursive to include ViTPose and other dependencies  
git clone \--recursive \[https://github.com/KateKovzalenko/hamer\_liu\](https://github.com/KateKovzalenko/hamer\_liu)  
cd hamer

\# Switch to the development branch  
git fetch origin dev  
git checkout dev  
git pull origin dev

### **MANO Model Acquisition**

The MANO model is proprietary and cannot be redistributed. It must be manually acquired and placed in the correct directory structure to allow the container to mount it.

1. Register and download the model from the [MANO website](https://mano.is.tue.mpg.de/).  
2. Download the right-hand model: MANO\_RIGHT.pkl.  
3. Place the file in the following local directory structure within the cloned repository:  
   hamer/\_DATA/data/mano/MANO\_RIGHT.pkl

## **3\. Development Environment Setup**

This project utilizes **VS Code Dev Containers** to create an isolated, reproducible Python environment with resolved dependencies (NumPy, OpenCV, PyTorch, XTCocoTools).

### **Dev Container Initialization**

1. Open the hamer folder in **Visual Studio Code**.  
2. Access the Command Palette (F1 or Ctrl+Shift+P).  
3. Select: **Dev Containers: Reopen in Container**.  
   * *Note: If prompted, select the definition file located in the .devcontainer directory.*  
4. The terminal will display build logs. The initialization is complete when the terminal within VS Code becomes interactive and displays the container's shell prompt.

## **4\. Execution & Debugging**

Once inside the Dev Container, the Python environment is pre-configured.

### **Launching the Web Server**

1. Open the Command Palette (Ctrl+Shift+P).  
2. Select: **Python: Select Interpreter**.  
3. Choose the virtual environment path: Python 3.10.x (venv).  
4. Navigate to the **Run and Debug** view (Ctrl+Shift+D).  
5. Select **"Debug using launch.json"** (or the specific configuration name defined in .vscode/launch.json) and press F5.  
6. The Debug Console will indicate execution. The server is active when the port (default: 8080\) is displayed.

## **5\. Client Usage & Testing**

### **Option A: Web Interface (GUI)**

1. Open a browser on the host machine.  
2. Navigate to: http://localhost:8080  
3. Upload an image via the form to view the JSON output.

**WARNING:**

* Do not upload video files via the image selection input (System Error).  
* Images uploaded via the video selection input will be processed as a single-frame video.

### **Option B: Python Client Script**

This method validates the API programmatically and visualizes the mesh tracking on the input image.

1. Install Client Dependencies (Host Machine):  
   Open a terminal on your host machine (not the container) and install the visualization requirements:  
   pip install requests matplotlib Pillow

2. Execute Client:  
   Run the client script targeting the running local server.  
   \# Replace "path/to/image.jpg" with your actual file path  
   python client/client.py \--input "path/to/image.jpg"

## **Architecture Notes & Roadmap**

**Current Status:**

* Docker containerization allows for consistent CUDA runtimes across different host GPU drivers.  
* MediaPipe integration is currently active.

**Future Roadmap:**

* \[ \] Full HaMeR processing integration.  
* \[ \] Deprecation of MediaPipe logic following HaMeR stabilization.