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


**Clone the Repository:** 
Clone the `hamer` repository, including its submodules (like ViTPose).

```bash
git clone --recursive https://github.com/KateKovzalenko/hamer_liu
cd hamer
```

**Download MANO Model:**

The MANO model is required but cannot be redistributed due to its license.

  * Visit the [MANO website](https://mano.is.tue.mpg.de/) and register to access the downloads section.
  * Download the right hand model (`MANO_RIGHT.pkl`).
  * Create the required data directory structure and place the file there. The final path on your host machine must be:
    `hamer/_DATA/data/mano/MANO_RIGHT.pkl`


Contains instructions for running the Docker file for hand tracking server, located in the docker folder.  

Future plans:  
- Add Hamer processing.  
- After integrating Hamer to hand tracking, remove code related to MediaPipe.


# Switch to our temporary development branch
git fetch origin dev
git checkout dev
git pull origin dev

**Build the Docker Image**

	DEV CONTAINER PHASE
	* Open the files in VisualStudio Code
	* Press F1 to open search bar
	* Choose : 'dev container- open folder in container'
	* in pop-up window choose HaMeR repository file on your device
	* Notice visible log in terminal of the container being build
	* Container is build when dev container terminal displays the port
	
	PYTHON DEBUG CONSOLE PHASE
	* Press : Ctrl+Shift+P
	* Choose: Python: Select interpreter > Python 3.10.112 (venv)
	* Run using (in options:) Python Debugger : Debug using launch.json
	* Web app is running when debugger console displays the port 


### 3\. Test the Application**

You have two ways to test your running API:

*** Using the Web Browser (GUI) ***

1.  Open your favorite web browser (like Chrome, Firefox, or Edge).
2.  Navigate to the following URL: **[http://localhost:8080](https://www.google.com/search?q=http://localhost:8080)**
3.  You should see a simple webpage with an "Upload an Image" form.
4.  Use the form to upload your image to see the JSON results.
5.  WARNING:
	if you choose image in video browsing window, it will be treated as single-frame video
	if you choose video in image browsing window, you will get an error

*** Using the Python Client Script ***

This method provides a visual confirmation by plotting the results on the image.

1.  **Install Client Dependencies**: In your terminal, install the required Python libraries for the client script.
    ```powershell
    pip install requests matplotlib Pillow
    ```
2.  **Run the Client**: Execute the `client.py` script.

    ```powershell
    python client/client.py --input "path"
    ```
