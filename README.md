# Depth Estimator with Skeleton Detection

This repository is modified from [Depth-Estimator-with-ArUco-Corner-Detection](https://github.com/bebo1010/Depth-Estimator-with-ArUco-Corner-Detection)

## Demo Video
TODO

## Environment Setup

1. Install Python 3.8 or higher.
2. Install the required packages:
    ```bash
    pip install -e .
    ```
3. Install the PySpin library:
    - Download the appropriate PySpin wheel file from [here](https://www.flir.com/products/spinnaker-sdk/)
    - [PySpin 4.0.0.116, Python 3.8](https://drive.google.com/file/d/1G4BkDU8xr4Tgu4M9vk-Q2gX3HwvO-WSZ/view?usp=sharing)
    - Install the wheel file:
        ```bash
        pip install <path_to_wheel_file>
        ```
4. Install Pytorch
    ```bash
    pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
    pip install cython_bbox
    ```
    Remove `anaconda3\envs\your_envs_name\libiomp5md.dll`
5. Setup conda environment
    ```bash
    conda install -c conda-forge faiss-gpu
    ```
6. Setup Openmim
    ```bash
    pip install -U openmim
    mim install mmcv-full
    pip install mmcv==2.0.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
    mim install "mmdet==3.1.0"
    ```
7. Setup mmengine related packages
    ```bash
    cd Mmengines\mmpose_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\

    cd Mmengines\mmyolo_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\

    mim install mmengine

    cd Mmengines\mmpretrain_main
    pip install -r requirements.txt
    pip install -v -e .
    cd ..\..\
    ```
8. Setup Bytetrack
    ```bash
    cd Bytetrack
    pip install -r requirements.txt
    python setup.py develop
    ```
9. Run program
    ```bash
    python -m src.main
    ```

## Functionality
- [x] Detect ArUco markers from left and right image
    - [x] Compute depths with detected ArUco markers
- [x] Include interfaces for other cameras
    - [x] Interface for intel RealSense cameras
        - [x] Single RealSense camera
        - [x] Dual RealSense camera
    - [x] Interface for FLIR cameras
    - [x] Add support for multiple RealSense cameras
- [x] Display Functionality
    - [x] Horizontal lines and Vertical lines
    - [x] Epipolar lines
        - Display epipolar lines from key points of scene
        - The method for detecting key points defaults to `ORB`
            - Can be swapped to `SIFT`
    - [x] Freeze frame
    - [x] Information Panel
        - [ ] Not implemented yet
        - [x] Mouse hover 3D position
- [x] Skeleton functionality
    - [x] 2D skeleton on image
    - [x] 3D skeleton in Open3D display
- [ ] Chessboard calibration for stereo camera
    - [x] Calibration and save image
    - [ ] Load back the parameters and rectify the images
    - [ ] (Optional) Show reprojection error per image
- [x] Auto rectification with fundamental matrix
- [ ] Load back the videos
    - [x] Include camera parameters like focal length, baseline, etc
- [x] Unit Tests (Need to check again)
    - [x] ArUco detector
    - [x] Camera systems (not possible for FLIR cameras)
    - [x] Utility functions
    - [x] Epipolar line detector
    - [x] Chessboard Calibration
    - [x] File utility functions
    - [x] Display utility functions

### buttons for opencv_ui_controller.py

- `h` or `H` to show horizontal lines
- `v` or `V` to show vertical lines
- `e` or `E` to show epipolar lines
    - `n`, `N`, `p`, or `P` to change algorithm
- `s` or `S` to save the images
- `f` or `F` to freeze frame
- `a` or `A` to display detected ArUco marker
- `c` or `C` to toggle on calibration mode
    - `s` or `S` to save chessboard image
    - `c` or `C` to toggle off calibration mode and start calibration
- `l` or `L` to load back previous saved images
    - `n`, `N`, `p`, or `P` to change image pairs
- `esc` to close program

## Goal
- Loading back the videos
- Allow loading back calibration parameters.

## Note
> [!NOTE]
> config for ORYX cameras are custom made, as the trigger lines can be connected differently.

> [!WARNING]
> config for ORYX cameras are still untested, require further testing to make sure it works.

- To run linting check:
```bash
python -m pylint ./src/**/*.py --max-line-length=120 --disable=E1101,E0611,E0401,E0633,R0801 --max-args=10 --max-locals=30 --max-attribute=15
```
> [!NOTE]
> `E1101`: No member error. Suppressing this for opencv-python and pyrealsense2 packages.
> `E0611`: No name in module error.  Suppressing this for opencv-python and pyrealsense2 packages.
> `E0401`: Unable to import error. Suppressing this for unable to install PySpin on workflow dispatch.
> `E0633`: Unpacking non sequence error. Suppressing this for ArUco detector.
> `R0801`: Duplicate code between files warning. Suppressing this for main functions.
> `R0903`: Too few public methods warning. Raised when a class has less than 2 public methods.
> `max-line-length`: Limits max characters per line.
> `max-args`: Limits max arguments for a function.
> `max-locals`: Limits max local variables within a function.
> `max-attribute`: Limits max instance attribute for a class.

- To run unit tests:
```bash
python -m unittest discover -s ./tests -v
```
> [!NOTE]
> discover all the unit tests in `./test` and run all tests