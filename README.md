#Horn AI: Intelligent Real-time Vehicle Detection and Horn Intensity Control System
Overview
Horn AI is an intelligent real-time vehicle detection and horn intensity control system aimed at reducing noise pollution and enhancing road safety. The project leverages the YOLOv8 model for vehicle detection and OpenCV for video processing.
Features
- Real-time vehicle detection using YOLOv8
- Horn intensity control to reduce noise pollution
- Video processing using OpenCV
Installation
To get started with the project, follow the steps below:
1. Clone the repository:
git clone https://github.com/yourusername/horn-ai.git
cd horn-ai
2. Create a virtual environment:
python3 -m venv venv
source venv/bin/activate   # On Windows use `venv\Scripts\activate`
3. Install the required dependencies:
pip install -r requirements.txt
Usage
1. Run the detection script for images:
python detection_image.py
2. Run the detection script for videos:
python detection_video.py
Project Structure
- `detection_image.py`: Script for detecting vehicles in images.
- `detection_video.py`: Script for detecting vehicles in videos.
- `requirements.txt`: List of dependencies required for the project.
Dependencies
The project requires the following Python packages:
- opencv-python
- torch
- torchvision
- numpy

For a complete list of dependencies, refer to the `requirements.txt` file.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Contact
For any questions or suggestions, feel free to reach out:
- Email: Harishwar.ds.ai@gmail.com
- GitHub: github.com/harishwarj
