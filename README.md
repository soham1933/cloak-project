
🧙‍♂️ Harry Potter Invisibility Cloak (OpenCV + MediaPipe)

Bring a bit of Hogwarts magic to life with Python!
This project recreates the famous Invisibility Cloak effect from Harry Potter using OpenCV and MediaPipe Selfie Segmentation.

When you wear a cloak of a chosen color (red/blue/green), the program detects it and replaces it with the background, making it appear invisible. ✨

📸 Demo
<img src="demo.gif" alt="Invisibility Cloak Demo" width="600"/>

(Replace demo.gif with your actual demo GIF or screenshot)

🚀 Features

Real-time video processing with OpenCV

Accurate person masking using MediaPipe

Support for multiple cloak colors (Red, Blue, Green)

Easy hotkeys for switching cloak color and capturing background

Interactive help overlay

Works with any standard webcam

🛠️ Installation
1. Clone this repo
git clone https://github.com/yourusername/invisibility-cloak.git
cd invisibility-cloak

2. Setup Python (⚠️ Important)

MediaPipe only supports Python 3.9 – 3.12.
Make sure you install Python 3.12.x (not 3.13+).

Create a virtual environment:

py -3.12 -m venv cloak-env
.\cloak-env\Scripts\Activate.ps1   # Windows PowerShell
# OR
source cloak-env/bin/activate      # Linux / macOS

3. Install dependencies
pip install mediapipe opencv-python numpy

▶️ Usage

Run the program:

python invisibility_cloak.py

🎮 Controls
Key	Action
B	Capture background (stand aside first!)
1	Set cloak color = Red
2	Set cloak color = Blue
3	Set cloak color = Green
H	Toggle help overlay
Q / ESC	Quit
⚙️ How It Works

Background Capture – A clean frame of your background is saved.

Color Detection (HSV) – Cloak pixels are detected by color (red/blue/green).

Person Segmentation (MediaPipe) – Ensures only cloak on the person becomes invisible.

Compositing – Cloak pixels are replaced with background pixels → invisibility effect.

🧩 Project Structure
invisibility-cloak/
│
├── invisibility_cloak.py    # Main script
├── README.md                # Project documentation
└── demo.gif                 # Demo video/gif (optional)

📦 Dependencies

OpenCV

MediaPipe

NumPy

🙌 Acknowledgements

Inspired by the Harry Potter Invisibility Cloak idea

Powered by Google MediaPipe and OpenCV

💡 Future Improvements

Add a live HSV range tuner for different cloak shades

Support for recording video with the invisibility effect

Mobile version using Android/iOS camera

🪄 Author

Created with ❤️ by soham bandbe,
If you like this repo, ⭐ star it on GitHub!
