# 🏥 Stroke Detection Application - Beginner's Guide

## 📋 What You Need Before Starting
- A computer with **Python** installed (version 3.8 or newer)
- Basic familiarity with using the **terminal/command prompt**
- **10-15 minutes** for first-time setup
- **Internet connection** for downloading packages

## 🐍 Installing Python (If You Don't Have It)

### Check if Python is Already Installed
1. Open terminal/command prompt (see Step 1 below for how)
2. Try these commands (try all until one works):
   - `python --version`
   - `python3 --version`
   - `py --version` (Windows)
3. If you see "Python 3.8.x" or higher (like 3.9, 3.10, 3.11, 3.12), you're good to go!
4. If you see Python 2.x.x or get an error, you need to install Python 3.8+

### If Python is Not Installed:

**Windows:**
1. Go to [python.org](https://python.org)
2. Click "Download Python" (latest version)
3. Run the installer
4. **IMPORTANT**: Check "Add Python to PATH" during installation
5. Restart your computer

**Mac:**
1. Go to [python.org](https://python.org)
2. Download Python for macOS
3. Run the installer
4. Or use Homebrew: `brew install python`

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

## 🆕 First Time Setup on a New Computer

If this is your first time setting up this application on a new computer, follow these steps:

### Option A: Download the Application Files

1. **Get the application files** from your source (USB drive, email, cloud storage, etc.)
2. **Extract/Copy** all files to a folder on your computer
3. **Remember the folder location** - you'll need this path later

### Option B: If You Have Git (Advanced Users)
```bash
git clone [repository-url]
cd stroke-detection
```

### Create Python Virtual Environment
Once you have the files on your computer:

1. **Open terminal** and navigate to your application folder:
   ```bash
   cd /path/to/your/stroke-detection
   ```
   > Replace `/path/to/your/stroke-detection` with the actual folder path

2. **Create a virtual environment** (use the command that worked for checking your Python version):
   ```bash
   # Try one of these commands:
   python3 -m venv stroke_prediction_env     # Linux/Mac
   python -m venv stroke_prediction_env      # Windows/Some Linux
   py -m venv stroke_prediction_env          # Windows (Microsoft Store Python)
   ```
   ⏱️ **Wait time**: This takes 1-2 minutes

3. **Activate the environment**:
   - **Linux/Mac**: `source stroke_prediction_env/bin/activate`
   - **Windows (Command Prompt)**: `stroke_prediction_env\Scripts\activate`
   - **Windows (PowerShell)**: `stroke_prediction_env\Scripts\Activate.ps1`
   - **Windows (Git Bash)**: `source stroke_prediction_env/Scripts/activate`

4. **Install all required packages**:
   ```bash
   pip install flask numpy pandas scikit-learn xgboost seaborn matplotlib openpyxl
   ```
   ⏱️ **Wait time**: This takes 3-5 minutes

✅ **Setup Complete!** You only need to do this once per computer.

---

## 🚀 How to Start the Application (After Setup)

### Step 1: Open Your Terminal
- **Windows**: Press `Win + R`, type `cmd`, press Enter
- **Mac**: Press `Cmd + Space`, type `Terminal`, press Enter  
- **Linux**: Press `Ctrl + Alt + T`

### Step 2: Navigate to the Application Folder
Copy and paste this command, then press Enter (replace with your actual path):
```bash
cd /path/to/your/stroke-detection
```
> **Example paths:**
> - Windows: `cd C:\Users\YourName\stroke-detection`
> - Mac/Linux: `cd /home/username/stroke-detection`

### Step 3: Activate the Python Environment
This prepares Python to run our application:
- **Linux/Mac**: `source stroke_prediction_env/bin/activate`
- **Windows (Command Prompt)**: `stroke_prediction_env\Scripts\activate`
- **Windows (PowerShell)**: `stroke_prediction_env\Scripts\Activate.ps1`
- **Windows (Git Bash)**: `source stroke_prediction_env/Scripts/activate`

✅ **Success indicator**: You should see `(stroke_prediction_env)` at the start of your command line

### Step 4: Verify You're in the Right Directory
**Note**: The application files (like app.py) should be in the main directory where you are. You can verify by listing the files:

**Linux/Mac/Git Bash:**
```bash
ls
```

**Windows (Command Prompt):**
```cmd
dir
```

You should see files like `app.py`, `requirements.txt`, and folders like `templates`, `static`.

### Step 5: Install Required Software Packages
This downloads all the tools our application needs:

**Option A: Install from requirements file (recommended):**
```bash
pip install -r requirements.txt
```
⏱️ **Wait time**: This may take 5-10 minutes to complete

**Option B: If Option A fails or times out, install packages individually:**
```bash
pip install flask numpy pandas scikit-learn xgboost matplotlib seaborn openpyxl
```
⏱️ **Wait time**: This may take 8-12 minutes to complete

**Note**: Package installation times vary based on your internet connection and computer speed. Be patient and let the process complete.

### Step 6: Start the Application
Use the same Python command that worked for you earlier:
```bash
# Try one of these commands:
python app.py        # Most common
python3 app.py       # Linux/Mac
py app.py            # Windows (Microsoft Store Python)
```

🎉 **Success!** You should see messages like:
- "🚀 Initializing XAI Engine..."
- "✅ XAI Engine ready for explanations!"
- "Running on http://127.0.0.1:5002"

**Note**: The application runs on port 5002. If you see a message about port being in use, the application will automatically try the next available port.

### Step 7: Open the Application in Your Browser
Click this link or copy it to your web browser:
**http://127.0.0.1:5002**

## 🌐 What Can This Application Do?

### 🏠 Main Pages You Can Visit
Once the app is running, you can access these pages in your browser:

1. **Home Page**: http://127.0.0.1:5002/
   - Main landing page with information about the app

2. **Individual Prediction**: http://127.0.0.1:5002/prediction
   - Enter a person's health information to predict stroke risk
   - Get instant results with explanations

3. **Admin Login**: http://127.0.0.1:5002/login
   - Username: `admin@gmail.com`
   - Password: `admin`
   - Access data upload features

4. **Data Upload**: http://127.0.0.1:5002/upload (after login)
   - Upload Excel or CSV files with multiple patients' data
   - Process many predictions at once

### 🎯 What the App Does (In Simple Terms)
- **Predicts stroke risk** for individuals based on health factors like age, blood pressure, etc.
- **Explains why** it made that prediction (using AI explanation technology)
- **Risk levels**: Gives results from LOW risk to EXTREME risk
- **Processes files**: Can analyze data for many people at once from spreadsheet files

## ⏹️ How to Stop the Application

When you're done using the app:
1. Go back to your terminal (where you started the app)
2. Press `Ctrl + C` (hold Ctrl key and press C)
3. The app will stop and you can close the terminal

## 🔧 Common Problems and Solutions

### Problem 1: "Virtual environment doesn't work"
**What to do:**
```bash
python3 -m venv stroke_prediction_env
source stroke_prediction_env/bin/activate
```
This creates a fresh Python environment.

### Problem 2: "Can't install packages" 
**What to do:** Install them one by one:
```bash
pip install flask
pip install numpy
pip install pandas
pip install scikit-learn
pip install xgboost
pip install matplotlib
pip install seaborn
```

### Problem 3: "Address already in use" or "Port busy"
**What this means:** Another program is using the same web address
**What to do:**
1. Open the file `app.py` in a text editor
2. Find the last line that says: `app.run(debug=True, port=XXXX)`
3. Change the port number to a different one (try 5003, 5004, etc.)
4. Save the file and run the app again
5. Now visit: http://127.0.0.1:[NEW_PORT] instead

**Example**: If you change to port 5003, visit http://127.0.0.1:5003

### Problem 4: "Python not found" 
**What to do:**
- Make sure Python is installed on your computer
- Try using `python3` instead of `python` in commands
- Download Python from [python.org](https://python.org) if needed

## 📁 What's Inside This Application
- **app.py** - The main program that runs everything
- **templates/** - The web pages you see in your browser
- **static/** - Pictures, styles, and interactive elements  
- **model/** - The AI brain that makes predictions
- **requirements.txt** - List of software packages needed

---
## 💡 Tips for Beginners
- **Copy commands exactly** - Don't change spacing or punctuation
- **Wait for commands to finish** - Some steps take time, be patient
- **Keep terminal open** - Don't close it while the app is running
- **Use Ctrl+C to stop** - This is the universal "stop" command

## 📋 Quick Reference (After First Setup)

### Daily Use Commands (Copy & Paste)
```bash
# Navigate to your app folder
cd /path/to/your/stroke-detection

# Activate Python environment
source stroke_prediction_env/bin/activate              # Linux/Mac
# OR
stroke_prediction_env\Scripts\activate                 # Windows (Command Prompt)
# OR  
stroke_prediction_env\Scripts\Activate.ps1             # Windows (PowerShell)
# OR
source stroke_prediction_env/Scripts/activate          # Windows (Git Bash)

# Verify you're in the right directory (should see app.py)
ls          # Linux/Mac/Git Bash
# OR
dir         # Windows Command Prompt

# Start the app (use the Python command that works for you)
python app.py        # Most common
# OR
python3 app.py       # Linux/Mac  
# OR
py app.py            # Windows (Microsoft Store Python)

# Open in browser: http://127.0.0.1:5002
```

### Important URLs
- **Main App**: http://127.0.0.1:5002
- **Predictions**: http://127.0.0.1:5002/prediction  
- **Login**: http://127.0.0.1:5002/login (admin@gmail.com / admin)

### Stop the App
- Press `Ctrl + C` in terminal