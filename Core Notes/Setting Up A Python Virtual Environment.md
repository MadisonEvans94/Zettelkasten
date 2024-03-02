
# Setting Up a Python Virtual Environment with Flask

## Introduction
This document guides you through the process of setting up a virtual Python environment and installing Flask, a popular web framework. Using virtual environments is a best practice in Python development, allowing you to manage project-specific dependencies without affecting the global Python installation.

## Prerequisites
- Python installed on your system (Python 3 is recommended).
- Basic familiarity with command line operations.

## Steps

### 1. Creating a Virtual Environment
First, navigate to your project's directory in the terminal or command prompt. Then, create a virtual environment.

**For Windows:**
```bash
python -m venv venv
```

**For macOS/Linux:**
```bash
python3 -m venv venv
```

`venv` is the name of the virtual environment directory. You can choose any name you prefer.

### 2. Activating the Virtual Environment
Activate the virtual environment to use it.

**For Windows:**
```bash
.\venv\Scripts\activate
```

**For macOS/Linux:**
```bash
source venv/bin/activate
```

Once activated, your command line will typically show the virtual environment's name.

### 3. Installing Flask
With the virtual environment active, install Flask using `pip`.

```bash
pip install Flask
```

### 4. Verifying Installation
To ensure Flask has been installed correctly, check the installed packages.

```bash
pip freeze
```

You should see `Flask` and its dependencies listed.

### 5. Deactivating the Virtual Environment
When you're done working in the virtual environment, you can deactivate it.

```bash
deactivate
```

This command returns you to the global Python environment.

## Conclusion
You now have a virtual Python environment with Flask installed. This setup is ideal for developing Flask-based web applications while maintaining a clean and organized development environment.

