FROM python:3.8-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-tk \
    libgl1-mesa-glx \
    libglib2.0-0 \
    usbutils \
    libqt5gui5 \
    libqt5core5a \
    libqt5widgets5 \
    libxcb-xinerama0 \
    libxkbcommon-x11-0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# Set QT platform and disable GUI threading warnings
ENV QT_X11_NO_MITSHM=1
ENV QT_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins
ENV QT_DEBUG_PLUGINS=1
ENV PYTHONUNBUFFERED=1
ENV OPENCV_VIDEOIO_PRIORITY_MSMF=0

# Install Python dependencies
COPY .docker/req.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


WORKDIR /app

# Set environment variable for OpenCV GUI
ENV DISPLAY=:0
