 #!/usr/bin/env bash

# 1. Network Optimizations
echo "Applying UDP buffer optimizations..."
sudo sysctl -w net.core.rmem_max=26214400
sudo sysctl -w net.core.wmem_max=26214400
sudo sysctl -w net.core.rmem_default=26214400
sudo sysctl -w net.core.wmem_default=26214400

# Identify active interface and apply MTU fix
# We take the interface down, set MTU, and bring it up to avoid 'Device or resource busy'
INTERFACE=$(ip route get 8.8.8.8 | awk '{print $5; exit}')

if [ -n "$INTERFACE" ]; then
    echo "Cycling $INTERFACE to set MTU to 1300..."
    sudo ip link set "$INTERFACE" mtu 1300
    echo "MTU successfully updated."
else
    echo "Error: Could not determine active network interface."
fi

# 2. Cleanup existing sessions
echo "Cleaning up existing Xvfb, Openbox, and Sunshine sessions..."
sudo pkill -9 Xvfb 2>/dev/null
sudo pkill -9 openbox 2>/dev/null
sudo pkill -9 sunshine 2>/dev/null
sudo pkill -9 xclock 2>/dev/null
sleep 2

# 3. Initialize Virtual Display
echo "Starting Virtual Display (Xvfb) on :99..."
Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset > /dev/null 2>&1 &
sleep 2

echo "Starting Window Manager (Openbox)..."
DISPLAY=:99 openbox > /dev/null 2>&1 &
sleep 1

# Start xclock to keep the encoder "pulsing" 
DISPLAY=:99 xclock -digital -update 1 > /dev/null 2>&1 &

# 4. Launch Sunshine
# Using the cores 8-11 isolation you found stable
echo "Launching Sunshine on cores 8,9,10,11..."
sudo env DISPLAY=:99 taskset -c 8,9,10,11 /usr/bin/sunshine-v0.23.1 > /dev/null 2>&1 &
echo "---------------------------------------------------"
echo "Setup Complete! Moonlight should now be able to connect."
echo "Instruct the team to prefix their commands with DISPLAY=:99"
echo "Example: DISPLAY=:99 python demo.py"
echo "---------------------------------------------------"

