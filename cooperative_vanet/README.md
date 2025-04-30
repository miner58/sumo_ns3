# Cooperative Vehicular Networks Simulation

This project reproduces the simulation environment for the paper: *"Cooperative Vehicular Networks: An Optimal and Machine-Learning Approach"* (Computers & Electrical Engineering, 2022). It combines NFV-enabled Road-Side Units (RSUs) with a Mobile-Edge-Computing (MEC) server that runs a stacked-LSTM traffic-flow predictor.

## Overview

The simulation integrates:
- **SUMO** for realistic traffic simulation
- **NS-3** for network communication simulation
- **PyTorch Lightning** for machine learning (traffic prediction)
- **Optimal stopping theory** for intelligent relay selection

## Project Structure

```
cooperative_vanet/
├── configs/           # Configuration files for simulations
├── data/              # Data directory
│   ├── raw/           # Raw data from simulations
│   └── processed/     # Processed data for ML and analysis
├── docs/              # Documentation
├── scripts/           # Utility scripts
├── src/               # Source code
│   ├── ml/            # Machine learning components
│   ├── ns3_integration/  # NS-3 integration code
│   ├── sumo_integration/ # SUMO integration code
│   ├── relay_selection/  # Relay selection algorithms
│   └── utils/         # Utility functions
├── tests/             # Test files
├── README.md          # This file
└── requirements.txt   # Dependencies
```

## Prerequisites

### Software Requirements
- Python 3.8+
- NS-3 v3.37 (net-dev branch with IEEE 802.11p stack)
- SUMO 1.19
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Hardware Recommendations
- Modern CPU (8+ cores)
- 16 GB RAM
- GPU with 8+ GB memory (optional, for faster ML training)

## Setup Instructions

### 1. NS-3 Setup

NS-3 requires specific installation steps:

```bash
# 의존성 패키지 설치
sudo apt-get install g++ cmake python3 mercurial -y

# select one of two options
# option1
wget https://www.nsnam.org/releases/ns-allinone-3.44.tar.bz2
tar xfj ns-allinone-3.44.tar.bz2
cd ns-allinone-3.44/ns-3.44

# option2
git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev
git checkout -b ns-3.44-release ns-3.44

# builging ns-3
./ns3 configure --enable-examples --enable-tests

# test ns-3
./test.py

# Set environment variables (add to your .bashrc)
# ~/.bashrc  (또는 ~/.zshrc 등 셸 설정 파일)

## 1) ns-3 소스/빌드 디렉터리 지정
export NS3_DIR="/home/jsw-docker/sumo_ns3/ns-allinone-3.44/ns-3.44"

## 2) 파이썬 바인딩 모듈 경로 추가
export PYTHONPATH="$NS3_DIR/build/bindings/python:$PYTHONPATH"

## 3) (권장) C++ 공유 라이브러리 경로 추가
export LD_LIBRARY_PATH="$NS3_DIR/build/lib:$LD_LIBRARY_PATH"

## 4) (선택) waf 빌드 스크립트를 바로 쓰고 싶다면
export PATH="$NS3_DIR:$PATH"

# 설정 저장 후
source ~/.bashrc      # 현재 셸에 즉시 적용
# 또는 새 터미널을 열기

# python binding
sudo apt install python3-pip bzr
# ns-3.42 and newer:
python3 -m pip install --user cppyy==3.1.2
# add path
nano ~/.bashrc
# 2) 파일 맨 아래에 추가
# ─────────────────────────────────────
# 사용자 전용 Python 스크립트 경로
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc    # 또는 새 터미널 열기

```

### 2. SUMO Setup

```bash
# Install SUMO
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt install sumo sumo-tools sumo-doc

# Set environment variables (add to your .bashrc)
#export SUMO_HOME="/usr/share/sumo"
#export PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH
echo -e '\nexport SUMO_HOME="/usr/share/sumo"\nexport PYTHONPATH=$SUMO_HOME/tools:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

```

### 3. Python Environment Setup

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```

## Simulation Components

### 1. Traffic Simulation (SUMO)
The traffic simulation uses a 1500m × 750m rectangular urban grid with two-lane roads. It simulates 100 vehicles using the Krauss car-following model with a maximum speed of 13 m/s.

### 2. Network Simulation (NS-3)
The network simulation uses IEEE 802.11p for vehicle communication with 10 MHz channels and transmit power between 5-35 dBm. RSUs are deployed using K-means clustering.

### 3. Traffic Prediction Model (PyTorch Lightning)
A stacked LSTM model (4 layers × 128 neurons, dropout 0.2) predicts vehicle density based on historical traffic data. The model is trained on time-series data extracted from a Vehicle Information Table (VIT).

### 4. Relay Selection
An optimal-stopping algorithm is implemented to select relays when the SNR of a direct V2V link falls below a threshold, using the "look-then-leap" rule.

## Usage

### 1. Generate SUMO Traffic Scenarios

```bash
python scripts/gen_sumo_routes.py --map configs/urban.net.xml --flows 100 --seed 42 --duration 43200 -o data/raw/flows.rou.xml
```

### 2. Train the LSTM Predictor

```bash
python src/ml/train_lstm.py --data data/processed/vehicles_per_minute.csv --output models/traffic_lstm.pt
```

### 3. Run the Integrated Simulation

```bash
python src/main.py --sumo-trace data/raw/flows.rou.xml --model models/traffic_lstm.pt --tx-gain-min 5 --tx-gain-max 35
```

### 4. Analyze Results

```bash
python scripts/analyze_results.py --results-dir data/processed/ --output-dir results/
```

## Metrics and Analysis

The simulation collects the following metrics:
- Signal-to-Noise Ratio (SNR)
- Packet Delivery Ratio (PDR)
- End-to-end delay
- Throughput
- RSU power consumption

Results are exported as CSV files and can be visualized using the provided analysis scripts.

## Troubleshooting

### Common Issues

1. **NS-3 Python Binding Issues**
   - Ensure NS-3 was built with Python bindings enabled
   - Check that environment variables are correctly set

2. **SUMO TraCI Connection Errors**
   - Verify SUMO is installed properly
   - Check port availability (default: 8813)
   - Ensure SUMO_HOME is set correctly

3. **CUDA/PyTorch Errors**
   - If using GPU, ensure CUDA drivers match your PyTorch version
   - Fall back to CPU training by setting `--device cpu` in training script

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is based on the paper *"Cooperative Vehicular Networks: An Optimal and Machine-Learning Approach"* (Computers & Electrical Engineering, 2022). 