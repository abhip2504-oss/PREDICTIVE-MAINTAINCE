---
description: How to run the Predictive Maintenance Local Simulation
---

1. Make sure Docker Desktop is started and running.

2. Start the LocalStack (mock AWS S3) container:
// turbo
```bash
docker-compose up -d
```

3. Ensure dependencies are installed:
// turbo
```bash
pip install -r requirements.txt
```

4. Initialize the environment (create buckets and upload historical data):
// turbo
```bash
python init_env.py
```

5. Train the Random Forest model:
// turbo
```bash
python train_model.py
```

6. Start the IoT sensor data stream (Run in a separate terminal if manually running):
// turbo
```bash
python sensor_stream.py
```

7. Launch the Streamlit dashboard (Run in a separate terminal if manually running):
// turbo
```bash
streamlit run dashboard.py
```
