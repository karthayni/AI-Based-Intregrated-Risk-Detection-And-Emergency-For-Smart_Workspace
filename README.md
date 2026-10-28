# AI-Based Integrated Safety And Risk System For Smart Workspace


Team: B. Karthayayini (team leader), G. Vishnu Priya, Likitha S, Spoorthi S Tellur. Mentor: Indranil Bakshi.

One system that watches a workplace through a camera, follows employee buses on a map, and shows every alert in one website.

## Modules

| Module | What it does | Folder |
|---|---|---|
| 1. PPE monitoring | YOLOv8 finds people and checks helmet, vest, boots and gloves. A zone drawn on the camera view marks where PPE is mandatory. | `ppe_module` |
| 2. Stress detection | A Teachable Machine model labels each face as Stressed or Normal. It runs in the same camera view as Module 1. | `ppe_module` |
| 3. Employee transport | GPS positions are compared with the planned route and stops. Alerts: route deviation, unexpected stop, signal lost, stop missed. | `transport_module` |
| Website and dashboard | Home, Dashboard (overview, live camera, workplace alerts, transport) and About pages. | `transport_module` |

Status of each person: SAFE, CAUTION (complete PPE but stress high), VIOLATION (PPE missing), HIGH RISK (PPE missing inside the danger zone, or PPE missing with stress high).

## Folder layout

```
ppe_module/        PPE and stress detection (ppe_detect.py, stress.py, training and evaluation scripts)
transport_module/  website, dashboard, transport tracker and GPS simulator
```

## How to run

You need Python 3 and a webcam. Use one virtual environment for each folder.

### 1. Detection (Modules 1 and 2)

```
cd ppe_module
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-stress.txt
python ppe_detect.py --weights best.pt --source 0 --stress-model model_unquant.tflite --stress-labels labels.txt
```

Draw the danger zone once with `python ppe_detect.py --source 0 --draw-zone`. Press `q` to quit, `s` for a screenshot, `f` for full screen.

### 2. Website with the dashboard (Module 3 and everything together)

```
cd transport_module
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py --ppe-dir "path\to\ppe_module"
```

Open http://127.0.0.1:5000. On **Dashboard, Live camera** press Start detection. In a second terminal run `python simulator.py` to move the demo buses.

The website starts the detection with the Python of the `venv` inside `ppe_module`, so install that folder's requirements first.

## Training your own models

- PPE model: `ppe_module/train.py` with a PPE dataset (see `data.yaml`).
- Stress model: collect or prepare pictures with `collect_faces.py` or `prepare_fer.py`, train in Teachable Machine (classes `Stressed` and `Normal`), export TensorFlow Lite (floating point), then measure it with `evaluate_stress.py`.

Datasets and pictures of people are not stored in this repository.

## Limits

- The stress label comes from facial expression only. It is a risk indicator, not a medical measurement.
- The stress model was trained on a public expression dataset, so it is less accurate on a live webcam than on its own test pictures.
- Buses are simulated unless a phone or GPS device sends real positions.
- Gloves are detected less reliably than helmets and vests.
