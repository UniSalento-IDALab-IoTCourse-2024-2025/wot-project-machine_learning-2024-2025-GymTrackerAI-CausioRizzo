# GymTrackerAI – Machine Learning

## Descrizione Generale

Questo componente contiene il modello di **classificazione esercizi** utilizzato dall'app GymTrackerAI. Il modello è addestrato per riconoscere 3 esercizi (Plank, JumpingJack, SquatJack) a partire dai dati dell'accelerometro raccolti tramite SensorTile.Box PRO.

## Architettura del Sistema

- **Linguaggio**: Python  
- **Librerie principali**: pandas, numpy, scikit-learn, onnx, flask  
- **Struttura**:
  - `dataset/`: file CSV per ogni classe  
  - `preprocessing.py`: normalizza e bilancia il dataset  
  - `random_forest.py`: addestra e converte il modello in ONNX  
  - `random_forest.onnx`: modello finale    
- **Servizio**: Docker container con Flask su porta 5000

## Repository Correlati

- Frontend:
  (https://github.com/UniSalento-IDALab-IoTCourse-2024-2025/wot-project-frontend-2024-2025-GymTrackerAI-CausioRizzo)
- Backend:
  (https://github.com/UniSalento-IDALab-IoTCourse-2024-2025/wot-project-backend-2024-2025-GymTrackerAI-CausioRizzo)
- Machine Learning (questo repo):
  (https://github.com/UniSalento-IDALab-IoTCourse-2024-2025/wot-project-machine_learning-2024-2025-GymTrackerAI-CausioRizzo)
