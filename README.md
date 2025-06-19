# GymTrackerAI – Machine Learning

---

## Descrizione Generale

Questo componente contiene il modello di **classificazione esercizi** utilizzato dall'app GymTrackerAI. Il modello è addestrato per riconoscere 3 esercizi (Plank, JumpingJack, SquatJack) a partire dai dati dell'accelerometro raccolti tramite SensorTile.Box PRO.

---

## Architettura del Sistema

- **Linguaggio**: Python  
- **Librerie principali**: pandas, numpy, scikit-learn, onnx, flask  
- **Struttura**:
  - `dataset/`: file CSV per ogni classe  
  - `preprocessing.py`: normalizza e bilancia il dataset  
  - `random_forest.py`: addestra e converte il modello in ONNX  
  - `random_forest.onnx`: modello finale    
- **Servizio**: Docker container con Flask su porta 5000

---

## Repository Correlati

- Frontend: [github.com/tuo-user/gymtracker-frontend](https://github.com/tuo-user/gymtracker-frontend)
- Backend: [github.com/tuo-user/gymtracker-backend](https://github.com/tuo-user/gymtracker-backend)
- Machine Learning (questo repo): [github.com/tuo-user/gymtracker-ml](https://github.com/tuo-user/gymtracker-ml)
