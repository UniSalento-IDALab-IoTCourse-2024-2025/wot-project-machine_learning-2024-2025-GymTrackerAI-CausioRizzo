import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il file CSV
df = pd.read_csv('dataset_index.csv', sep=';', low_memory=False)

# Separazione delle colonne delle feature e del target
X = df.drop(columns='Activity')
Y = df['Activity']

# Dividi il dataset
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=150)

# Addestramento Random Forest
random_forest = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=150)

# Leave One Out Cross Validation
loo = LeaveOneOut()
cv_scores = cross_val_score(random_forest, X, Y, cv=loo)
print(f'Leave-One-Out Cross Validation Accuracy Mean: {cv_scores.mean()}')
print(f'Leave-One-Out Cross Validation Accuracy Variance: {cv_scores.var()}')

print('Fit Random Forest')
random_forest.fit(X_train, y_train)

print('Predict')
y_pred = random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

report = classification_report(y_test, y_pred)
print(f'Classification Report:\n{report}')

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Converte il modello in formato ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]
onnx_model = convert_sklearn(random_forest, initial_types=initial_type)

# Salva il modello in formato ONNX
with open('random_forest.onnx', 'wb') as file:
    file.write(onnx_model.SerializeToString())

# Stampo la matrice di confusione per le 3 classi: Plank, JumpingJack, SquatJack
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Plank', 'JumpingJack', 'SquatJack'], yticklabels=['Plank', 'JumpingJack', 'SquatJack'])
plt.xlabel('Valori predetti')
plt.ylabel('Valori reali')
plt.title('Matrice di Confusione Random Forest')
plt.show()

# Leave-One-Out Cross Validation Accuracy Mean: 0.928763440860215
# Leave-One-Out Cross Validation Accuracy Variance: 0.06616191178170885
# Fit Random Forest
# Predict
# Accuracy: 0.9261744966442953
# Classification Report:
#               precision    recall  f1-score   support
#
#  JumpingJack       0.95      0.80      0.87        45
#        Plank       1.00      1.00      1.00        47
#    SquatJack       0.86      0.96      0.91        57
#
#     accuracy                           0.93       149
#    macro avg       0.94      0.92      0.93       149
# weighted avg       0.93      0.93      0.93       149
#
# Confusion Matrix:
# [[36  0  9]
#  [ 0 47  0]
#  [ 2  0 55]]