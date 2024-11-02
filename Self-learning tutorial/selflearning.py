from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

from dateutil.relativedelta import relativedelta
from datetime import datetime
import pandas as pd

# prepare and save the data
patients = pd.read_csv('D:\homework\AI Healthcare\CSV\PATIENTS.csv')
adm = pd.read_csv('D:\homework\AI Healthcare\CSV\ADMISSIONS.csv')
diag = pd.read_csv('D:\homework\AI Healthcare\CSV\DIAGNOSES_ICD.csv')

merged = pd.merge(patients, adm, on='SUBJECT_ID', how='inner')
merged_df = pd.merge(merged, diag, on='HADM_ID', how='inner')


# filter the patients with code 584
akf = merged_df[merged_df['ICD9_CODE'].astype(str).str.startswith('584')]
#print(akf.info())
akf = akf.copy()

akf['ADMITTIME'] = pd.to_datetime(akf['ADMITTIME'], errors='coerce')
akf['DISCHTIME'] = pd.to_datetime(akf['DISCHTIME'], errors='coerce')

akf = akf.dropna(subset=['ADMITTIME', 'DISCHTIME'])
akf['LOS'] = (akf['DISCHTIME'] - akf['ADMITTIME']).dt.days

# remove the LOS<=1
akf = akf[akf['LOS'] > 1]


def calculate_age(DOB, ADMITTIME):
    def parse_date(date):
        if isinstance(date, pd.Timestamp):
            return date
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(date, fmt)
            except (ValueError, TypeError):
                continue
        return None

    dob_date = parse_date(DOB)
    admit_date = parse_date(ADMITTIME) if not pd.isna(ADMITTIME) else datetime.now()

    return relativedelta(admit_date, dob_date).years if dob_date and admit_date else None
akf["AGE"] = akf.apply(lambda row: calculate_age(row["DOB"], row["ADMITTIME"]), axis=1)
akf = akf[(akf['AGE'].between(20, 100)) & (~akf['AGE'].isna())]
#print(akf[['SUBJECT_ID_x', 'AGE', 'DOB', 'ADMITTIME']].head())
#print(akf['AGE'].describe())  # Summary statistics for age
akf['DEATHTIME'] = pd.to_datetime(akf['DEATHTIME'], errors='coerce')
akf['DIED_DURING_LOS'] = ((akf['DEATHTIME'] >= akf['ADMITTIME']) & (akf['DEATHTIME'] <= akf['DISCHTIME'])).astype(int)
akf['GENDER'] = akf['GENDER'].map({'M': 0, 'F': 1})
#print(akf[['SUBJECT_ID_x', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'DIED_DURING_LOS']].head())
# Now the data is ready


# For Log Reg

X = akf[['LOS', 'AGE', 'GENDER', 'SEQ_NUM', 'HADM_ID']]
y = akf['DIED_DURING_LOS']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


sgd_log_reg = SGDClassifier(
    loss='log', class_weight='balanced', max_iter=2000, tol=1e-3,
    random_state=42, eta0=1, learning_rate='constant'
)

# train the model
sgd_log_reg.fit(X_train, y_train)
y_train_pred = sgd_log_reg.predict(X_train)
y_test_pred = sgd_log_reg.predict(X_test)

print("Training Set Report For Logistic Regression:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred, zero_division=0))

print("\nTest Set Report For Logistic Regression:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))




# For RandomForest classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

rf_model = RandomForestClassifier(
    class_weight='balanced',
    n_estimators=1000,
    max_depth=None,
    random_state=42
)


rf_model.fit(X_train, y_train)
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Set Report For Random Forest:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred, zero_division=0))


print("\nTest Set Report For Random Forest:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))



# For xgboost

from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
xgb_model = XGBClassifier(
    scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
    n_estimators=3000,
    learning_rate=0.1,
    max_depth=None,
    eval_metric="error",
    random_state=42
)


eval_set = [(X_train, y_train), (X_test, y_test)]
xgb_model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
results = xgb_model.evals_result()
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

print("Training Set Report For XGBoost:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred, zero_division=0))

print("\nTest Set Report For XGBoost:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))


# plot error
plt.figure(figsize=(10, 6))
plt.plot(results["validation_0"]["error"], label="Training Error")
plt.plot(results["validation_1"]["error"], label="Testing Error")
plt.xlabel("Boosting Rounds")
plt.ylabel("Error Rate")
plt.title("Training and Testing Error over Boosting Rounds")
plt.legend()

plt.tight_layout()
plt.savefig("error-xgboost.svg", format="svg")
plt.show()

# err to accuracy
train_accuracy = [1 - err for err in results["validation_0"]["error"]]
test_accuracy = [1 - err for err in results["validation_1"]["error"]]

# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(train_accuracy, label="Training Accuracy")
plt.plot(test_accuracy, label="Testing Accuracy")
plt.xlabel("Boosting Rounds")
plt.ylabel("Accuracy")
plt.title("Training and Testing Accuracy over Boosting Rounds")
plt.legend()

plt.tight_layout()
plt.savefig("acc-xgboost.svg", format="svg")
plt.show()






# For Neural Network (Multi-Layer Perceptron or MLP)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 64),  # 2 hidden layers
    max_iter=100,
    warm_start=True,
    random_state=42,
    solver='adam',
    learning_rate_init=0.00025,
    alpha=0.000
)

epochs = 100
train_accuracies = []
test_accuracies = []
losses = []

for epoch in range(epochs):
    mlp.fit(X_train, y_train)
    losses.append(mlp.loss_)

    y_train_pred = mlp.predict(X_train)
    y_test_pred = mlp.predict(X_test)
    train_accuracies.append(accuracy_score(y_train, y_train_pred))
    test_accuracies.append(accuracy_score(y_test, y_test_pred))

print("Training Set Report For Multi-Layer Perceptron:")
print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred, zero_division=0))

print("\nTest Set Report For Multi-Layer Perceptron:")
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred, zero_division=0))

# plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, epochs + 1), test_accuracies, label='Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy over Epochs')
plt.legend()

plt.tight_layout
plt.savefig("acc-mlp.svg", format="svg")
plt.show()
# plot loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.tight_layout()
plt.savefig("loss-mlp.svg", format="svg")
plt.show()
y_train_pred = mlp.predict(X_train)
y_test_pred = mlp.predict(X_test)






