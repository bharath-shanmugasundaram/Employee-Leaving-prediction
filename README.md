# 🧠 Employee Attrition Prediction — Deep Neural Network (PyTorch)

This project builds a **Deep Learning model** using PyTorch to predict **employee attrition** (whether an employee will leave a company).  
The project includes:

- Full dataset preprocessing  
- Feature encoding & normalization  
- Deep Neural Network with multiple layers  
- Dropout regularization  
- L2 weight decay regularization  
- Training/Dev/Test split  
- Multiple experiment comparisons  
- Final accuracy evaluation  

This is an end-to-end HR analytics project demonstrating how to use structured data with neural networks.

---

## 📂 Dataset

The model uses the dataset:
HR-Employee-Attrition.csv


Target column:

| Column | Description |
|--------|-------------|
| `Attrition` | 1 = Employee Left, 0 = Employee Stayed |

Categorical columns (BusinessTravel, Department, Gender, JobRole, etc.) are converted into numerical encoded versions.

---

## 🧹 Data Preprocessing

### 1️⃣ Convert target variable to numerical
```python
df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
2️⃣ Encode categorical variables
Example:

df["BusinessTravel"] = df["BusinessTravel"].map({
    'Travel_Rarely': 1,
    'Travel_Frequently': 2,
    'Non-Travel': 0
})
Multiple columns are encoded similarly:

Department
EducationField
Gender
JobRole
MaritalStatus
Over18
OverTime
3️⃣ Split features + labels
Y = df["Attrition"].to_numpy(dtype=np.float32)
df = df.drop("Attrition", axis=1)
X = df.to_numpy(dtype=np.float32)
4️⃣ Standardization
scaler = StandardScaler()
X = scaler.fit_transform(X)
5️⃣ Train/Dev/Test Split
70% — Train  
15% — Dev  
15% — Test
X_train, X_test, Y_train, Y_test = train_test_split(...)
X_test, X_dev, Y_test, Y_dev = train_test_split(...)
🧠 Neural Network Architecture
A deep fully connected neural network:

Input → 64 → 32 → 16 → 8 → 4 → Output(1)
ReLU activation + Dropout(0.3)
class Neural(nn.Module):
    def __init__(self, input_size):
        super(Neural, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(4, 1)
        )
Loss & Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
🔁 Training Loop
1000 training epochs
Tracks training & dev accuracy every 10 epochs
y_pred = model(X_train)
loss = criterion(y_pred, Y_train)

loss.backward()
optimizer.step()
optimizer.zero_grad()
📊 Model Evaluation
On Test Set
with torch.no_grad():
    y_test_pred = model(X_test)
    y_test_class = (torch.sigmoid(y_test_pred) >= 0.5).float()
    test_acc = (y_test_class.eq(Y_test).sum() / Y_test.shape[0]) * 100
🧪 Experiments & Results
You ran the model with different regularization strategies.
1️⃣ No Regularization
Train Accuracy: 100.00%
Dev Accuracy:   84.62%
Test Accuracy:  83.64%
2️⃣ Dropout Regularization (0.3)
Train Accuracy: 96.99%
Dev Accuracy:   81.90%
Test Accuracy:  87.27%
📈 Best Test Accuracy: 87.27%
3️⃣ Dropout + L2 Regularization
Train Accuracy: 82.90%
Dev Accuracy:   86.88%
Test Accuracy:  85.45%
4️⃣ L2 Regularization Only
Train Accuracy: 99.61%
Dev Accuracy:   85.52%
Test Accuracy:  85.00%
📌 Best-performing model
⭐ Dropout Regularization (0.3)
Test Accuracy = **87.27%**
This model generalizes the best.
📈 Loss Curve Visualization
Loss plotted using seaborn:

sns.lineplot(cost)
🏁 Summary
This project demonstrates:
✔ End-to-end preprocessing of HR attrition dataset
✔ Encoding categorical features
✔ Standardization using StandardScaler
✔ Deep neural network built using PyTorch
✔ Comparison of multiple regularization techniques
✔ Proper Train/Dev/Test evaluation
✔ Achieving 87% accuracy on real employee attrition data
It’s a solid example of tabular deep learning using PyTorch.
⭐ If you like this project
Please give the repository a star ⭐ on GitHub!
Need help generating:

model.py
train.py
predict.py
inference notebook
Folder structure
Just tell me!

