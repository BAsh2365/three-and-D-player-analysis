import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Advanced.csv')

df = df[(df['season'] >= 1980)].copy()

df = df[df['pos'].isin(['SF', 'PF', 'SG'])].copy()

print(df.head())

print(df.shape)

True_Shooting_Percantage = [col for col in df.columns if 'ts_percent' in col]
df[True_Shooting_Percantage] = df[True_Shooting_Percantage] * 100

#rename 'per' to 'Points_per_game'
df.rename(columns={'per': 'Points_per_game'}, inplace=True)

#rename 'gs' to 'games_started'
df.rename(columns={'gs': 'games_started'}, inplace=True)

#rename f_tr to free throw rate
df.rename(columns={'f_tr': 'free_throw_rate'}, inplace=True)

#rename x3p_ar to three point attempt rate
df.rename(columns={'x3p_ar': 'three_point_attempt_rate'}, inplace=True)



print(df.dtypes)

Features = [
    'Points_per_game','three_point_attempt_rate', 'ts_percent', 'blk_percent',
    'stl_percent', 'usg_percent', 'free_throw_rate', 'trb_percent', 'ast_percent', 'games_started'
]

y = df['ws'] #do three and D players have higher win shares?

X = df[Features]

X = X.fillna(0)
y = y.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

RF_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
RF_model.fit(X_train_scaled, y_train)


y_pred_rf = RF_model.predict(X_test_scaled)

print("Random Forest Results:")
print(f"MSE: {mean_squared_error(y_test, y_pred_rf):.3f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.3f}")
print(f"R2: {r2_score(y_test, y_pred_rf):.3f}")


fig, axes = plt.subplots(1, figsize=(14, 5))

axes.scatter(y_test, y_pred_rf, alpha=0.5)
axes.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
axes.set_xlabel('Actual')
axes.set_ylabel('Predicted')
axes.set_title(f'Random Forest (R2={r2_score(y_test, y_pred_rf):.3f})')

plt.tight_layout()
plt.show()

feature_importance_rf = pd.DataFrame({
    'Feature': Features,
    'Importance': RF_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nRandom Forest Feature Importance:")
print(feature_importance_rf)


fig, axes = plt.subplots(1, figsize=(14, 5))

axes.barh(feature_importance_rf['Feature'], feature_importance_rf['Importance'])
axes.set_xlabel('Importance')
axes.set_ylabel('Feature')
axes.set_title('Random Forest Feature Importance')
axes.invert_yaxis()

plt.tight_layout()
plt.show()