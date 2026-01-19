import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("carbon_emission.csv")

# Independent and Dependent variables
X = data[['Year']]
y = data['CO2_Emission']

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predict future emissions
future_years = pd.DataFrame({'Year': [2023, 2024, 2025, 2026, 2027]})
predictions = model.predict(future_years)

# Print predictions
print("Future Carbon Emission Predictions:")
for year, pred in zip(future_years['Year'], predictions):
    print(f"{year}: {pred:.2f}")

# Plot graph
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.xlabel("Year")
plt.ylabel("CO2 Emission")
plt.title("Carbon Emission Prediction")
plt.show()
