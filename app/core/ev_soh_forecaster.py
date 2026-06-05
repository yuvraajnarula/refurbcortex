import os
import torch
import torch.nn as nn
import numpy as np
from app.utils.logger import app_logger

class EVSoHLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.relu(self.fc(out[:, -1, :]))

class EVSoHForecaster:
    def __init__(self, model_path="./models/ev_soh_lstm.pth"):
        self.model_path = model_path
        self.model = EVSoHLSTM()
        self._load_or_train()
        self.model.eval()
        app_logger.info("EV SoH LSTM Forecaster initialized")

    def _generate_synthetic_data(self, n=10000):
        ages = np.random.uniform(1, 10, n)
        kms = np.random.uniform(10000, 150000, n)
        fast_charge = np.random.uniform(0, 0.6, n)
        temp = np.random.uniform(20, 45, n)
        init_cap = np.full(n, 40.0)

        decay = (ages * 1.8) + (kms / 8000) + (fast_charge * 8) + (temp * 0.15)
        noise = np.random.normal(0, 1.5, n)
        soh = np.clip(100 - decay - noise, 40, 100)

        X = np.column_stack([ages, kms, fast_charge, temp, init_cap]).reshape(n, 1, 5)
        y = (soh / 100.0).reshape(-1, 1)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location="cpu", weights_only=True))
            return

        app_logger.info("Training EV SoH LSTM on synthetic degradation curves...")
        X, y = self._generate_synthetic_data()
        criterion, optimizer = nn.MSELoss(), torch.optim.Adam(self.model.parameters(), lr=0.01)

        for epoch in range(50):
            optimizer.zero_grad()
            loss = criterion(self.model(X), y)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0: app_logger.info(f"  Epoch {epoch} | Loss: {loss.item():.4f}")

        torch.save(self.model.state_dict(), self.model_path)
        app_logger.info("EV SoH LSTM saved")

    def predict(self, age_years: float, odometer_km: float, fast_charge_ratio: float = 0.3, avg_temp: float = 30.0) -> dict:
        x = torch.tensor([[age_years, odometer_km, fast_charge_ratio, avg_temp, 40.0]], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            soh_pct = max(40.0, min(100.0, self.model(x).item() * 100))

        if soh_pct >= 85: tier, margin_impact = "Excellent", 1.0
        elif soh_pct >= 70: tier, margin_impact = "Good", 0.9
        elif soh_pct >= 55: tier, margin_impact = "Fair", 0.7
        else: tier, margin_impact = "Poor/Critical", 0.4

        return {
            "predicted_soh_pct": round(soh_pct, 1),
            "health_tier": tier,
            "refurb_margin_impact": margin_impact,
            "warranty_risk": soh_pct < 70,
            "ev_action_flag": "BATTERY_DIAGNOSTIC" if soh_pct < 60 else "PROCEED"
        }