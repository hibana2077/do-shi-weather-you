import torch
import torch.nn as nn
import torch.nn.functional as F


def bounded_regression_output(x, limit=5.0):
    return limit * torch.tanh(x / limit)


class LTSFLinear(nn.Module):
    """
    Just one Linear layer
    """
    def __init__(self, configs):
        super(LTSFLinear, self).__init__()
        self.seq_len = configs['seq_len']
        self.pred_len = configs['pred_len']
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = configs['enc_in']
        self.individual = configs['individual']
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            nn.init.constant_(self.Linear.weight, 1.0 / self.seq_len)
            nn.init.zeros_(self.Linear.bias)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        return x # [Batch, Output length, Channel]


class LSTMForecaster(nn.Module):
    """LSTM baseline for one-step temperature forecasting."""

    def __init__(self, input_size, hidden_size=64, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        return bounded_regression_output(self.head(x[:, -1]))


class LTSFForecaster(nn.Module):
    """LTSF-Linear baseline followed by a channel projection head."""

    def __init__(self, seq_len, pred_len, enc_in):
        super().__init__()
        self.ltsf = LTSFLinear(
            configs={
                "seq_len": seq_len,
                "pred_len": pred_len,
                "enc_in": enc_in,
                "individual": False,
            }
        )
        self.head = nn.Linear(enc_in, 1)

    def forward(self, x):
        x = self.ltsf(x).squeeze(1)
        x = F.gelu(x)
        return bounded_regression_output(self.head(x))


class WeatherFusionForecaster(nn.Module):
    """
    Hybrid LTSF/LSTM forecaster.

    The disagreement feature is the absolute difference between the linear and
    recurrent predictions. Disabling it gives an ablation of the novelty claim.
    """

    def __init__(self, seq_len, pred_len, enc_in, hidden_size=64, use_disagreement=True):
        super().__init__()
        self.use_disagreement = use_disagreement
        self.ltsf = LTSFLinear(
            configs={
                "seq_len": seq_len,
                "pred_len": pred_len,
                "enc_in": enc_in,
                "individual": False,
            }
        )
        self.ltsf_head = nn.Linear(enc_in, 1)
        self.lstm = nn.LSTM(enc_in, hidden_size, batch_first=True)
        self.lstm_head = nn.Linear(hidden_size, 1)
        fusion_features = 3 if use_disagreement else 2
        self.fusion_head = nn.Linear(fusion_features, 1)

    def forward(self, x):
        ltsf_pred = self.ltsf(x).squeeze(1)
        ltsf_pred = self.ltsf_head(F.gelu(ltsf_pred))

        lstm_state, _ = self.lstm(x)
        lstm_pred = self.lstm_head(lstm_state[:, -1])

        if self.use_disagreement:
            features = [ltsf_pred, lstm_pred, (ltsf_pred - lstm_pred).abs()]
        else:
            features = [ltsf_pred, lstm_pred]
        return bounded_regression_output(self.fusion_head(torch.cat(features, dim=1)))


# Backward-compatible names used by the original notebook/results.
WeatherModel = LSTMForecaster
WeatherModel_with_LTSF = LTSFForecaster
WeatherModel_Mix = WeatherFusionForecaster
