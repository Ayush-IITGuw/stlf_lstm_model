import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config import CONFIG
from model.PowerModel import PowerEncoder,PowerDecoder
from preprocess.timestamp import TimestampBuilder
from preprocess.features import FeatureMaker
from preprocess.scalers import scale_data
from preprocess.window import make_windows
from model.metrics import model_report


def to_tensor(arr):
    return arr if torch.is_tensor(arr) else torch.tensor(arr, dtype=torch.float32)
 
 
def make_loader(x, y, batch_size, shuffle):
    ds = TensorDataset(to_tensor(x), to_tensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def run_epoch(encoder, decoder, loader, horizon_steps, num_output_features,loss_fn, optimizer=None, teacher_forcing=True, device="cpu"):
    """One pass over `loader`. If optimizer is given, trains; otherwise evaluates."""
    is_train = optimizer is not None
    encoder.train(is_train)
    decoder.train(is_train)
 
    total_loss, n_batches = 0.0,0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
         
        with torch.set_grad_enabled(is_train):
            encoder_outputs, encoder_hidden = encoder(x_batch)
            start_value = x_batch[:, -1, :num_output_features]
            
            preds, _, _ = decoder(
                encoder_outputs,
                encoder_hidden,
                horizon_steps=horizon_steps,
                target_tensor=y_batch if teacher_forcing else None,
                start_value=start_value,
            )
            
            preds = preds.squeeze(-1)
            loss = loss_fn(preds,y_batch)
            
        if(is_train):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                
        total_loss += loss.item()
        n_batches+=1
        
    return total_loss/n_batches
    
@torch.no_grad()
def predict(encoder, decoder, loader, horizon_steps, num_output_features, device="cpu"):
    encoder.eval()
    decoder.eval()
 
    all_preds = []
    for x_batch, _ in loader:
        x_batch = x_batch.to(device)
        encoder_outputs, encoder_hidden = encoder(x_batch)
        start_value = x_batch[:, -1, :num_output_features]
        preds, _, _ = decoder(
            encoder_outputs, encoder_hidden,
            horizon_steps=horizon_steps,
            target_tensor=None,
            start_value=start_value,
        )
        all_preds.append(preds.cpu().numpy())
 
    return np.concatenate(all_preds, axis=0)


def main():
    cfg = CONFIG
    os.makedirs(cfg["artifacts_dir"], exist_ok=True)
    
    df = pd.read_excel(cfg["excel_path"], sheet_name=cfg["sheet_name"],na_values=["NA", "NaN", "", "null", "?"])
 
    # 2. Preprocessing
    pre_pipe = Pipeline(steps=[
        ("timestamps", TimestampBuilder(tz_local=cfg["tz_local"])),
        ("features", FeatureMaker(target_col=cfg["target_col"], drop_cols=cfg["drop_cols"])),
    ])
    df_prep = pre_pipe.fit_transform(df)
 
    # 3. Train/test split
    cutoff = pd.Timestamp(cfg["train_cutoff_utc"])
    train = df_prep.loc[:cutoff].copy()
    test = df_prep.loc[cutoff + pd.Timedelta(seconds=1):].copy()
 
    feature_cols = cfg["feature_cols"]
    ycol = cfg["target_col"]
 
    # 4. Scale data 
    X_scaler = scale_data(cfg["x_scaler"]).fit(train[feature_cols])
    y_scaler = scale_data(cfg["y_scaler"]).fit(train[[ycol]])
 
    X_train = X_scaler.transform(train[feature_cols]).astype("float32")
    y_train = y_scaler.transform(train[[ycol]]).astype("float32").ravel()
    X_test = X_scaler.transform(test[feature_cols]).astype("float32")
    y_test = y_scaler.transform(test[[ycol]]).astype("float32").ravel()
 
    # 5. Make windows
    HISTORY, HORIZON = cfg["history"], cfg["horizon"]
    X_tr_w, y_tr_w = make_windows(X_train, y_train, HISTORY, HORIZON)
    X_te_w, y_te_w = make_windows(X_test, y_test, HISTORY, HORIZON)
 
    num_features = X_tr_w.shape[-1]
    num_output_features = 1 
    
    dropout = cfg["dropout"]
    hidden_size = cfg["hidden_size"]
    num_layers = cfg.get("num_layers", 2)
    batch_size = cfg["batch_size"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    device = cfg.get("device", "cpu")
 
    train_loader = make_loader(X_tr_w, y_tr_w, batch_size, shuffle=True)
    val_loader = make_loader(X_te_w, y_te_w, batch_size, shuffle=False)
 
    encoder = PowerEncoder(num_features, hidden_size, num_layers=num_layers, dropout=dropout).to(device)
    decoder = PowerDecoder(num_output_features, hidden_size, num_layers=num_layers, dropout=dropout).to(device)
 
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=lr
    )
    loss_fn = nn.MSELoss()
 
    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            encoder, decoder, train_loader, HORIZON, num_output_features,
            loss_fn, optimizer=optimizer, teacher_forcing=True, device=device,
        )
        val_loss = run_epoch(
            encoder, decoder, val_loader, HORIZON, num_output_features,
            loss_fn, optimizer=None, teacher_forcing=False, device=device,
        )
        print(f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f}")
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()},
                os.path.join(cfg["artifacts_dir"], "best_power_seq2seq.pt"),
            )
 
    # Load best checkpoint before evaluating / explaining
    best = torch.load(os.path.join(cfg["artifacts_dir"], "best_power_seq2seq.pt"), map_location=device)
    encoder.load_state_dict(best["encoder"])
    decoder.load_state_dict(best["decoder"])
 
    # 7. Evaluate 
    y_pred_scaled = predict(encoder, decoder, val_loader, HORIZON, num_output_features, device=device)
 
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1, HORIZON)
    y_true = y_scaler.inverse_transform(y_te_w.reshape(-1, 1)).reshape(-1, HORIZON)
 
    report = model_report(y_true, y_pred)
    print(report)
    
    n = 200

    plt.figure(figsize=(15, 5))
    plt.plot(y_true[:, 0][:n], label="Actual", linewidth=2)
    plt.plot(y_pred[:, 0][:n], label="Predicted", linewidth=2)

    plt.title("Power Forecast: Actual vs Predicted")
    plt.xlabel("Time step")
    plt.ylabel(ycol)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
 
    # 8. Preserve trained artifacts before optional explainability work
    joblib.dump(pre_pipe, cfg["pipeline_path"])
    joblib.dump(X_scaler, cfg["x_scaler_path"])
    joblib.dump(y_scaler, cfg["y_scaler_path"])
 
                
if __name__ == "__main__":
    main()