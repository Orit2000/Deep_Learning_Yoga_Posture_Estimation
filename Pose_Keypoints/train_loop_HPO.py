import json
import optuna
import pandas as pd
import torch
from torchmetrics import Accuracy, F1Score
#from tqdm.auto import tqdm
from tqdm import tqdm  # ⬅️ Replace tqdm.auto for lighter IO in some setups
from torchmetrics.classification import MulticlassAveragePrecision

# ────────────────────────────────────────────────────────────────
# helper steps (signature unchanged)
# ────────────────────────────────────────────────────────────────
def train_step_HPO(model, loader, optim, loss_fn, device, acc_m, f1_m, AP_m):
    model.train()
    t_loss = t_acc = t_f1 = t_AP = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        t_loss += loss.item()

        cls = torch.argmax(pred, dim=1)
        t_acc += acc_m(cls, y).item()
        t_f1  += f1_m(cls, y).item()
        t_AP += AP_m(pred, y).item()
        
        optim.zero_grad()
        loss.backward()
        optim.step()

    n = len(loader)
    return t_loss / n, t_acc / n, t_f1 / n , t_AP / n


@torch.inference_mode()
def test_step_HPO(model, loader, loss_fn, device, acc_m, f1_m, AP_m):
    model.eval()
    v_loss = v_acc = v_f1 = v_AP =  0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        v_loss += loss_fn(pred, y).item()

        cls = torch.argmax(pred, dim=1)
        v_acc += acc_m(cls, y).item()
        v_f1  += f1_m(cls, y).item()
        v_AP += AP_m(pred, y).item()
        
    n = len(loader)
    return v_loss / n, v_acc / n, v_f1 / n, v_AP / n


# ────────────────────────────────────────────────────────────────
# main training loop
# ────────────────────────────────────────────────────────────────
def train_loop_HPO(
        model, trainloader, testloader,
        optimizer, loss_fn,
        epochs, num_classes,
        verbose=True, trial=None, patience=5,
        save_history=False
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_m = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_m  = F1Score(task="multiclass", num_classes=num_classes).to(device)
    AP_m = MulticlassAveragePrecision(num_classes=num_classes).to(device)
    

    hist = {k: [] for k in ( 
        "train_loss", "train_accuracy", "train_f1", "train_AP",
        "test_loss",  "test_accuracy",  "test_f1", "test_AP"
    )}

    best_acc, best_ep = 0.0, -1
    best_state = model.state_dict()
    no_improve = 0

    for ep in tqdm(range(epochs)):
        tr_loss, tr_acc, tr_f1 , tr_AP= train_step_HPO(
            model, trainloader, optimizer, loss_fn, device, acc_m, f1_m, AP_m
        )
        va_loss, va_acc, va_f1, va_AP= test_step_HPO(
            model, testloader, loss_fn, device, acc_m, f1_m, AP_m
        )

        for k, v in zip(hist, (tr_loss, tr_acc, tr_f1, tr_AP,
                               va_loss, va_acc, va_f1, va_AP)):
            hist[k].append(v)

        # ― Optuna pruning ―
        if trial is not None:
            trial.report(va_acc, step=ep)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # ― early-stopping bookkeeping ―
        if va_acc > best_acc:
            best_acc, best_ep, no_improve = va_acc, ep, 0
            best_state = model.state_dict()
            torch.save(best_state, "best_model_weights_kp.pth") #Saves the model’s training parameters (weights) with the best validation accuracy.
            tag = "✅ improved"
        else:
            no_improve += 1
            tag = f"⚠️ no-improve {no_improve}/{patience}"

        if verbose:
            print(f"ep {ep:02d} | tr_loss {tr_loss:.3f} tr_acc {tr_acc:.3f} va_loss {va_loss:.3f} va_acc {va_acc:.3f} va_AP {va_AP:.3f} | {tag}")

        if no_improve >= patience:
            if verbose:
                print(f"⏹ early-stop @ep {ep}")
            break

    # restore best weights *once* after training
    model.load_state_dict(best_state)
    if verbose:
        print(f"🏁 best val-acc {best_acc:.4f} at ep {best_ep}")

    # # save history of the best model state
    # pd.DataFrame(hist).to_csv("training_history_kp.csv", index=False) # Saves all epoch-wise metrics as a CSV table- log of training and validation metrics for every epoch (not just the best one).


    # with open("training_history_kp.json", "w") as fp:
    #     json.dump(hist, fp)
    # return hist, best_ep

    model.load_state_dict(best_state)

    # ⬇️ Only save history if this is the best trial
    if save_history:
        pd.DataFrame(hist).to_csv("training_history_kp.csv", index=False)
        with open("training_history_kp.json", "w") as fp:
            json.dump(hist, fp)

    return hist, best_ep