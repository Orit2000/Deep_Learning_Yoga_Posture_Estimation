import json
import optuna
import pandas as pd
import torch
from torchmetrics import Accuracy, F1Score
from tqdm.auto import tqdm


# ────────────────────────────────────────────────────────────────
# helper steps (signature unchanged)
# ────────────────────────────────────────────────────────────────
def train_step(model, loader, optim, loss_fn, device, acc_m, f1_m):
    model.train()
    t_loss = t_acc = t_f1 = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        t_loss += loss.item()

        logits = torch.softmax(pred, 1)
        cls = torch.argmax(logits, 1)
        t_acc += acc_m(cls, y).item()
        t_f1  += f1_m(cls, y).item()

        optim.zero_grad()
        loss.backward()
        optim.step()

    n = len(loader)
    return t_loss / n, t_acc / n, t_f1 / n


@torch.inference_mode()
def test_step(model, loader, loss_fn, device, acc_m, f1_m):
    model.eval()
    v_loss = v_acc = v_f1 = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)

        pred = model(X)
        v_loss += loss_fn(pred, y).item()

        logits = torch.softmax(pred, 1)
        cls = torch.argmax(logits, 1)
        v_acc += acc_m(cls, y).item()
        v_f1  += f1_m(cls, y).item()

    n = len(loader)
    return v_loss / n, v_acc / n, v_f1 / n


# ────────────────────────────────────────────────────────────────
# main training loop
# ────────────────────────────────────────────────────────────────
def train_loop_HPO(
        model, trainloader, testloader,
        optimizer, loss_fn,
        epochs, num_classes,
        verbose=True, trial=None, patience=5
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_m = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_m  = F1Score(task="multiclass", num_classes=num_classes).to(device)

    hist = {k: [] for k in (
        "train_loss", "train_accuracy", "train_f1",
        "test_loss",  "test_accuracy",  "test_f1"
    )}

    best_acc, best_ep = 0.0, -1
    best_state = model.state_dict()
    no_improve = 0

    for ep in tqdm(range(epochs)):
        tr_loss, tr_acc, tr_f1 = train_step(
            model, trainloader, optimizer, loss_fn, device, acc_m, f1_m
        )
        va_loss, va_acc, va_f1 = test_step(
            model, testloader, loss_fn, device, acc_m, f1_m
        )

        for k, v in zip(hist, (tr_loss, tr_acc, tr_f1,
                               va_loss, va_acc, va_f1)):
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
            torch.save(best_state, "best.pth")
            tag = "✅ improved"
        else:
            no_improve += 1
            tag = f"⚠️ no-improve {no_improve}/{patience}"

        if verbose:
            print(f"ep {ep:02d} | tr_acc {tr_acc:.3f} va_acc {va_acc:.3f} | {tag}")

        if no_improve >= patience:
            if verbose:
                print(f"⏹ early-stop @ep {ep}")
            break

    # restore best weights *once* after training
    model.load_state_dict(best_state)
    if verbose:
        print(f"🏁 best val-acc {best_acc:.4f} at ep {best_ep}")

    # save history
    pd.DataFrame(hist).to_csv("training_history.csv", index=False)
    with open("training_history.json", "w") as fp:
        json.dump(hist, fp)

    return hist, best_ep