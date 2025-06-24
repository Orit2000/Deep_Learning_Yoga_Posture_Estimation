import pandas as pd
import torch
from tqdm.auto import tqdm
import json
from torchmetrics import Accuracy, F1Score

def train_step(model, dataloader, optimizer, loss_fn,device,accuracy_score,f1_score):
    model.train()
    train_loss, train_acc, train_f1 = 0, 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        #loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        #accuracy
        logits = torch.softmax(y_pred, dim=1)
        class_prediction = torch.argmax(logits, dim=1)
        acc = accuracy_score(class_prediction, y)
        f1 = f1_score(class_prediction, y)
        train_acc += acc.item()
        train_f1 += f1.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    train_f1 /= len(dataloader)

    return train_loss, train_acc, train_f1

def test_step(model, dataloader, loss_fn,device,accuracy_score,f1_score):
    model.eval()
    test_loss, test_acc, test_f1 = 0, 0, 0
    with torch.inference_mode():
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            #loss
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            #accuracy
            logits = torch.softmax(y_pred, dim=1)
            class_prediction = torch.argmax(logits, dim=1)
            acc = accuracy_score(class_prediction, y)
            f1 = f1_score(class_prediction, y)
            test_acc += acc.item()
            test_f1 += f1.item()

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    test_f1 /= len(dataloader)
    return test_loss, test_acc, test_f1

def train_loop(model, trainloader, testloader, optimizer, loss_fn, epochs, num_classes,verbose=True, trial=None, patience=5): #patience for early stopping
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    accuracy_score = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    f1_score = F1Score(task="multiclass", num_classes=num_classes).to(device)
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_f1": [],
    }

    no_improve_epochs = 0
    best_acc = 0.0;  
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc, train_f1 = train_step(
            model=model,
            dataloader=trainloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            accuracy_score=accuracy_score,
            f1_score=f1_score
        )

        test_loss, test_acc, test_f1 = test_step(
            model=model,
            dataloader=testloader,
            loss_fn=loss_fn,
            device=device,
            accuracy_score=accuracy_score,
            f1_score=f1_score
        )

        
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_acc)
        history['test_f1'].append(test_f1)
          
        # Early-stopping bookkeeping
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            best_weights = model.state_dict()
            no_improve_epochs = 0
            torch.save(best_weights, 'best1.pth')
            status = f"✅ Accuracy improved, weights saved (epoch {epoch})"
        else:
            no_improve_epochs += 1
            status = f"⚠️ Accuracy not improved (epoch {epoch}) — Patience: {no_improve_epochs}/{patience}"

        if verbose:
            print(f"Epoch {epoch}")
            print(f"train loss: {train_loss:.4f} | test loss: {test_loss:.4f}")
            print(f"train accuracy: {train_acc:.4f} | test accuracy: {test_acc:.4f}")
            print(f"train f1: {train_f1:.4f} | test f1: {test_f1:.4f}")
            print(status)
            print("-------------------------------------------------")

        if no_improve_epochs >= patience:
            print(f"⏹️ Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

        model.load_state_dict(best_weights)
        print(f"🏁 Best accuracy: {best_acc:.4f} (epoch {best_epoch})")
        #return history, best_epoch

    model.load_state_dict(best_weights)
    if verbose:
        print(f"🏁 Best val accuracy {best_acc:.4f} at epoch {best_epoch}")

    # Save history
    pd.DataFrame(history).to_csv("training_history1.csv", index=False)
    with open("training_history1.json", "w") as f:
        json.dump(history, f)
    print("You are in train_loop.py")
    return history, best_epoch
    
    #     best = max(history['test_accuracy'])
    #     best_epoch = history['test_accuracy'].index(best) 
   
    #     if test_acc < best:
    #         status = f"Accuracy not improved from epoch {best_epoch}"
    #     else: 
    #         status = f"Accuracy improved, saving weight....."
    #         torch.save(model.state_dict(), 'best.pth')

    #     if verbose:
    #         print(f"Epoch {epoch}")
    #         print(f"train loss: {train_loss} | test loss: {test_loss}")
    #         print(f"train accuracy: {train_acc} | test accuracy: {test_acc}")
    #         print(f"train f1: {train_f1} | test f1: {test_f1}")
    #         print(status)
    #         print("-------------------------------------------------")

    # print(f"Best accuracy on epoch: {best_epoch}, accuracy: {best}")
    # return history, best_epoch