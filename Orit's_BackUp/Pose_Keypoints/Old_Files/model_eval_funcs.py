# Model Eval
def model_eval(model, dataloader):
    y_preds = []
    y_true = []
    model.eval()
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
            y_preds.append(y_pred)
            y_true.append(y)

    y_pred = torch.cat(y_preds).cpu().numpy()
    y_test = torch.cat(y_true).cpu().numpy()

    report = classification_report(y_pred, y_test)
    cm = confusion_matrix(y_pred, y_test)

    return report, cm

def make_prediction(image, true_label):
    results = model_yolo.predict(image, boxes=False, verbose=False)
    for r in results:
        im_array = r.plot(boxes=False)  # plot a BGR numpy array of predictions
        keypoints = r.keypoints.xyn.cpu().numpy()[0]
        keypoints = keypoints.reshape((1, keypoints.shape[0]*keypoints.shape[1]))[0].tolist()

    #Prediction
    model_pose.cpu()
    model_pose.eval()
    with torch.inference_mode():
        logit = model_pose(torch.tensor(keypoints[2:]))
        pred = torch.softmax(logit, dim=0).argmax(dim=0).item()
        prediction = classes_dict[pred]
    
    # if prediciton is correct, title color will be green.
    if prediction == true_label:
        color = 'green'
    else:
        color = 'red'
    
    plt.imshow(im_array[..., ::-1])
    plt.title(f"prediction:{prediction}\ntrue label:{true_label}", color=color)
    plt.show()