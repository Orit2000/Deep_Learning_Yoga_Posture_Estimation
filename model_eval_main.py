## Eval
model_pose = YogaClassifier(num_classes=num_classes, input_length=input_length).to(device)
model_pose.load_state_dict(torch.load(f="best.pth"))

cls_report, cls_cm = model_eval(model_pose, test_dataloader)

print(cls_report)

plt.figure(figsize=(10,10))
sns.heatmap(cls_cm, annot=True, fmt='g', cmap='Blues')
plt.show()