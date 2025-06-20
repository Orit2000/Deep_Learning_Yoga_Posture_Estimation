import torch
data = torch.load('resnet18_embeddings.pt')
embeddings = data['embeddings']  # shape [N, 512]
labels = data['labels']          # shape [N]
print("Embedding shape:", embeddings.shape)
print("First embedding vector:", embeddings[0])
print("First 5 labels:", labels[:5])