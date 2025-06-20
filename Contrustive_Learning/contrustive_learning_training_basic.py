from Contrustive_Learning.contrustive_learning_classes import ContrastiveModel
import torch
import torch.nn.functional as F

def nt_xent_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    batch_size = z1.size(0)
    representations = torch.cat([z1, z2], dim=0)  # [2B, D]

    sim = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
    labels = torch.arange(batch_size).to(z1.device)
    labels = torch.cat([labels, labels], dim=0)

    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z1.device)
    sim = sim.masked_fill(mask, -1e9)
    sim = sim / temperature

    positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0)
    labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z1.device)

    loss = F.cross_entropy(sim, labels)
    return loss

model = ContrastiveModel(resnet_dim=512, keypoint_dim=68, embed_dim=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    for images, kp_xy in dataloader:
        images, kp_xy = images.to(device), kp_xy.to(device)

        # 1) get frozen‐backbone embeddings
        with torch.no_grad():
            feats_img = resnet_backbone(images)    # [B, 512, 1, 1] → flatten → [B,512]
            feats_kp  = yolo_keypoint_encoder(kp_xy)  # [B, keypoint_dim]

        # 2) project into shared space
        z_img, z_kp = model(feats_img, feats_kp)

        # 3) loss + update
        loss = nt_xent_loss(z_img, z_kp, temperature=0.5)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
