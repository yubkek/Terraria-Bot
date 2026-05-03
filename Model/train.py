import torch
from torch.utils.data import DataLoader
from dataset_loader import TerrariaDataLoader
from model import TerrariaBot

dataset = TerrariaDataLoader(
    "dataset_v1/data.jsonl",
    "dataset_v1/frames"
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = TerrariaBot(action_dim=22)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

bce = torch.nn.BCELoss()
mse = torch.nn.MSELoss()

for epoch in range(10):
    total_loss = 0

    for imgs, actions in loader:
        preds = model(imgs)

        keys_pred = preds[:, :20]
        mouse_pred = preds[:, :20]

        keys_true = actions[:, :20]
        mouse_true = actions[:, :20]

        loss_keys = bce(keys_pred, keys_true)
        loss_mouse = mse(mouse_pred, mouse_true)

        loss = loss_keys + loss_mouse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"epoch {epoch}: {total_loss:.4f}")

torch.save(model.state_dict(), "model.pth")
print("Saved")