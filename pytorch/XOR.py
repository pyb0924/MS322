import torch

NUM_INPUT = 2
NUM_HIDDEN = 2
NUM_OUTPUT = 1

model = torch.nn.Sequential(
    torch.nn.Linear(NUM_INPUT, NUM_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUM_HIDDEN, NUM_OUTPUT)
)
x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)

y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)

loss_function = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)


for i in range(50):
    for j in range(4):
        y_pred = model(x[j])
        loss = loss_function(y_pred, y[j])
        print(loss.item())

        loss.backward()
        optimizer.zero_grad()
        optimizer.step()
