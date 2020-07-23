import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, requires_grad=False)
y = torch.randn(N, D_out, requires_grad=False)

w1 = torch.randn(D_in, H, requires_grad=True)
w2 = torch.randn(H, D_out, requires_grad=True)

learning_rate = 1e-3
adam = torch.optim.Adam([w1, w2], lr=learning_rate)

for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())

    adam.zero_grad()
    loss.backward()
    adam.step()
