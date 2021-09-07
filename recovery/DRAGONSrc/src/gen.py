import torch
from copy import deepcopy

def scale(data):
    return torch.max(torch.tensor(0), torch.min(data, torch.tensor(1)))

def gen(model, init_vec, schedule, real_label, loss, epsilon=1e-4, num_examples=1):
    lr = 0.01
    iteration = 0; equal = 0
    init = deepcopy(init_vec)
    init.requires_grad = True
    copyz = 10; optimizer = torch.optim.Adam([init] , lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    for _ in range(100):
        copy = deepcopy(init.data)
        res = model(init, schedule)
        z = loss(res, real_label)
        optimizer.zero_grad(); z.backward(); optimizer.step(); scheduler.step()
        init.data = scale(init.data)
        equal = equal + 1 if torch.all(abs(copy - init.data) < epsilon) or (0 < copyz - z.item() < epsilon) else 0
        if equal > 10: break
        copyz = z.item()
        iteration += 1
    init.requires_grad = False
    return init


