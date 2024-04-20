import torch

x = torch.tensor([2.0])
y = torch.tensor([10.0])
a = torch.tensor([1.0], requires_grad = True)

loss = y - (a*x)
loss.backward()
print("grad:",a.grad)

for _ in range(100):
    a.grad.zero_()
    loss = y - (a*x)
    loss.backward()
    with torch.no_grad():
        a -= a.grad * 0.1 * loss 

print("a:",a)
print("result:",(a*x))
print("loss:",(y - (a * x)))
