# 重新复习

def optim_warmup(step, optim, lr, warmup_iters):
    '''
        optim_warmup 函数用于实现学习率预热（Learning Rate Warmup）。在训练的初始阶段，学习率会从零逐步增加到预设的学习率值，从而避免训练一开始就使用过高的学习率，导致模型参数发生剧烈变化。
        
        warmup_iters: 热过程的迭代次数
    '''
    lr = lr * float(step) / warmup_iters
    for param_group in optim.param_groups:
        param_group["lr"] = lr


def update_ema(model, ema_model, ema_rate):
    '''
        p2 = ema_rate * p2 + (1 - ema_rate) * p1
    '''
    for p1, p2 in zip(model.parameters(), ema_model.parameters()):
        # Beta * previous ema weights + (1 - Beta) * current non ema weight
        p2.data.mul_(ema_rate)
        p2.data.add_(p1.data * (1 - ema_rate))
