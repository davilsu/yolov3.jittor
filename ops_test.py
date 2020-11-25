loss = 
log_weight = (pos_weight - 1).mul(target).add_(1);
loss = (1 - target).mul_(input).add_(log_weight.mul_(((-max_val).exp_().add_((-input - max_val).exp_())).log_().add_(max_val)));