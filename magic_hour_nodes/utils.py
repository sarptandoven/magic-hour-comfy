def latent_delta_mse(a,b):
    return ((a-b)**2).mean().item() if hasattr(a,'mean') else 0.0
