import torch


def lat_weighted_mse(pred, y, vars, lat, weighted=False, weight_dict=None, postfix=""):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, N, V, H, W]
        pred: [B, N, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, N, V, H, W]
    dtype, device = error.dtype, error.device

    # lattitude weights
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat[None, None, :, None].to(dtype=dtype, device=device)  # (1, 1, H, 1)

    loss_dict = {}
    for i, var in enumerate(vars):
        loss_dict[f"w_mse_{var}{postfix}"] = (error[:, :, i] * w_lat).mean()
        
    if weighted:
        weights = torch.Tensor([weight_dict[var] for var in vars]).to(device=device).view(1, -1, 1, 1)
        weights = weights / weights.sum()
    else:
        weights = torch.ones(len(vars)).to(device=device).view(1, -1, 1, 1) / len(vars)
    
    loss_dict[f"w_mse_agg{postfix}"] = (error * w_lat.unsqueeze(1) * weights).sum(dim=1).mean()

    return loss_dict


def lat_weighted_rmse(pred, y, vars, lat, chosen_vars=None, postfix=""):
    """Latitude weighted root mean squared error

    Args:
        y: [B, N, V, H, W]
        pred: [B, N, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, N, V, H, W]
    dtype, device = error.dtype, error.device

    # lattitude weights
    w_lat = torch.cos(torch.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = w_lat[None, None, :, None].to(dtype=dtype, device=device)  # (1, 1, H, 1)
    chosen_vars = chosen_vars or vars

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if var in chosen_vars:
                loss_dict[f"w_rmse_{var}{postfix}"] = torch.mean(
                    torch.sqrt(torch.mean(error[:, :, i] * w_lat, dim=(-2, -1)))
                )

    return loss_dict

# def lat_weighted_crps(pred: torch.Tensor, y: torch.Tensor, vars, lat, chosen_vars=None, postfix=""):
#     assert len(pred.shape) == len(y.shape) + 1
#     # pred: [B, N, V, H, W] because there are N ensemble members
#     # y: [B, V, H, W]
    
#     H, N = pred.shape[-2], pred.shape[1]
    
#     # lattitude weights
#     w_lat = np.cos(np.deg2rad(lat))
#     w_lat = w_lat / w_lat.mean()
#     w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device) # (H, )    
    
#     def crps_var(pred_var: torch.Tensor, y_var: torch.Tensor):
#         # pred_var: [B, N, H, W]
#         # y: [B, H, W]
#         # first term: prediction errors
#         with torch.no_grad():
#             error_term = torch.abs(pred_var - y_var.unsqueeze(1)) # [B, N, H, W]
#             error_term = error_term * w_lat.view(1, 1, H, 1) # [B, N, H, W]
#             error_term = torch.mean(error_term)
        
#         # second term: ensemble spread
#         with torch.no_grad():
#             spread_term = torch.abs(pred_var.unsqueeze(2) - pred_var.unsqueeze(1)) # [B, N, N, H, W]
#             spread_term = spread_term * w_lat.view(1, 1, 1, H, 1) # [B, N, N, H, W]
#             spread_term = spread_term.mean(dim=(-2, -1)) # [B, N, N]
#             spread_term = spread_term.sum(dim=(1, 2)) / (2 * N * (N - 1)) # [B]
#             spread_term = spread_term.mean()
            
#         return error_term - spread_term
    
#     chosen_vars = chosen_vars or vars
#     loss_dict = {}
#     for i, var in enumerate(vars):
#         if var in chosen_vars:
#             loss_dict[f"w_crps_{var}{postfix}"] = crps_var(pred[:, :, i], y[:, i])
        
#     return loss_dict

# def lat_weighted_spread_skill_ratio(pred: torch.Tensor, y: torch.Tensor, vars, lat, chosen_vars=None, postfix=""):
#     assert len(pred.shape) == len(y.shape) + 1
#     # pred: [B, N, V, H, W] because there are N ensemble members
#     # y: [B, V, H, W]
#     rmse_dict = lat_weighted_rmse(pred.mean(dim=1), y, vars, lat, chosen_vars)
    
#     H = pred.shape[-2]
    
#     # lattitude weights
#     w_lat = np.cos(np.deg2rad(lat))
#     w_lat = w_lat / w_lat.mean()
#     w_lat = torch.from_numpy(w_lat).to(dtype=pred.dtype, device=pred.device) # (H, )    
    
#     var = torch.var(pred, dim=1) # [B, V, H, W]
#     var = var * w_lat.view(1, 1, H, 1) # [B, V, H, W]
#     spread = var.mean(dim=(-2, -1)).sqrt().mean(dim=0) # [V]
    
#     chosen_vars = chosen_vars or vars
#     loss_dict = {}
#     for i, variable in enumerate(vars):
#         if variable in chosen_vars:
#             loss_dict[f"w_ssr_{variable}{postfix}"] = spread[i] / rmse_dict[f"w_rmse_{variable}"]
        
#     return loss_dict

# def lat_weighted_acc(pred, y, vars, lat, clim):
#     """
#     y: [B, V, H, W]
#     pred: [B V, H, W]
#     vars: list of variable names
#     lat: H
#     """

#     # lattitude weights
#     w_lat = np.cos(np.deg2rad(lat))
#     w_lat = w_lat / w_lat.mean()  # (H, )
#     w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

#     # clim = torch.mean(y, dim=(0, 1), keepdim=True)
#     clim = clim.to(device=y.device).unsqueeze(0)
#     pred = pred - clim
#     y = y - clim
#     loss_dict = {}

#     with torch.no_grad():
#         for i, var in enumerate(vars):
#             pred_prime = pred[:, i] - torch.mean(pred[:, i])
#             y_prime = y[:, i] - torch.mean(y[:, i])
#             loss_dict[f"acc_{var}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
#                 torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
#             )

#     return loss_dict