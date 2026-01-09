import dataclasses
import torch

def rollout_with_gpu(
        model,
        batch,
        steps,
    ):
    """Perform a roll-out to make long-term predictions.
    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = model(batch)
        yield pred
        batch = dataclasses.replace(
            pred,
            surf_vars = {
                k: torch.cat([batch.surf_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars = {
                k: torch.cat([batch.atmos_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.atmos_vars.items()
            },
        )

def rollout_with_gpu_toy(
        model,
        batch,
        steps,
        ground_truth_list=None,  # Pass _label_list here
    ):
    """
    Perform a roll-out. If ground_truth_list is provided, use it for 
    Teacher Forcing (updating history with actual data).
    """
    # 1. Init
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(model.patch_size)
    batch = batch.to(p.device)

    # 2. Loop
    # Change '_' to 'i' so we can use the index to find the matching label
    for i in range(steps):
        pred = model(batch)
        yield pred
        
        # --- PREPARE INPUT FOR NEXT STEP ---
        
        if ground_truth_list is not None:
            # === TEACHER FORCING MODE ===
            
            # Step 1: Get the ground truth for the CURRENT step 'i'
            # (If i=0, we just predicted T+1. We pick the label for T+1 
            #  to serve as history for the next prediction.)
            gt_step = ground_truth_list[i] 

            batch = dataclasses.replace(
                pred, # Template (Metadata from pred)
                surf_vars={
                    # Access gt_step directly. It is already the dict for this timestamp.
                    k: torch.cat([batch.surf_vars[k][:, 1:], gt_step["surf_vars"][k]], dim=1)
                    for k in batch.surf_vars.keys()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], gt_step["atmos_vars"][k]], dim=1)
                    for k in batch.atmos_vars.keys()
                },
            )

        else:
            # === AUTOREGRESSIVE MODE (Original) ===
            # Use the model's own prediction 'pred'
            
            batch = dataclasses.replace(
                pred,
                surf_vars={
                    k: torch.cat([batch.surf_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.surf_vars.items()
                },
                atmos_vars={
                    k: torch.cat([batch.atmos_vars[k][:, 1:], v], dim=1)
                    for k, v in pred.atmos_vars.items()
                },
            )


def rollout_with_multiple_gpu(
        model,
        unwrap_model,
        batch,
        steps,
    ):
    """Perform a roll-out to make long-term predictions.
    Args:
        model (:class:`aurora.model.aurora.Aurora`): The model to roll out.
        unwrap_model (:class:`aurora.model.aurora.Aurora`): The unwrapped model to access attributes.
        batch (:class:`aurora.batch.Batch`): The batch to start the roll-out from.
        steps (int): The number of roll-out steps.
    Yields:
        :class:`aurora.batch.Batch`: The prediction after every step.
    """
    batch = unwrap_model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype)
    batch = batch.crop(unwrap_model.patch_size)
    batch = batch.to(p.device)

    for _ in range(steps):
        pred = model(batch)
        yield pred
        batch = dataclasses.replace(
            pred,
            surf_vars = {
                k: torch.cat([batch.surf_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.surf_vars.items()
            },
            atmos_vars = {
                k: torch.cat([batch.atmos_vars[k][:, 1 :], v], dim = 1)
                for k, v in pred.atmos_vars.items()
            },
        )