import dataclasses
import torch

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