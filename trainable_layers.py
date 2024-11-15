# freezing_layers.py

import re

def set_trainable_layers(base_model, unfreeze_from_stage=2, unfreeze_from_block_in_last_stage_freezed=1):
    """
    Freezes all layers in the base_model except for those from the specified stage onwards
    and unfreezes specific blocks in the last frozen stage.

    Parameters:
    - base_model: The pre-trained base model whose layers need to be frozen/unfrozen.
    - unfreeze_from_stage (int): The stage from which layers should be unfrozen.
    - unfreeze_from_block_in_last_stage_freezed (int): The block number in the last frozen stage to start unfreezing.

    Returns:
    - None: The function modifies the base_model layers in place.
    """
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze layers from the specified stage
    if unfreeze_from_stage is not None:
        for layer in base_model.layers:
            # Unfreeze downsampling block preceding the stage
            ds_match = re.search(r'downsampling_block_(\d+)', layer.name)
            if ds_match:
                ds_stage_num = int(ds_match.group(1))
                if ds_stage_num == unfreeze_from_stage - 1:
                    layer.trainable = True

            # Unfreeze stages from the specified stage onwards
            stage_match = re.search(r'stage_(\d+)', layer.name)
            if stage_match:
                stage_num = int(stage_match.group(1))
                if stage_num >= unfreeze_from_stage:
                    layer.trainable = True

    # Unfreeze layers from a specific block in the last frozen stage
    if unfreeze_from_block_in_last_stage_freezed is not None:
        last_frozen_stage = unfreeze_from_stage - 1
        # Unfreeze blocks from the specified block number in the last frozen stage
        for layer in base_model.layers:
            block_match = re.search(rf'stage_{last_frozen_stage}_block_(\d+)', layer.name)
            if block_match:
                block_num = int(block_match.group(1))
                if block_num >= unfreeze_from_block_in_last_stage_freezed:
                    layer.trainable = True

    # Optional: Print layers with their trainable status for debugging
    print("Layer Trainable Status:")
    for i, layer in enumerate(base_model.layers):
        trainable_status = 'yes' if layer.trainable else 'no'
        print(f"{i}: {layer.name} : trainable {trainable_status}")
