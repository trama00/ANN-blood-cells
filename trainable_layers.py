import re

def set_trainable_layers(base_model, unfreeze_from_stage=2, unfreeze_from_block_in_last_stage_freezed=1):
    """
    Freezes all layers in the base_model except for those from the specified stage onwards
    and unfreezes specific blocks in the last frozen stage.
    Additionally, ensures that all BatchNormalization layers remain non-trainable.

    Parameters:
    - base_model: The pre-trained base model whose layers need to be frozen/unfrozen.
    - unfreeze_from_stage (int): The stage from which layers should be unfrozen.
    - unfreeze_from_block_in_last_stage_freezed (int): The block number in the last frozen stage to start unfreezing.

    Returns:
    - None: The function modifies the base_model layers in place.
    """
    # Step 1: Initially freeze all layers
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the layer by default

    # Step 2: Unfreeze layers based on the specified stage
    if unfreeze_from_stage is not None:
        for layer in base_model.layers:
            # Check for downsampling blocks preceding the specified stage
            ds_match = re.search(r'downsampling_block_(\d+)', layer.name)
            if ds_match:
                ds_stage_num = int(ds_match.group(1))
                if ds_stage_num == unfreeze_from_stage - 1:
                    layer.trainable = True  # Unfreeze the downsampling block

            # Check for stages from the specified stage onwards
            stage_match = re.search(r'stage_(\d+)', layer.name)
            if stage_match:
                stage_num = int(stage_match.group(1))
                if stage_num >= unfreeze_from_stage:
                    layer.trainable = True  # Unfreeze layers in these stages

    # Step 3: Unfreeze specific blocks in the last frozen stage
    if unfreeze_from_block_in_last_stage_freezed is not None:
        last_frozen_stage = unfreeze_from_stage - 1
        for layer in base_model.layers:
            block_match = re.search(rf'stage_{last_frozen_stage}_block_(\d+)', layer.name)
            if block_match:
                block_num = int(block_match.group(1))
                if block_num >= unfreeze_from_block_in_last_stage_freezed:
                    layer.trainable = True  # Unfreeze specific blocks

    # Step 4: Ensure all BatchNormalization layers are non-trainable
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # Override to keep BatchNorm layers non-trainable

    # Optional: Print layers with their trainable status for debugging
    print("\nLayer Trainable Status:")
    for i, layer in enumerate(base_model.layers):
        trainable_status = 'Yes' if layer.trainable else 'No'
        print(f"{i}: {layer.name} : Trainable = {trainable_status}")
