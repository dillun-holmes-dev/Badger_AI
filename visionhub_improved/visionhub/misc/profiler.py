import copy
import torch
from calflops import calculate_flops
from typing import Tuple

def stats(
    model,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    try:
        # Move to CPU to avoid CUDA compatibility issues with newer architectures
        model_for_info = copy.deepcopy(model).cpu()
        
        # Deploy mode conversion (fuses BatchNorm into Conv)
        if hasattr(model_for_info, 'deploy'):
            model_for_info = model_for_info.deploy()

        flops, macs, _ = calculate_flops(model=model_for_info,
                                            input_shape=input_shape,
                                            output_as_string=True,
                                            output_precision=4,
                                            print_detailed=False)
        params = sum(p.numel() for p in model_for_info.parameters())
        del model_for_info
        return {'flops': flops, 'macs': macs, 'params': params}
    
    except Exception as e:
        # If profiling fails (e.g., CUDA compatibility issues), return basic param count
        print(f"Warning: Model profiling failed ({type(e).__name__}: {e})")
        print("Returning parameter count only.")
        params = sum(p.numel() for p in model.parameters())
        return {'flops': 'N/A', 'macs': 'N/A', 'params': params}
