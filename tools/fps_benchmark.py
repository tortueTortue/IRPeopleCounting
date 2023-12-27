import numpy as np
import torch


def measure_fps(model, image_dimensions = [3, 224, 224], device: str = "cpu", repetitions: int = 10000, warmup_reps: int = 100):
    assert device.lower() in ["cpu", "cuda"], f"Wrong device, {device} is not cpu or cuda"
    device = torch.device(device.lower())
    model.to(device)
    im_dim = [1] + image_dimensions
    dummy_input = torch.randn(im_dim, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((repetitions, 1))
    #GPU-WARM-UP
    for _ in range(warmup_reps):
        _ = model(dummy_input)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    return {"Mean_Inference_Time" : mean_syn, "Std_Inference_time" : std_syn, "MEAN_FPS" : 1 / (mean_syn / 1000), "STD_FPS" : (std_syn / 1000)}