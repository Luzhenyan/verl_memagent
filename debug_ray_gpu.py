import torch
import ray
import os

def check_cuda():
    try:
        ray.init(ignore_reinit_error=True)
        print(f"Main Process - CUDA available: {torch.cuda.is_available()}")
        print(f"Main Process - CUDA device count: {torch.cuda.device_count()}")
        print(f"Main Process - CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        print(f"Ray Cluster Resources: {ray.cluster_resources()}")

        @ray.remote(num_gpus=1)
        def worker_check():
            import torch
            import os
            return {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")
            }

        result = ray.get(worker_check.remote())
        print(f"Ray Worker Result: {result}")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    check_cuda()

