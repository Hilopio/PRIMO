import torch
import kornia.feature as KF
import time


def find_optimal_batch_size(max_batch=128, img_height=400, img_width=600, num_trials=10):
    device = torch.device('cuda:6')

    print(f"Using device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(6)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(6).total_memory / (1024**3):.2f} GB")

    model = KF.LoFTR(pretrained='outdoor').to(device).eval()

    results = []

    for batch_size in range(1, max_batch + 1):
        print(f"\nTesting batch size: {batch_size}")
        success = True
        avg_time = 0
        max_mem = 0

        for trial in range(num_trials):
            try:
                # Create dummy images: batch of pairs
                img0 = torch.rand(batch_size, 1, img_height, img_width, device=device)
                img1 = torch.rand(batch_size, 1, img_height, img_width, device=device)

                torch.cuda.reset_peak_memory_stats(6)

                start_time = time.time()
                with torch.no_grad():
                    output = model({'image0': img0, 'image1': img1})
                end_time = time.time()

                trial_time = end_time - start_time
                avg_time += trial_time

                trial_mem = torch.cuda.max_memory_allocated(6) / (1024**2)  # in MB
                max_mem = max(max_mem, trial_mem)

                # Clean up
                del img0, img1, output
                torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                print(f"Out of memory at batch size {batch_size}")
                success = False
                break

        if not success:
            break

        avg_time /= num_trials
        throughput = batch_size / avg_time

        results.append({
            'batch_size': batch_size,
            'avg_time': avg_time,
            'throughput': throughput,
            'max_memory_mb': max_mem
        })

        print(f"Avg time per batch: {avg_time:.4f} s")
        print(f"Throughput (pairs/sec): {throughput:.2f}")
        print(f"Max memory used: {max_mem:.2f} MB")

    if results:
        # Find the batch size with the highest throughput
        optimal = max(results, key=lambda x: x['throughput'])
        print("\nOptimal batch size based on throughput:")
        print(f"Batch size: {optimal['batch_size']}")
        print(f"Avg time: {optimal['avg_time']:.4f} s")
        print(f"Throughput: {optimal['throughput']:.2f} pairs/sec")
        print(f"Max memory: {optimal['max_memory_mb']:.2f} MB")
    else:
        print("No successful batch sizes found. Try smaller images or check GPU.")


# Run the function
find_optimal_batch_size(max_batch=64)  # Adjust max_batch if needed
