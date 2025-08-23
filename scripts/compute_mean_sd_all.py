import numpy as np
import os
import glob
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dinov2.data.datasets.webdataset import WebDatasetVisionPNG

ROOT_DIR = "/media/sharedaccess/Manthan_RSL_SSD/test_webdataset"

def process_shard_with_dataset(shard_file, worker_id, samples_per_shard=100):
    """Process a shard using WebDatasetVisionPNG class directly."""
    
    shard_name = os.path.basename(shard_file)
    dataset_name = os.path.basename(os.path.dirname(shard_file))
    
    print(f"🧵 Worker {worker_id}: Processing {samples_per_shard} samples from {dataset_name}/{shard_name}")
    
    try:
        # Create dataset with just this shard using your modified constructor
        dataset = WebDatasetVisionPNG(
            root="",  # Not needed since we're passing shard_files directly
            shard_files=[shard_file],  # Use your new parameter
            shuffle_buffer=1000,
        )
        
        print(f"🧵 Worker {worker_id}: Dataset created, getting iterator...")
        
        # Get iterator from the dataset
        dataset_iter = iter(dataset)
        
        # Initialize accumulators
        local_sum = np.zeros(3, dtype=np.float64)
        local_sum_sq = np.zeros(3, dtype=np.float64)
        local_pixels = 0
        processed_samples = 0
        
        # Process samples
        while processed_samples < samples_per_shard:
            try:
                # Get next sample using your dataset's processing
                image, target = next(dataset_iter)
                
                # Convert to numpy if needed
                if not isinstance(image, np.ndarray):
                    image = np.array(image)
                
                # Ensure float64 for precision
                if image.dtype != np.float64:
                    image = image.astype(np.float64)
                
                # Check shape
                if len(image.shape) == 3 and image.shape[-1] == 3:
                    # Accumulate statistics
                    h, w, c = image.shape
                    img_flat = image.reshape(-1, 3)
                    
                    local_sum += np.sum(img_flat, axis=0)
                    local_sum_sq += np.sum(img_flat ** 2, axis=0)
                    local_pixels += h * w
                    processed_samples += 1
                    
                    # Progress update
                    if processed_samples % 20 == 0:
                        print(f"🧵 Worker {worker_id}: {processed_samples}/{samples_per_shard} samples processed")
                
                else:
                    print(f"⚠️ Worker {worker_id}: Unexpected image shape {image.shape}")
                    continue
                
            except StopIteration:
                print(f"🧵 Worker {worker_id}: Dataset exhausted after {processed_samples} samples")
                break
            except Exception as e:
                print(f"⚠️ Worker {worker_id}: Error processing sample: {e}")
                continue
        
        result = {
            'worker_id': worker_id,
            'shard_file': shard_file,
            'dataset_name': dataset_name,
            'shard_name': shard_name,
            'processed_samples': processed_samples,
            'pixels': local_pixels,
            'sum': local_sum,
            'sum_sq': local_sum_sq
        }
        
        print(f"✅ Worker {worker_id}: {dataset_name}/{shard_name} - {processed_samples} samples, {local_pixels:,} pixels")
        return result
        
    except Exception as e:
        print(f"❌ Worker {worker_id}: Failed processing {shard_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_single_shard():
    """Test processing a single shard to verify everything works."""
    
    print("🧪 Testing single shard processing...")
    
    # Get one shard for testing
    dataset_list_file = os.path.join(ROOT_DIR, "webd_list.txt")
    
    if os.path.exists(dataset_list_file):
        with open(dataset_list_file, 'r') as f:
            dataset_folders = [line.strip() for line in f if line.strip()]
    else:
        dataset_folders = [d for d in os.listdir(ROOT_DIR) 
                          if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.endswith('_webd')]
    
    test_shard = None
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(ROOT_DIR, dataset_name)
        if os.path.exists(dataset_path):
            shard_pattern = os.path.join(dataset_path, "*.tar")
            shards = glob.glob(shard_pattern)
            if shards:
                test_shard = shards[0]
                break
    
    if not test_shard:
        print("❌ No test shard found!")
        return False
    
    print(f"🎯 Testing shard: {test_shard}")
    
    # Test with 10 samples
    result = process_shard_with_dataset(test_shard, 0, samples_per_shard=10)
    
    if result and result['processed_samples'] > 0:
        print(f"✅ Single shard test PASSED: {result['processed_samples']} samples processed")
        print(f"   Mean per channel: {result['sum'] / result['pixels']}")
        return True
    else:
        print("❌ Single shard test FAILED")
        return False

def get_all_shards():
    """Get all shard files from dataset."""
    
    dataset_list_file = os.path.join(ROOT_DIR, "webd_list.txt")
    
    if os.path.exists(dataset_list_file):
        with open(dataset_list_file, 'r') as f:
            dataset_folders = [line.strip() for line in f if line.strip()]
    else:
        dataset_folders = [d for d in os.listdir(ROOT_DIR) 
                          if os.path.isdir(os.path.join(ROOT_DIR, d)) and d.endswith('_webd')]
    
    all_shards = []
    dataset_shard_counts = {}
    
    for dataset_name in dataset_folders:
        dataset_path = os.path.join(ROOT_DIR, dataset_name)
        if os.path.exists(dataset_path):
            shard_pattern = os.path.join(dataset_path, "*.tar")
            shards = sorted(glob.glob(shard_pattern))
            all_shards.extend(shards)
            dataset_shard_counts[dataset_name] = len(shards)
            print(f"📁 {dataset_name}: {len(shards)} shards")
    
    return all_shards, dataset_shard_counts

def compute_mean_std_clean(samples_per_shard=100, max_workers=70):
    """
    Clean version using WebDatasetVisionPNG directly with shard_files parameter.
    """
    
    print(f"🚀 Computing mean/std using WebDatasetVisionPNG with direct shard access")
    print(f"🧵 Max workers: {max_workers}")
    print(f"🎯 Samples per shard: {samples_per_shard}")
    print("=" * 70)
    
    # Get all shards
    all_shards, dataset_shard_counts = get_all_shards()
    
    if not all_shards:
        print("❌ No shards found!")
        return None, None
    
    total_shards = len(all_shards)
    estimated_samples = total_shards * samples_per_shard
    
    print(f"\n📊 Processing Plan:")
    print(f"   📂 Total shards: {total_shards:,}")
    print(f"   📈 Estimated total samples: {estimated_samples:,}")
    print(f"   🧵 Workers: {min(max_workers, total_shards)}")
    
    # Show dataset breakdown
    print(f"\n📂 Dataset breakdown:")
    for dataset_name, shard_count in dataset_shard_counts.items():
        dataset_samples = shard_count * samples_per_shard
        percentage = (dataset_samples / estimated_samples) * 100
        print(f"   {dataset_name}: {shard_count} shards → {dataset_samples:,} samples ({percentage:.1f}%)")
    
    # Test single shard first
    print(f"\n🧪 Testing single shard first...")
    if not test_single_shard():
        print("❌ Single shard test failed - aborting")
        return None, None
    
    print(f"\n🚦 Single shard test passed! Proceeding with full processing...")
    input("Press Enter to continue or Ctrl+C to abort...")
    
    # Initialize accumulators
    total_sum = np.zeros(3, dtype=np.float64)
    total_sum_sq = np.zeros(3, dtype=np.float64)
    total_pixels = 0
    total_samples = 0
    dataset_stats = {name: {'samples': 0, 'pixels': 0, 'shards': 0} 
                    for name in dataset_shard_counts.keys()}
    
    # Start timing
    start_time = time.time()
    
    # Process in parallel
    num_workers = min(max_workers, total_shards)
    
    print(f"🔄 Starting processing with {num_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all jobs
        futures = {
            executor.submit(process_shard_with_dataset, shard, i, samples_per_shard): shard 
            for i, shard in enumerate(all_shards)
        }
        
        completed = 0
        failed = 0
        
        for future in as_completed(futures):
            try:
                result = future.result()
                if result and result['processed_samples'] > 0:
                    # Accumulate results
                    total_sum += result['sum']
                    total_sum_sq += result['sum_sq']
                    total_pixels += result['pixels']
                    total_samples += result['processed_samples']
                    
                    # Track per-dataset stats
                    dataset_name = result['dataset_name']
                    if dataset_name in dataset_stats:
                        dataset_stats[dataset_name]['samples'] += result['processed_samples']
                        dataset_stats[dataset_name]['pixels'] += result['pixels']
                        dataset_stats[dataset_name]['shards'] += 1
                
                completed += 1
                if completed % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    eta_hours = (total_shards - completed) / rate / 3600 if rate > 0 else 0
                    print(f"📊 Progress: {completed}/{total_shards} ({completed/total_shards*100:.1f}%) - "
                          f"Samples: {total_samples:,} - ETA: {eta_hours:.1f}h")
                
            except Exception as e:
                failed += 1
                print(f"❌ Worker error: {e}")
    
    # Final results
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print("=" * 70)
    
    if total_pixels == 0:
        print("❌ No pixels processed!")
        return None, None
    
    # Calculate statistics
    mean = total_sum / total_pixels
    mean_sq = total_sum_sq / total_pixels
    variance = mean_sq - (mean ** 2)
    std = np.sqrt(variance)
    
    # Print results
    print(f"📈 Total samples processed: {total_samples:,}")
    print(f"🔢 Total pixels processed: {total_pixels:,}")
    print(f"⏱️ Processing time: {elapsed_time:.2f} seconds ({elapsed_time/3600:.2f} hours)")
    print(f"✅ Completed shards: {completed}")
    print(f"❌ Failed shards: {failed}")
    
    print(f"\n🎯 DATASET STATISTICS:")
    print(f"📊 Mean: [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"📊 Std:  [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    
    print(f"\n🐍 PyTorch format:")
    print(f"mean = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]")
    print(f"std = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]")
    
    print(f"\n📋 For transforms.Normalize():")
    print(f"transforms.Normalize(mean={mean.tolist()}, std={std.tolist()})")
    
    # Per-dataset contributions
    print(f"\n📂 Per-dataset contributions:")
    for dataset_name, stats in dataset_stats.items():
        if stats['samples'] > 0:
            percentage = (stats['samples'] / total_samples) * 100
            avg_per_shard = stats['samples'] / stats['shards'] if stats['shards'] > 0 else 0
            print(f"   {dataset_name}: {stats['samples']:,} samples from {stats['shards']} shards "
                  f"({percentage:.1f}%, avg {avg_per_shard:.0f}/shard)")
    
    # Save results
    results = {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'total_samples': int(total_samples),
        'total_pixels': int(total_pixels),
        'samples_per_shard': samples_per_shard,
        'processing_time_hours': elapsed_time / 3600,
        'dataset_stats': dataset_stats
    }
    
    with open("clean_sampling_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to clean_sampling_stats.json")
    
    return mean, std

if __name__ == "__main__":
    print("📊 Clean Mean/Std Computation Using WebDatasetVisionPNG")
    print("=" * 70)
    
    # Configuration
    SAMPLES_PER_SHARD = 1000
    MAX_WORKERS = 16
    
    print(f"⚙️ Configuration:")
    print(f"   🎯 Samples per shard: {SAMPLES_PER_SHARD}")
    print(f"   🧵 Max workers: {MAX_WORKERS}")
    print(f"   🔧 Using WebDatasetVisionPNG with shard_files parameter")
    print()
    
    mean, std = compute_mean_std_clean(
        samples_per_shard=SAMPLES_PER_SHARD,
        max_workers=MAX_WORKERS
    )
    
    if mean is not None:
        print(f"\n🎉 Computation completed successfully!")
    else:
        print(f"\n❌ Computation failed")