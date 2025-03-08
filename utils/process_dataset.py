import pandas as pd
import logging
from tqdm import tqdm
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Configure logging
logger = logging.getLogger(__name__)

def is_valid_image_url(url, timeout=3, max_retries=1):
    """Check if the image URL is valid."""
    for attempt in range(max_retries):
        try:
            response = requests.head(url, allow_redirects=True, timeout=timeout)
            return response.status_code == 200
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.debug(f"URL validation failed for {url}: {str(e)}")
                return False
            time.sleep(1)  # Wait before retrying

def download_image(url, timeout=3, max_retries=1):
    """Download the image from the URL."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=timeout)
            if response.status_code == 200:
                return True
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                logger.debug(f"Image download failed for {url}: {str(e)}")
                return False
            time.sleep(1)  # Wait before retrying
    return False

def process_dataset(input_path, output_path, batch_size=100000, start_index=0, max_workers=8):
    """
    Process dataset by filtering out invalid image URLs and saving as CSV in batches.
    
    Args:
        input_path (str): Path to input TSV file
        output_path (str): Path to save processed CSV file
        batch_size (int): Number of instances to process in each batch
        start_index (int): Index to start processing from (for resuming)
        max_workers (int): Number of parallel workers for processing
    """
    try:
        # Read TSV file
        logger.info(f"Reading dataset from {input_path}")
        df = pd.read_csv(input_path, sep='\t', header=None, names=['caption', 'url'])
        
        # Skip already processed entries if resuming
        if start_index > 0:
            logger.info(f"Resuming from index {start_index}")
            df = df.iloc[start_index:]
        
        # Initialize counters
        total_count = len(df)
        valid_count = 0
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint file path
        checkpoint_path = output_dir / "checkpoint.txt"
        
        # Read checkpoint if exists
        if checkpoint_path.exists() and start_index == 0:
            try:
                with open(checkpoint_path, 'r') as f:
                    start_index = int(f.read().strip())
                logger.info(f"Resuming from checkpoint at index {start_index}")
            except Exception as e:
                logger.warning(f"Failed to read checkpoint: {e}. Starting from beginning")
        
        # Process in batches
        logger.info("Processing URLs in batches...")
        for start in range(0, total_count, batch_size):
            end = min(start + batch_size, total_count)
            batch = df.iloc[start:end]
            valid_entries = []
            
            # Process URLs in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, row in batch.iterrows():
                    url = row['url']
                    futures.append(executor.submit(
                        lambda r: (r, is_valid_image_url(r['url']) and download_image(r['url'])),
                        row
                    ))
                
                # Collect results
                valid_entries = []
                for future in tqdm(as_completed(futures), total=len(futures)):
                    row, is_valid = future.result()
                    if is_valid:
                        valid_entries.append(row)
                        valid_count += 1
                
                # Write batch of valid entries
                if valid_entries:
                    pd.DataFrame(valid_entries).to_csv(
                        output_path,
                        mode='a',
                        header=not Path(output_path).exists(),
                        index=False
                    )
            
            logger.info(f"Processed batch {start // batch_size + 1}: {len(valid_entries)} valid entries found.")
            
            # Update checkpoint
            try:
                with open(checkpoint_path, 'w') as f:
                    f.write(str(start_index + batch_size))
            except Exception as e:
                logger.warning(f"Failed to update checkpoint: {e}")
        
        # Log statistics with percentage as float
        completion_percentage = (float(valid_count) / total_count) * 100 if total_count > 0 else 0
        logger.info(f"Processing complete. {valid_count}/{total_count} URLs were valid ({completion_percentage:.4f}%).")
        logger.info(f"Removed {total_count - valid_count} invalid URLs")
        
        # Clean up checkpoint file
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint file removed - processing complete")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint file: {e}")
        
    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Define paths
    input_path = "uploads/Train_GCC-training.tsv"
    output_path = "uploads/Train_GCC-processed.csv"
    
    # Process dataset
    process_dataset(input_path, output_path)
