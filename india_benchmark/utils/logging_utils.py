import os
import re
from datetime import datetime


def extract_latest_run_id(root_dir, logger_name):
    """
    Extract the ID of the latest wandb run based on the directory names.
    
    Args:
        root_dir (str): The root directory path
        logger_name (str): The name of the logger
        
    Returns:
        str: The ID of the latest run, or None if no runs found
    """
    wandb_dir = os.path.join(root_dir, logger_name, 'wandb')
    
    if not os.path.exists(wandb_dir):
        return None
    
    run_dirs = [
        d for d in os.listdir(wandb_dir) 
        if os.path.isdir(os.path.join(wandb_dir, d)) and d.startswith('run-')
    ]
    
    if not run_dirs:
        return None
    
    # Parse the directory names to extract date, time, and ID
    run_info = []
    pattern = r'run-(\d{8})_(\d{6})-(\w+)'
    
    for dir_name in run_dirs:
        match = re.match(pattern, dir_name)
        if match:
            date_str, time_str, run_id = match.groups()
            # Convert date and time strings to datetime object
            timestamp = datetime.strptime(f"{date_str}_{time_str}", "%Y%m%d_%H%M%S")
            run_info.append((timestamp, run_id, dir_name))
    
    if not run_info:
        return None
    
    # Sort by timestamp (latest first)
    run_info.sort(reverse=True)
    
    # Return the ID of the latest run
    return run_info[0][1]