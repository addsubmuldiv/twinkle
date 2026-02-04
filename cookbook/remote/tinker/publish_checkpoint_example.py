"""
Example: Publish a checkpoint to the hub using Tinker server

This example demonstrates how to use the publish_checkpoint endpoint
to upload training checkpoints to the hub via the Tinker-compatible server.
"""
import requests
import os

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.environ.get('MODELSCOPE_SDK_TOKEN')
HEADERS = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-Type': 'application/json'
}

def list_training_runs():
    """List all training runs for the current user."""
    response = requests.get(
        f"{BASE_URL}/api/v1/training_runs",
        headers=HEADERS
    )
    response.raise_for_status()
    return response.json()

def list_checkpoints(run_id):
    """List checkpoints for a specific training run."""
    response = requests.get(
        f"{BASE_URL}/api/v1/training_runs/{run_id}/checkpoints",
        headers=HEADERS
    )
    response.raise_for_status()
    return response.json()

def publish_checkpoint(run_id, checkpoint_id):
    """Publish a checkpoint to the hub."""
    response = requests.post(
        f"{BASE_URL}/api/v1/training_runs/{run_id}/checkpoints/{checkpoint_id}/publish",
        headers=HEADERS,
        json={}  # No request body required
    )
    response.raise_for_status()
    return response.status_code == 204

def main():
    print("Listing training runs...")
    runs_response = list_training_runs()
    
    if not runs_response.get('training_runs'):
        print("No training runs found.")
        return
    
    # Select the first run
    run = runs_response['training_runs'][0]
    run_id = run['training_run_id']
    print(f"\nSelected training run: {run_id}")
    print(f"  Base model: {run['base_model']}")
    print(f"  Is LoRA: {run['is_lora']}")
    
    # List checkpoints for the selected run
    print(f"\nListing checkpoints for run {run_id}...")
    checkpoints_response = list_checkpoints(run_id)
    checkpoints = checkpoints_response.get('checkpoints', [])
    
    if not checkpoints:
        print("No checkpoints found for this run.")
        return
    
    # Select the latest checkpoint
    checkpoint = checkpoints[-1]
    checkpoint_id = checkpoint['checkpoint_id']
    print(f"\nSelected checkpoint: {checkpoint_id}")
    print(f"  Type: {checkpoint['checkpoint_type']}")
    print(f"  Time: {checkpoint['time']}")
    print(f"  Path: {checkpoint.get('tinker_path', 'N/A')}")
    
    # Publish the checkpoint to the hub
    print(f"\nPublishing checkpoint to hub...")
    try:
        success = publish_checkpoint(run_id, checkpoint_id)
        
        if success:
            print("✓ Checkpoint publish request submitted successfully!")
            print("  The upload is happening asynchronously in the background.")
            checkpoint_name = checkpoint_id.split('/')[-1]
            print(f"  Hub model ID will be: {{username}}/{run_id}_{checkpoint_name}")
        else:
            print("✗ Failed to publish checkpoint.")
    except requests.exceptions.HTTPError as e:
        print(f"✗ Error publishing checkpoint: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == '__main__':
    main()
