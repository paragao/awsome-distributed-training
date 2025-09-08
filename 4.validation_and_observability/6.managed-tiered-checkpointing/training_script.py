import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.checkpoint import async_save, load
from amzn_sagemaker_checkpointing.config.sagemaker_checkpoint_config import SageMakerCheckpointConfig
from amzn_sagemaker_checkpointing.checkpointing.filesystem.filesystem import (
    SageMakerTieredStorageWriter,
    SageMakerTieredStorageReader
)

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def create_model():
    """Create and wrap model with DDP"""
    model = nn.Linear(1000, 10).cuda()
    return DDP(model, device_ids=[dist.get_rank()])

def main():
    setup_distributed()
    
    # Model and optimizer setup
    model = create_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # Checkpoint configuration
    checkpoint_config = SageMakerCheckpointConfig(
        namespace=os.environ.get('TRAINING_JOB_NAME', f'job-{int(time.time())}'),
        world_size=dist.get_world_size(),
        s3_tier_base_path="s3://my-training-bucket/checkpoints",
    )
    
    # Resume from checkpoint if available
    start_step = 0
    try:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict(),
            "step": 0
        }
        
        storage_reader = SageMakerTieredStorageReader(checkpoint_config=checkpoint_config)
        load(state_dict, storage_reader=storage_reader)
        
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        start_step = state_dict["step"] + 1
        
        print(f"Resumed training from step {start_step}")
    except BaseException as e:
        print(f"No checkpoint found, starting from scratch: {str(e)}")
    
    # Training loop
    in_memory_ckpt_freq = 10
    s3_ckpt_freq = 50
    future = None
    for step in range(start_step, 1000):
        # Training step
        optimizer.zero_grad()
        
        # Dummy forward pass (replace with your actual training logic)
        inputs = torch.randn(32, 1000).cuda()
        targets = torch.randint(0, 10, (32,)).cuda()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        
        # Save checkpoint
        if (step % in_memory_ckpt_freq == 0 or
            step % s3_ckpt_freq == 0):
            state_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "step": step
            }
            
            # Configure is S3 save is required for the step
            checkpoint_config.save_to_s3 = step % s3_ckpt_freq == 0

            # Create storage writer for current step
            storage_writer = SageMakerTieredStorageWriter(
                checkpoint_config=checkpoint_config,
                step=step
            )
            
            # Optional: wait for previous checkpoint
            if future is not None:
                exc = future.exception()
                if exc:
                    print(f"Failure in saving previous checkpoint: {str(exc)}")
                    # Handle failures as required
                else:
                    result = future.result()
                    # Process results from save, if required
            
            # Async save (non-blocking)
            future = async_save(state_dict=state_dict, storage_writer=storage_writer)
            
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
