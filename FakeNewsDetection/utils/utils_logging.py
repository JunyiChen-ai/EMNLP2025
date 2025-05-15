import logging
import os
import sys
import time
import torch

class AverageMeter(object):
    """
    Utility class for computing and storing current value and average
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_logging(log_dir, filename="training.log"):
    """
    Initialize the logging system

    Args:
        log_dir (str): directory for log files
        filename (str): log file name
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, filename)
    
    # Remove all existing handlers
    root = logging.getLogger('')
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    
    # Configure log format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(file_handler)
    
    # Add console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
    # Set root logger level
    logging.getLogger('').setLevel(logging.INFO)
    
    print(f"Log file created at {log_file}")  # Print directly to ensure visibility
    logging.info(f"Log file created at {log_file}")


class CallBackLogging:
    """
    Callback for logging training process information
    """
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        """
        Initialize callback

        Args:
            frequent (int): logging frequency (in steps)
            total_step (int): total number of training steps
            batch_size (int): batch size
            start_step (int): starting step number
            writer: TensorBoard SummaryWriter
        """
        self.frequent = frequent
        self.total_step = total_step
        self.batch_size = batch_size
        self.start_step = start_step
        self.writer = writer
        
        self.init = False
        self.tic = 0
        
    def __call__(self, global_step, loss, acc, lr, epoch=None):
        """
        Execute callback

        Args:
            global_step (int): global training step
            loss (float): loss value
            acc (float): accuracy
            lr (float): learning rate
            epoch (int): current epoch
        """
        if global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * (self.total_step - global_step) / 3600
                except ZeroDivisionError:
                    speed = float("inf")
                    speed_total = float("inf")
                
                time_now = (time.time() - self.tic) / 3600
                time_total = time_now + speed_total
                
                msg = f"Epoch: {epoch}, Step: {global_step}, Speed: {speed:.2f} samples/sec, "
                msg += f"Time: {time_now:.2f}/{time_total:.2f} hours, "
                msg += f"Loss: {loss:.4f}, Acc: {acc:.4f}, LR: {lr:.6f}"
                
                logging.info(msg)
                
                if self.writer is not None:
                    self.writer.add_scalar('loss', loss, global_step)
                    self.writer.add_scalar('accuracy', acc, global_step)
                    self.writer.add_scalar('learning_rate', lr, global_step)
                
            self.tic = time.time()
            self.init = True


class CallBackModelCheckpoint:
    """
    Callback for saving model checkpoints
    """
    def __init__(self, output_dir, save_freq=1):
        """
        Initialize callback

        Args:
            output_dir (str): directory to save checkpoints
            save_freq (int): save frequency (in epochs)
        """
        self.output_dir = output_dir
        self.save_freq = save_freq
        
        os.makedirs(output_dir, exist_ok=True)
    
    def __call__(self, epoch, backbone, classifier, optimizer, scheduler, global_step):
        """
        Execute callback

        Args:
            epoch (int): current epoch
            backbone: backbone model
            classifier: classifier model
            optimizer: optimizer
            scheduler: learning rate scheduler
            global_step (int): global training step

        Returns:
            str: path to the saved checkpoint (or None)
        """
        if (epoch + 1) % self.save_freq == 0:
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_{epoch + 1}.pt")
            
            backbone_state = backbone.state_dict()
            if hasattr(backbone, 'module'):
                backbone_state = backbone.module.state_dict()
            
            classifier_state = classifier.state_dict()
            if hasattr(classifier, 'module'):
                classifier_state = classifier.module.state_dict()
            
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'backbone': backbone_state,
                'classifier': classifier_state,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None
            }, checkpoint_path)
            
            logging.info(f"Model checkpoint saved to {checkpoint_path}")
            
            return checkpoint_path
        
        return None
