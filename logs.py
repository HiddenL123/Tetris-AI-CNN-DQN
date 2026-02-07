from torch.utils.tensorboard import SummaryWriter


class CustomTensorBoard:
    def __init__(self, log_dir='./logs', **kwargs):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=log_dir)
    
    def log(self, step, stats: dict):
        """
        stats: dict with nested structure
        Example:
        stats = {
            'score': {'avg': 0.5, 'min': 0.0, 'max': 1.0},
            'q_value': {'avg': 0.3, 'min': 0.1, 'max': 0.7},
            'bellman': {'avg': 0.25, 'min': 0.05, 'max': 0.6},
            'epsilon': 0.1
        }
        """
        for category, values in stats.items():
            if isinstance(values, dict):
                # This will put min, avg, max on SAME graph for the category
                self.writer.add_scalars(category, values, step)
            else:
                # Single scalar
                self.writer.add_scalar(category, values, step)
    
    def close(self):
        self.writer.close()