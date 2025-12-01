import numpy as np
import evaluate 


GLUE_METRIC_DICT = {
    "cola": ["matthews_correlation"],
    "sst2": ["accuracy"],
    "qqp": ["accuracy", "f1"],
    "mrpc": ["accuracy", "f1"],
    "mnli": ["accuracy"],
    "qnli": ["accuracy"],
    "rte": ["accuracy"],
    "wnli": ["accuracy"]
}

class GlueMetric:
    def __init__(self, task_name: str):
        self.task_name = task_name
        self.metric = evaluate.load("glue", self.task_name)

    def add_batch(self, predictions, references):
        # Add batch predictions and references 
        self.metric.add_batch(predictions=predictions, references=references)

    def clear(self):
        # Load new metric to "clear out" old one
        self.metric = evaluate.load("glue", self.task_name)

    def compute(self):
        # Compute metric based on preds and labels in the current batch 
        result = self.metric.compute()
        assert result is not None
        return result