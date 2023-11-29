from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch
import sys
import argparse
from datasets import load_dataset
from torch.utils.data import DataLoader, SequentialSampler
import time
import math
import numpy as np
from torchinfo import summary
from transformers import GPT2Model,AutoConfig


from transformers import pipeline
pipes=pipeline('text-classification')