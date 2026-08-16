"""
Quick training test - run on 5 stocks for fast verification
"""

import sys
sys.path.insert(0, '/media/antar-chandra-nath/Media/Research/Dataset/scripts')

from train_baseline import BaselineMLTrainer

trainer = BaselineMLTrainer()
trainer.train_all_stocks(max_stocks=5)
