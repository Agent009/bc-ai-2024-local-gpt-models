#import sys
#from lib.sentiment import run_sa
#from lib.generation import run_gt
from lib.local import run_local

# ---------- Main script
if __name__ == '__main__':
    # ---------- Sentiment analysis
    #run_sa()

    # ---------- Text generation
    #run_gt()

    # ---------- Local completions API
    run_local()