import numpy as np

def make_windows(X, y, input_width=48, horizon=24):
  """
    Returns:
      Xs: (N, input_width, n_features)
      ys: (N, horizon)
    """


  Xs, ys = [], []
  T = len(y)
  for start in range(0, T - input_width - horizon + 1):
        end_in  = start + input_width       
        end_out = end_in + horizon          
        Xs.append(X[start:end_in])          
        ys.append(y[end_in:end_out])        
  return np.array(Xs), np.array(ys)
