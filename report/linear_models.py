import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score
import shap
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns