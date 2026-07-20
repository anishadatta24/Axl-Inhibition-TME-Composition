# Upload the integrated h5ad file to Google Colab and run the following code
%pip install scanpy

import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scanpy.external as sce
import csv


adata = sc.read('/content/20250214-ADatta-Harmony-integrated-HVGonly-batch2only.h5ad')

X = adata.layers['counts']
X = pd.DataFrame(X.todense())
X_transposed = X.transpose()
X_transposed.to_csv('01_counts_transposed.csv')
# Download the csv file and complete the conversion of the h5ad file to a Seurat object with 01_Convert_h5ad_Seurat_Object.Rmd