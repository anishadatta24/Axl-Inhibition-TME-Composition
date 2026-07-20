import cell2cell as c2c
data_folder = '/home/anidatta/data'
tensor = c2c.io.read_data.load_tensor(data_folder + '/03_Tensor_pre_decomp.pkl')
tensor = c2c.analysis.run_tensor_cell2cell_pipeline(tensor,tensor_metadata=None,
                                           copy_tensor=True,
                                           rank=None, # Number of factors to perform the factorization. If None, it is automatically determined by an elbow analysis
                                           tf_optimization='robust', # To define how robust we want the analysis to be.
                                           random_state=0, # Random seed for reproducibility
                                           device='cpu', # Device to use. If using GPU and PyTorch, use 'cuda'. For CPU use 'cpu'
                                           output_folder=None, # Whether to save the figures in files. If so, a folder pathname must be passed
                                          )
# Save tensor, copy to local machine, then do visualization + downstream analyses in R with 04_Tensor_cell2cell_Post_Decomposition.Rmd
c2c.io.export_variable_with_pickle(variable=tensor, filename=data_folder + '/04_Tensor_post_decomp.pkl')
