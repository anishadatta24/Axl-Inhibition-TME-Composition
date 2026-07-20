# reading in necessary packages
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import scanpy.external as sce
# import scvi  # need anndata downgrade=

# this is an unimportant one-liner that I usually use to standardize generated figures (makes them square with larger font)
sc.set_figure_params(figsize=(4, 4), fontsize=20)


# here I'm setting up the file paths to your data on the server. I have a main path (path), and then different
# extensions to that path depending on 1) which sample and 2) which donor we're reading in
path = '/Volumes/dallgglab/users/anidatta/241016Lau/240925Lau/'

# samples = ['10x-6909L/test1multi/', '10x-6939E/D24-16228/', '10x-6939E/D24-16229/', '10x-6939E/D24-16230/',
#            '10x-6939E/D24-16231/', '10x-6939E/D24-16232/', '10x-6939E/D24-16233/', '10x-6939E/D24-16234/',
#            '10x-6939E/D24-16235/']

# removing batch 1 from obj
samples = ['10x-6939E/D24-16228/', '10x-6939E/D24-16229/', '10x-6939E/D24-16230/',
           '10x-6939E/D24-16231/', '10x-6939E/D24-16232/', '10x-6939E/D24-16233/', '10x-6939E/D24-16234/',
           '10x-6939E/D24-16235/']

donors = ['A', 'B', 'C', 'D']

# I use this dictionary effectively as a patch to account for the fact that the four donors have different numbers
# appended to them depending on the sample
donor_dict = {'10x-6909L/test1multi/': ['A2', 'B2', 'C2', 'D2'],
              '10x-6939E/D24-16228/': ['A1', 'B1', 'C1', 'D1'],
              '10x-6939E/D24-16229/': ['A2', 'B2', 'C2', 'D2'],
              '10x-6939E/D24-16230/': ['A3', 'B3', 'C3', 'D3'],
              '10x-6939E/D24-16231/': ['A4', 'B4', 'C4', 'D4'],
              '10x-6939E/D24-16232/': ['A5', 'B5', 'C5', 'D5'],
              '10x-6939E/D24-16233/': ['A6', 'B6', 'C6', 'D6'],
              '10x-6939E/D24-16234/': ['A7', 'B7', 'C7', 'D7'],
              '10x-6939E/D24-16235/': ['A8', 'B8', 'C8', 'D8']}

# the following four dictionaries allow me to later add metadata to the combined object about 1) the sequencing
# batch, and 2) the culture condition, 3) injury status, and 4) treatment condition of each sample

batch_dict = {'10x-6909L/test1multi/': '1',
              '10x-6939E/D24-16228/': '2',
              '10x-6939E/D24-16229/': '2',
              '10x-6939E/D24-16230/': '2',
              '10x-6939E/D24-16231/': '2',
              '10x-6939E/D24-16232/': '2',
              '10x-6939E/D24-16233/': '2',
              '10x-6939E/D24-16234/': '2',
              '10x-6939E/D24-16235/': '2'}

culture_dict = {'10x-6909L/test1multi/': 'co', '10x-6939E/D24-16228/': 'co', '10x-6939E/D24-16229/': 'co',
                '10x-6939E/D24-16230/': 'co', '10x-6939E/D24-16231/': 'co', '10x-6939E/D24-16232/': 'tri',
                '10x-6939E/D24-16233/': 'tri', '10x-6939E/D24-16234/': 'tri', '10x-6939E/D24-16235/': 'tri'}

injury_dict = {'10x-6909L/test1multi/': 'N', '10x-6939E/D24-16228/': 'N', '10x-6939E/D24-16229/': 'N',
               '10x-6939E/D24-16230/': 'Y', '10x-6939E/D24-16231/': 'Y', '10x-6939E/D24-16232/': 'N',
               '10x-6939E/D24-16233/': 'N', '10x-6939E/D24-16234/': 'Y', '10x-6939E/D24-16235/': 'Y'}

treat_dict = {'10x-6909L/test1multi/': 'Bem', '10x-6939E/D24-16228/': 'DMSO', '10x-6939E/D24-16229/': 'Bem',
              '10x-6939E/D24-16230/': 'DMSO', '10x-6939E/D24-16231/': 'Bem', '10x-6939E/D24-16232/': 'DMSO',
              '10x-6939E/D24-16233/': 'Bem', '10x-6939E/D24-16234/': 'DMSO', '10x-6939E/D24-16235/': 'Bem'}

adatas = []

# we will use a double four loop to read in what is effectively 36 samples
# (9 samples (8 pools + pilot) x 4 donors/sample)

for sample in samples:
    adatas_donor = []
    for donor in donor_dict[sample]:
        # we read in using main path -> sample -> per sample outputs -> donor -> filtered data matrix
        adata = sc.read_10x_h5(path + sample + 'outs/per_sample_outs/' + donor + '/count/sample_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()  # make sure all gene names are unique
        adata.layers['counts'] = adata.X.copy()  # hanging on to mRNA counts that have not yet been normalized or logarithmized

        mito_genes = adata.var_names.str.startswith('MT-')
        # for each cell compute fraction of counts in mito genes vs. all genes
        # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
        adata.obs['percent_mito'] = adata[:, mito_genes].X.sum(axis=1).A1 / adata.X.sum(axis=1).A1
        # add the total counts per cell as observations-annotation to adata
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1

        # set upper bound based on upper bound by counts
        Q1 = adata.obs['n_counts'].quantile(0.25)
        Q3 = adata.obs['n_counts'].quantile(0.75)
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR
        print(upper_bound)

        # !! I have commented out all of the lines that generate QC plots !!

        # sc.pl.highest_expr_genes(adata, n_top=20)  #, save=True)
        # sc.pl.violin(adata, ['n_genes', 'n_counts', 'percent_mito'], multi_panel=True)  #, save='_QC_basic.pdf')
        # sc.pl.violin(adata, 'n_counts', log=True, cut=0)  #, save='_QC_counts.pdf')
        # sc.pl.scatter(adata, 'n_counts', 'n_genes', color='percent_mito')  #, save='_QC_countsvgenes.pdf')
        # sc.pl.violin(adata, 'percent_mito')
        # plt.axhline(0.055, color='orange')
        # # plt.savefig(fig_dir + 'violin_percentmito.pdf'), plt.close()

        # plt.figure()
        # sb.distplot(adata.obs['n_counts'], kde=False, bins=60)
        # # plt.axvline(filt_param[sample]['min_counts'])
        # # plt.axvline(filt_param[sample]['max_counts'])
        # # plt.savefig(fig_dir + 'distribution_n_counts.pdf'), plt.close()
        #
        # plt.figure()
        # p3 = sb.distplot(adata.obs['n_genes'], kde=False, bins=60)
        # plt.axvline(filt_param[sample]['min_genes'])
        # plt.axvline(filt_param[sample]['max_genes'])
        # # plt.savefig(fig_dir + 'distribution_n_genes.pdf'), plt.close()

        # Filter cells
        print('Filter sample: {}, donor: {}'.format(sample, donor))
        print('Number of cells before filters: {:d}'.format(adata.n_obs))
        sc.pp.filter_cells(adata, min_counts=2000)
        sc.pp.filter_cells(adata, max_counts=upper_bound)
        adata = adata[adata.obs['percent_mito'] < 0.1, :]
        sc.pp.filter_cells(adata, min_genes=300)
        # sc.pp.filter_cells(adata, max_genes=2500)
        print('Number of cells after filters: {:d}'.format(adata.n_obs))

        print('Normalizing...')
        # using standard norm for now - can use scran with factors if I feel that's appropriate
        sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e6)

        # Replot QC parameters after normalization
        # adata.obs['n_counts'] = adata.X.sum(axis=1).A1
        # adata.obs['n_genes'] = np.sum(adata.X > 0, axis=1).A1
        # sc.pl.scatter(adata, 'n_counts', 'n_genes', color='percent_mito')  #, save='_QC_postnorm_countsvsgenes.pdf')
        # sc.pl.scatter(adata, 'size_factors', 'n_counts')  #, save='_QC_postnorm_sizefactorscounts.pdf')
        # sc.pl.scatter(adata, 'size_factors', 'n_genes')  #, save='_QC_postnorm_sizefactorsgenes.pdf')

        adatas_donor.append(adata.copy())

    del adata

    # lumping donors together per sample
    adata_sample = adatas_donor[0].concatenate(adatas_donor[1:], batch_key='donor', batch_categories=donors)

    del adatas_donor
    adatas.append(adata_sample.copy())

# lumping all samples together into one object
adata = adatas[0].concatenate(adatas[1:], batch_key='sample', batch_categories=samples)
del adatas

# Filter genes after concatenation - again, I don't do any gene filtering here, but we can
print('Number of genes before filtering: {:d}'.format(adata.n_vars))
# sc.pp.filter_genes(adata, min_cells=10)
# print('Number of genes after filter: {:d}'.format(adata.n_vars))

# here will print a breakdown of total number of single cells per sample (for the 9 samples in total
print('Final dataset:')
print(adata.obs['sample'].value_counts())

# here we're adding metadata about each sample using the dictionaries defined above (lines 39-59)
adata.obs['batch'] = adata.obs['sample'].map(batch_dict)
adata.obs['culture'] = adata.obs['sample'].map(culture_dict)
adata.obs['injured'] = adata.obs['sample'].map(injury_dict)
adata.obs['treat'] = adata.obs['sample'].map(treat_dict)

# let's create a metadata slot which tells us both seq batch + donor (to be used for batch correction)
adata.obs['comb-batch'] = ['s' + adata.obs['batch'][i] + 'd' + adata.obs['donor'][i] for i in adata.obs.index]

# Log-normalize the data
sc.pp.log1p(adata)

########################################################################################################################
# General metrics of data quality
########################################################################################################################
X = adata.layers['counts']

# Get mean read count per cell
# X1 = np.expm1(X)
print(X.sum(axis=1).A1.mean())

# Get mean number of detected genes
X2 = X > 0
print(X2.sum(axis=1).A1.mean())

# initial UMAP viz (with and without batch correction)
# find 5000 highly variable genes using information about the seq batches+donors to try to mitigate technical variation
sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor="cell_ranger", batch_key='comb-batch')

# to generate a UMAP embedding, we first 1) do PCA on the data (denoises, and only uses highly variable genes by default)
sc.tl.pca(adata)

# 2 step UMAP embedding -> 1) find cell neighbors in lower dimensional (PCA) space and 2) use this to calculate UMAP embeddings
sc.pp.neighbors(adata)
sc.tl.umap(adata)

# plotting the data with UMAP shows that we definitely have some seq batch- and donor-specific variability to corect
sc.pl.umap(adata, color=['batch', 'donor'])

# writing unintegrated object to file for easy reference
adata.write('/Users/katebridges/Downloads/20250214-ADatta-unintegrated-batch2only.h5ad')

# reading from checkpoint
adata = sc.read('/Users/katebridges/Downloads/20250214-ADatta-unintegrated-batch2only.h5ad')

# removal of batch effects across samples
# let's use a more robust integration method b/c of more complex experimental design - I tried Harmony initially
# (which you can see is commented out) but I think scVI might be a more appropriate choice for this dataset
sce.pp.harmony_integrate(adata, 'comb-batch')
#
# # recompute PCA, UMAP
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.pl.umap(adata, color=['comb-batch', 'donor'])

adata.write('/Users/katebridges/Downloads/20250828-ADatta-Harmony-integrated-batch2only.h5ad')

adata = sc.read('/Users/katebridges/Downloads/20250203-ADatta-Harmony-integrated.h5ad')

# limit to consensus highly variable genes
adata_hvg = adata[:, adata.var["highly_variable"]].copy()

sce.pp.harmony_integrate(adata_hvg, 'comb-batch')
adata_hvg.obsm['X_pca'] = adata_hvg.obsm['X_pca_harmony']
sc.pp.neighbors(adata_hvg)
sc.tl.umap(adata_hvg)
sc.pl.umap(adata_hvg, color=['comb-batch', 'donor'])
adata_hvg.write('/Users/katebridges/Downloads/20250214-ADatta-Harmony-integrated-HVGonly-batch2only.h5ad')

#
# #
# # # recompute PCA, UMAP
# adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']
# sc.pp.neighbors(adata)
# sc.tl.umap(adata)
# sc.pl.umap(adata, color=['comb-batch', 'donor'])
#
# # see documentation: https://www.sc-best-practices.org/cellular_structure/integration.html starting at 12.5.1
# # for more in depth explanation of each step below
# adata_scvi = adata_hvg.copy()
# scvi.model.SCVI.setup_anndata(adata_scvi, layer="counts", batch_key='comb-batch')
# model_scvi = scvi.model.SCVI(adata_scvi)
#
# # setting recommended min epochs
# max_epochs_scvi = np.min([round((20000 / adata.n_obs) * 400), 400])
#
# # this training *can* be run locally, but it'll probably take ~1-2 hours
# model_scvi.train()
#
# # get scVI-corrected lower dimensional embeddings to use for corrected UMAP embedding
# adata_scvi.obsm["X_scVI"] = model_scvi.get_latent_representation()
# sc.pp.neighbors(adata_scvi, use_rep="X_scVI")
# sc.tl.umap(adata_scvi)
# sc.pl.umap(adata_scvi, color=['comb-batch', 'batch'])
#
# # write to file
# # fixing
# adata_scvi.obs['treat'] = adata_scvi.obs['sample'].map(treat_dict)
# adata_scvi.write('/Users/katebridges/Downloads/20241114-ADatta-scVI-integrated-version3.h5ad')
