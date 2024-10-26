import functools
import datetime
import numpy as np
import pandas as pd
import sklearn
import scipy
import scanpy as sc
import diptest
import matplotlib.pyplot as plt
import seaborn as sns

sc.settings.set_figure_params(dpi=128, facecolor="white")

class Ditu:
    
    def _log(
        self,
        message
    ):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time}] : {message}")
    
    def __init__(
        self,
        adata_transcript: sc.AnnData
    ):
        self._log('Initialization start...')
        self.adata_transcript = adata_transcript
        self._log('Merge transcripts to genes...')
        self.adata_gene_by_transcript = self.get_gene_by_transcript_adata()
        self._log('Preprocess gene data...')
        self.preprocess_adata(self.adata_gene_by_transcript)
        self.adata_transcript.obsm['X_umap'] = self.adata_gene_by_transcript.obsm['X_umap']
        self._log('Initialization completed.')
        return None
    
    def get_isoforms_of_gene(
        self,
        gene_name
    ):
        isoforms = self.adata_transcript[:,self.adata_transcript.var['gene_names'] == gene_name].var_names.tolist()
        return isoforms

    def get_transcript_expr(
        self,
        gene_name,
        filter_zero=False
    ):
        df = self.adata_transcript[:,self.adata_transcript.var['gene_names'] == gene_name].to_df()
        if filter_zero:
            df = df.loc[df.sum(axis=1) > 0]
        return df

    def get_transcript_abundance(
        self,
        gene_name,
        filter_na=False,
        min_cell_expr=0
    ):
        df = self.get_transcript_expr(gene_name, filter_zero=filter_na)
        if min_cell_expr > 0:
            df = df.loc[df.sum(axis=1) >= min_cell_expr]
        df = df.div(df.sum(axis=1), axis=0)
        return df

    def get_gene_by_transcript_adata(
        self
    ):
        df_transcript = self.adata_transcript.to_df()
        df_gene_by_transcript = df_transcript.T.groupby(
            self.adata_transcript.var['gene_names']
        ).sum().T
        adata_gene_by_transcript = sc.AnnData(df_gene_by_transcript)
        adata_gene_by_transcript.obs = self.adata_transcript.obs.loc[adata_gene_by_transcript.obs.index]
        return adata_gene_by_transcript
    
    def preprocess_adata(
        self,
        adata
    ):
        adata.layers["counts"] = adata.X.copy()
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
        sc.pp.pca(adata)
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        X_pca = adata.obsm['X_pca']
        PC_dis = sklearn.metrics.pairwise_distances(X_pca)
        adata.obsp['PC_distances'] = PC_dis
        return None

    def get_genes_with_isoforms(
        self,
        n_min_isoforms,
        min_nonzero_proportion=0,
    ):
        isoform_counts = self.adata_transcript.var.groupby('gene_names', observed=False).size()
        genes_with_isoforms = isoform_counts[isoform_counts >= n_min_isoforms].index.tolist()
        
        if min_nonzero_proportion > 0:
            n_nonzero_cells = (self.adata_gene_by_transcript.X > 0).sum(axis=0)
            cell_idx = n_nonzero_cells > (min_nonzero_proportion * self.adata_gene_by_transcript.shape[0])
            genes_nonzero = set(self.adata_gene_by_transcript.var.index[cell_idx])
            genes_with_isoforms = list(genes_nonzero.intersection(genes_with_isoforms))
            
        return genes_with_isoforms

    def get_knn_transcript_abundance(
        self,
        gene_name,
        n_neighbors
    ):
        knn_abundance_df = self.get_transcript_abundance(gene_name).dropna()
        adata_nonzero = self.adata_gene_by_transcript[knn_abundance_df.index]
        neighbors = adata_nonzero.obsp['PC_distances'].argsort()[:,:n_neighbors]
        # K-NN smoothing
        for isoform in knn_abundance_df.columns:
            # K-NN smoothing
            knn_abundance_df[isoform] = knn_abundance_df[isoform].to_numpy()[neighbors].mean(axis=1)
        return knn_abundance_df
    
    def _beta_kernel(
        self, 
        x, 
        x_i, 
        b
    ):
        def rho(x, b):
            return 2*(b**2) + 2.5 - np.sqrt(4*(b**4) + 6*(b**2) +2.25 - x**2 - x/b)
        if (0 <= x) and (x < 2*b):
            return scipy.stats.beta.pdf(x_i, rho(x, b), (1-x)/b)
        elif (2*b <= x) and (x <= 1-2*b):
            return scipy.stats.beta.pdf(x_i, x/b, (1-x)/b)
        elif (1-2*b < x) and (x <= 1):
            return scipy.stats.beta.pdf(x_i, x/b, rho(1-x, b))
        else:
            print(x)
    
    def beta_pdf(
        self,
        x,
        samples,
        b
    ):
        return sum(self._beta_kernel(x, samples, b)) / len(samples)
    
    def find_dtu_genes(
        self,
        n_min_isoforms=2,
        min_nonzero_proportion=0.05,
        n_neighbors=15,
        kde_kernel='beta',
        kde_bw=None,
    ):

        genes_with_isoforms = self.get_genes_with_isoforms(
            n_min_isoforms,
            min_nonzero_proportion
        )
        self._log(f'{len(genes_with_isoforms)} genes with nonzero proportion > {min_nonzero_proportion} and at least {n_min_isoforms} isoform found.')

        df_results = pd.DataFrame(columns = ['gene_names', 'dip', 'pval'])

        for i, gene_name in enumerate(genes_with_isoforms):

            if i % 100 == 0:
                self._log(f'Process {i}/{len(genes_with_isoforms)} genes...')

            knn_abundance_df = self.get_knn_transcript_abundance(
                gene_name,
                n_neighbors
            )
            
            for isoform in knn_abundance_df.columns:
                
                knn_abundance = knn_abundance_df[isoform].to_numpy()

                # Add noise
                noise = np.random.normal(0, 0.05, size=knn_abundance.shape)
                knn_abundance = knn_abundance + noise
                knn_abundance = knn_abundance.clip(0, 1)
                
                x_grids = np.linspace(0.001, 1-0.001, 1000)
                
                if kde_kernel == 'beta':
                    if kde_bw == None:
                        kde_bw = 0.02
                    kde = functools.partial(self.beta_pdf, samples=knn_abundance, b=kde_bw)
                    pdf = [kde(x) for x in x_grids]
                    kde_samples = np.random.choice(x_grids, size=1000, p=pdf/np.sum(pdf))
                
                elif kde_kernel == 'gaussian':
                    if kde_bw == None:
                        kde_bw = 'silverman'
                    kde = scipy.stats.gaussian_kde(knn_abundance, bw_method=kde_bw)
                    pdf = kde.pdf(x_grids)
                    kde_samples = np.random.choice(x_grids, size=2000, p=pdf/np.sum(pdf))
                
                dip, pval = diptest.diptest(kde_samples, n_threads=-1)
                
                df_results.loc[isoform] = [gene_name, dip, pval]
        
        df_results = df_results.sort_values(by=['pval', 'dip'], ascending=[True, False])

        return df_results

    
    def plot_gene_and_isoform_abundance(
        self,
        gene_name,
        reference_label=None
    ):
        sc.pl.umap(self.adata_gene_by_transcript, color=[gene_name, reference_label])
        df = self.get_transcript_abundance(gene_name)
        df = df.fillna(0)
        adata_transcript_abundance = sc.AnnData(df)
        adata_transcript_abundance.obs = self.adata_gene_by_transcript.obs.loc[adata_transcript_abundance.obs.index]
        adata_transcript_abundance.obsm['X_umap'] = self.adata_gene_by_transcript.obsm['X_umap']
        sc.pl.umap(adata_transcript_abundance, color=df.columns, cmap='coolwarm')
        return None
    
    def plot_abundance_dist(
        self,
        gene_name,
        knn=True,
        n_neighbors=15,
        kde=True,
        seed=42
    ):
        np.random.seed(seed)
        if knn:
            abundance_df = self.get_knn_transcript_abundance(gene_name, n_neighbors)
            color='lightgreen'
        else:
            abundance_df = self.get_transcript_abundance(gene_name).dropna()
            color='lightskyblue'
        num_isoforms = len(abundance_df.columns)
        fig, axes = plt.subplots(1, num_isoforms, figsize=(num_isoforms * 5, 5))
        if num_isoforms == 1:
            axes = [axes]

        for i, isoform in enumerate(abundance_df.columns):
            abundance = abundance_df[isoform]
            # Add noise
            if knn:
                noise = np.random.normal(0, 0.05, size=abundance.shape)
                abundance = abundance + noise
                abundance = abundance.clip(0, 1)
            x_grids = np.linspace(0.001, 1-0.001, 1000)
            beta_kde = functools.partial(self.beta_pdf, samples=abundance, b=0.02)
            pdf = [beta_kde(x) for x in x_grids]
            kde_samples = np.random.choice(x_grids, size=2000, p=pdf/np.sum(pdf))

            sns.histplot(abundance, binwidth=0.05, binrange=[0,1], stat='density', color=color, ax=axes[i])
            if kde:
                sns.lineplot(x=x_grids[5:-5], y=pdf[5:-5], color='red', linewidth=3, ax=axes[i])
            axes[i].grid(False)
            axes[i].set_title(f'{isoform}')
            axes[i].set_xlabel(f'Abundance\n\n')

        plt.tight_layout()
        plt.show()
        return None
