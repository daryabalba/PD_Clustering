import pandas as pd
import gensim
import numpy as np
from itertools import permutations
from src.data_extraction import (DataExtractionBase, DataExtractionPDTexts)


class ClustersDataBase:

    def __init__(self,
                 extractor: DataExtractionBase,
                 model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        self.extractor = extractor
        self.model = model
        self.healthy_data = None
        self.impediment_data = None
        self.impediment_type = ''

    def get_df(self, sheet):
        pass

    def add_column(self,
                   sheet_name: str,
                   category: str,
                   clusters: pd.Series) -> None:
        pass

    @staticmethod
    def avg_cluster_size(row: pd.Series) -> float:
        """
        Get average cluster size in a row
        """
        clusters_sizes = []
        for cell in row:
            clusters_sizes.extend(len(cluster) for cluster in cell)
        return sum(clusters_sizes) / len(clusters_sizes)

    def avg_cluster_distance(self, cluster_sequence):
        """
        Count average cluster distance
        """
        if not cluster_sequence:
            return np.NaN

        centroids_dict = {}
        distances = []

        for cluster in cluster_sequence:
            centroid = sum(self.model[word] for word in cluster) / len(cluster)
            centroids_dict[tuple(cluster)] = centroid

        for idx in range(0, len(cluster_sequence)-1):
            cluster_1 = cluster_sequence[idx]
            cluster_2 = cluster_sequence[idx+1]
            Dij = np.dot(
                gensim.matutils.unitvec(centroids_dict[tuple(cluster_1)]),
                gensim.matutils.unitvec(centroids_dict[tuple(cluster_2)])
            )
            distances.append(Dij)

        if not distances:
            return np.NaN

        return sum(distances)/len(distances)

    def silhouette_score(self, cluster_sequence):
        silhouette_coefs = []

        for idx, cluster in enumerate(cluster_sequence):
            for word_1 in cluster:

                a = sum(self.model.similarity(word_1, word_2)
                        for word_2 in cluster if word_1 != word_2) / len(cluster)

                if idx != len(cluster_sequence) - 1:
                    b = sum(self.model.similarity(word_1, word_2)
                            for word_2 in cluster_sequence[idx + 1]) / len(cluster_sequence[idx + 1])
                else:
                    b = sum(self.model.similarity(word_1, word_2)
                            for word_2 in cluster_sequence[idx - 1]) / len(cluster_sequence[idx - 1])

                s = (b - a) / max(a, b)
                silhouette_coefs.append(s)

        if silhouette_coefs:
            return sum(silhouette_coefs) / len(silhouette_coefs)
        return np.NaN

    @staticmethod
    def cluster_t_score(f_n, f_c, f_nc, N):
        if f_nc == 0:
            return 0
        numerator = f_nc - f_n * f_c / N
        denominator = np.sqrt(f_nc)
        return numerator / denominator

    def avg_cluster_t_score(self, cell, column_clusters):
        all_words = ' '.join([word for cell in column_clusters for cluster in cell for word in cluster])
        N = len(all_words)

        cell_t_scores = []
        for cluster in cell:
            all_wordpairs = list(permutations(cluster, 2))

            pairwise_t_scores = []
            for wordpair in all_wordpairs:
                f_n = all_words.count(wordpair[0])
                f_c = all_words.count(wordpair[1])
                f_nc = all_words.count(' '.join((wordpair[0], wordpair[1])))
                f_nc += all_words.count(' '.join((wordpair[1], wordpair[0])))

                t_score = self.cluster_t_score(f_n, f_c, f_nc, N)
                pairwise_t_scores.append(t_score)

            cell_t_scores.extend(pairwise_t_scores)

        return sum(cell_t_scores)

    def save_excel(self, path) -> None:
        """
        Saving data with clusters to an Excel file
        """
        with pd.ExcelWriter(path) as writer:
            self.healthy_data.to_excel(writer, sheet_name='healthy', index=False)
            self.impediment_data.to_excel(writer, sheet_name=self.impediment_type, index=False)


class ClustersDataPDTexts(ClustersDataBase):

    def __init__(self,
                 extractor: DataExtractionPDTexts,
                 model: gensim.models.fasttext.FastTextKeyedVectors) -> None:
        super().__init__(extractor, model)
        self.id_healthy = extractor.get_ids('healthy')
        self.id_impediment = extractor.get_ids('general_massive')
        self.healthy_data = pd.DataFrame(self.id_healthy)
        self.impediment_data = pd.DataFrame(self.id_impediment)
        self.impediment_type = 'PD'

    def get_df(self, sheet):
        if sheet == 'healthy':
            return self.healthy_data
        return self.impediment_data

    def add_column(self,
                   sheet_name: str,
                   category: str,
                   clusters: pd.Series) -> None:
        """
        Adding a column with clusters
        """
        if sheet_name == 'healthy':
            self.healthy_data[category] = clusters

        else:
            self.impediment_data[category] = clusters

    def count_num_switches(self,
                           sheet_name: str,
                           category: str) -> None:
        """
        Count number of switches for each cell
        """
        if sheet_name == 'healthy':
            new_column_name = f'Switch_number_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(lambda x: len(x) - 1)

        else:
            new_column_name = f'Switch_number_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(lambda x: len(x) - 1)

    def count_mean_cluster_size(self,
                                sheet_name: str,
                                category: str) -> None:
        """
        Count mean cluster size for each row
        """
        if sheet_name == 'healthy':
            new_column_name = f'Mean_cluster_size_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(self.avg_cluster_size)

        else:
            new_column_name = f'Mean_cluster_size_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(self.avg_cluster_size)

    def count_mean_distances(self,
                             sheet_name: str,
                             category: str):
        """
        Counting distances for all columns
        """
        if sheet_name == 'healthy':
            new_column_name = f'Mean_distance_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(self.avg_cluster_distance)

        else:
            new_column_name = f'Mean_distance_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(self.avg_cluster_distance)

    def count_mean_silhouette_score(self,
                                    sheet_name: str,
                                    category: str):
        """
        Counting silhouette scores for all columns
        """
        if sheet_name == 'healthy':
            new_column_name = f'Silhouette_score_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(self.silhouette_score)

        else:
            new_column_name = f'Silhouette_score_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(self.silhouette_score)

    def count_cluster_t_scores(self,
                               sheet_name: str,
                               category: str):
        """
        Counting cluster t-scores for all columns
        """
        if sheet_name == 'healthy':
            new_column_name = f'Mean_cluster_t_score_{category}'
            self.healthy_data[new_column_name] = self.healthy_data[category].apply(
                lambda x: self.avg_cluster_t_score(x, self.healthy_data[category])
            )

        else:
            new_column_name = f'Mean_cluster_t_score_{category}'
            self.impediment_data[new_column_name] = self.impediment_data[category].apply(
                lambda x: self.avg_cluster_t_score(x, self.impediment_data[category])
            )
