import pandas as pd


class DataExtractionBase:

    def __init__(self, link: str) -> None:
        self.category_types = None

    def get_ids(self, sheet_name: str):
        pass

    def get_column_name(self, category: str):
        pass

    def get_series(self, sheet_name: str, category: str):
        pass


class DataExtractionPDTexts(DataExtractionBase):

    def __init__(self, link: str) -> None:
        super().__init__(link)
        self.dataset_norm = pd.read_excel(link, sheet_name='healthy')
        self.dataset_pd = pd.read_excel(link, sheet_name='general_massive')
        self.category_types = ['tokens', 'tokens_without_stops', 'lemmas', 'lemmas_without_stops']

    def get_ids(self, sheet_name: str = 'healthy') -> int:
        """
        Getting ID column
        """
        if sheet_name == 'healthy':
            return self.dataset_norm['speakerID']
        return self.dataset_pd['ID']

    def get_series(self,
                   sheet_name: str,
                   category: str) -> pd.DataFrame:
        """
        Getting one of 8 columns:
          from one of the 2 pages of the dataset
          from one of the 4 categories

        sheet_name: healthy | PD
        category: tokens | tokens_without_stops | lemmas | lemmas_without_stops
        """
        if sheet_name == 'healthy':
            return self.dataset_norm[category]

        return self.dataset_pd[category]
