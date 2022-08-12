"""Template file to develop personal app
WARNINGS: if you run the app locally and don't have a GPU you should choose device='cpu'
"""

from typing import Dict, List, Optional

import joblib
import torch
from biodatasets import load_dataset
from biotransformers import BioTransformers
from deepchain.components import DeepChainApp
from deepchain.models import MLP
from joblib import dump, load
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import load

import model_pca

Score = Dict[str, float]
ScoreList = List[Score]


class App(DeepChainApp):
    """DeepChain App template:

    * Implement score_names() and compute_score() methods.
    * Choose a transformer available on bio-transformers (or others pacakge)
    * Choose a personal keras/tensorflow model (or not)
    * Build model and load the weights.
    * compute whatever score of interest based on protein sequence
    """

    def __init__(self, device: str = "cuda:0"):
        self._device = device
        self.num_gpus = 0 if device == "cpu" else 1
        self.transformer = BioTransformers(backend="esm1_t34_670M_UR100", num_gpus=self.num_gpus)

        # TODO: fill _checkpoint_filename if needed
        # Make sure to put your checkpoint file in your_app/checkpoint folder
        self._checkpoint_filename: Optional[str] = "model.pt"

        self.model = MLP(input_shape=159, n_class=2)

        # TODO:  Use proper loading function
        # load_model for tensorflow/keras model - load for pytorch model
        # torch model must be built before loading state_dict
        if self._checkpoint_filename is not None:
            state_dict = load(self.get_checkpoint_path(__file__))
            self.model.load_state_dict(state_dict)
            self.model.eval()


    @staticmethod
    def score_names() -> List[str]:
        return ["antibiotic_resistance_probability"]
        
    def compute_scores(self, sequences: List[str]) -> ScoreList:
        """Compute a score based on a user defines function.

        This function compute a score for each sequences receive in the input list.
        Caution :  to load extra file, put it in src/ folder and use
                   self.get_filepath(__file__, "extra_file.ext")

        Returns:
            ScoreList object
            Score must be a list of dict:
                    * element of list is protein score
                    * key of dict are score_names
        """
               
        x_embedding = self.transformer.compute_embeddings(sequences)["cls"]
        x_normalized = StandardScaler().fit_transform(x_embedding)
        
        pca = model_pca.ModelPCA().pca
        x_lowdim = pca.transform(x_normalized)

        probabilities = self.model(torch.tensor(x_lowdim).float())
        probabilities = probabilities.detach().cpu().numpy()

        prob_list = [{self.score_names()[0]: prob[0]} for prob in probabilities]

        return prob_list

if __name__ == "__main__":

    sequences = [
        "MKNTLLKLGVCVSLLGITPFVSTISSVQAERTVEHKVIKNETGTISISQLNKNVWVHTELGYFSGEAVPSNGLVLNTSKGLVLVDSSWDDKLTKELIEMVEKKFKKRVTDVIITHAHADRIGGMKTLKERGIKAHSTALTAELAKKNGYEEPLGDLQSVTNLKFGNMKVETFYPGKGHTEDNIVVWLPQYQILAGGCLVKSASSKDLGNVADAYVNEWSTSIENVLKRYGNINLVVPGHGEVGDRGLLLHTLDLLK",
        "MFKTTLCALLITASCSTFAAPQQINDIVHRTITPLIEQQKIPGMAVAVIYQGKPYYFTWGYADIAKKQPV TQQTLFELGS VSKTFTGVLG GDAIARGEIK LSDPTTKYWP ELTAKQWNGITLLHLATYTA GGLPLQVPDE VKSSSDLLRF YQNWQPAWAP GTQRLYANSS IGLFGALAVKPSGLSFEQAM QTRVFQPLKL NHTWINVPPA EEKNYAWGYR EGKAVHVSPG ALDAEAYGVKSTIEDMARWV QSNLKPLDIN EKTLQQGIQL AQSRYWQTGD MYQGLGWEML DWPVNPDSIINGSDNKIALA ARPVKAITPP TPAVRASWVH KTGATGGFGS YVAFIPEKEL GIVMLANKNYPNPARVDAAW QILNALQ",
    ]
    app = App("cpu")
    scores = app.compute_scores(sequences)
    print(scores)
