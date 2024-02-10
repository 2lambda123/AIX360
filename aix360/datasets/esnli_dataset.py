import os

from functools import lru_cache
from typing import Dict

# helper function that returns a specific
# sentence pair example from the e-SNLI dataset
@lru_cache(maxsize=120)
def _example(file: str, id: str) -> Dict:
    """Function:
    Returns the dictionary object for the given id from the provided file.
    Parameters:
        - file (str): Path to the file containing JSON objects.
        - id (str): The id of the desired object.
    Returns:
        - Dict: The dictionary object with the given id.
    Processing Logic:
        - Read the file line by line.
        - Check if the line is empty.
        - Convert the line to a dictionary object.
        - Check if the dictionary object has the desired id.
        - If found, return the dictionary object.
        - If not found, raise a RuntimeError."""
    
    import json
    with open(file, 'r', encoding='utf-8') as f:
        while True:
            try:
                line = f.readline(5_000_000).strip()

                if line == '':
                    raise EOFError

                d = json.loads(line)
                if d['docid'] == id:
                    return d
            except EOFError:
                raise RuntimeError(f"example {id} not found")

class eSNLIDataset:
    """
    The e-SNLI dataset [#]_ contains pairs of sentences 
    each accompanied by human-rationale annotations 
    as to which words are in each pairs are most
    important for matching.

    The sentence pairs are from the Stanford Natural
    Language Inference dataset with labels that indicate
    if the sentence pair is a logical entailment,
    contradiction or neutral.

    References:
        .. [#] `Camburu, Oana-Maria, Tim Rocktäschel, Thomas Lukasiewicz, and Phil Blunsom, 
          “E-SNLI: Natural Language Inference with Natural Language Explanations.”,
          2018
          <https://arxiv.org/abs/1812.01193>`_
    """

    def __init__(self):
        """This function initializes the _dirpath and _cache_doc variables.
        Parameters:
            - None
        Returns:
            - None
        Processing Logic:
            - Sets _dirpath to the path of the esnli_data folder.
            - Sets _cache_doc to an empty dictionary.
            - Uses os.path.join to create the _dirpath.
            - Uses os.path.dirname and os.path.abspath to get the current directory path.
            - Uses '..', 'data', and 'esnli_data' to create the path to the esnli_data folder."""
        
        self._dirpath = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'data','esnli_data'
        )

        self._cache_doc = {}
        
    def get_example(self, example_id: str) -> Dict:
        """
        Return an e-SNLI example.

        The example_id indexes the "docs.jsonl" file of the downloaded dataset.

        Args:
            example_id (str): the example index.

        Returns:
            e-SNLI example in dictionary form.
        """
        return _example(
            os.path.join(
                self._dirpath,
                'docs.jsonl'
            ),
            example_id,
        )

