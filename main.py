from deeppavlov import build_model, configs
from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.metrics.accuracy import sets_accuracy
from deeppavlov.models.embedders.bow_embedder import BoWEmbedder
from deeppavlov.models.sklearn import SklearnComponent
from deeppavlov.models.preprocessors.str_lower import str_lower
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
import pandas as pd
import requests
from sklearn.tree import DecisionTreeClassifier
import time


if __name__ == '__main__' and False:
    df = pd.DataFrame(columns=['Keyword', 'Make', 'Model'])
    all_makes_response = requests.get(url='https://www.autotrader.com/collections/ccServices/rest/ccs/makes').json()


    for make in all_makes_response:
        models = requests.get(
            url='https://www.autotrader.com/collections/ccServices/rest/ccs/models?makeCode=' + make['makeCode']).json()

        # models = all_models[i]
        # i += 1

        for model in models:
            if model['label'] != 'Any Model':
                final_make = make['makeDescription'].lower()
                final_model = model['label'].replace(' - ', '').lower()
                df.loc[len(df)] = [final_model, final_make, final_model]
                df.loc[len(df)] = [final_make + ' ' + final_model, final_make, final_model]
                df.loc[len(df)] = [final_model + ' ' + final_make, final_make, final_model]
                # print('This is the make', final_make, 'and model', final_model)
                # time.sleep(10)
    # print(df)
    df.to_csv('./new_makes_n_models.csv')

if __name__ == '__main__' and False:
    df = pd.DataFrame(columns=['Keyword', 'Make', 'Model'])
    all_makes_response = requests.get(url='https://classics.autotrader.com/vehicles/simple_search_form.json').json()
    for make_groups in all_makes_response['data']['options']['make']:
        for make in make_groups['options']:
            models = requests.get(
                url='https://classics.autotrader.com/vehicles/simple_search_form.json?make=' + make).json()
            time.sleep(15)
            for model in models['data']['options']['model']:
                final_make = make_groups['options'][make].lower()
                final_model = model['label'].replace(' - ', '').lower()
                # print('This is the make', final_make, 'and model', final_model)
                df.loc[len(df)] = [final_model, final_make, final_model]
                df.loc[len(df)] = [final_make + ' ' + final_model, final_make, final_model]
                df.loc[len(df)] = [final_model + ' ' + final_make, final_make, final_model]
    # print(df)
    df.to_csv('./classics_makes_n_models.csv')
