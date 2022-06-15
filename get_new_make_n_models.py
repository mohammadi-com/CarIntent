import pandas as pd
import requests

if __name__ == '__main__':
    df = pd.DataFrame(columns=['Keyword', 'Make', 'Model'])
    all_makes_response = requests.get(url='https://www.autotrader.com/collections/ccServices/rest/ccs/makes').json()

    for make in all_makes_response:
        models = requests.get(
            url='https://www.autotrader.com/collections/ccServices/rest/ccs/models?makeCode=' + make['makeCode']).json()

        for model in models:
            if model['label'] != 'Any Model':
                final_make = make['makeDescription'].lower()
                final_model = model['label'].replace(' - ', '').lower()
                df.loc[len(df)] = [final_model, final_make, final_model]
                df.loc[len(df)] = [final_make + ' ' + final_model, final_make, final_model]
                df.loc[len(df)] = [final_model + ' ' + final_make, final_make, final_model]
    df.to_csv('./new_makes_n_models.csv')
