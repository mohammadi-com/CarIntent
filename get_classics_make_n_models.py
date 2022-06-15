import pandas as pd
import requests
import time

if __name__ == '__main__':
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
