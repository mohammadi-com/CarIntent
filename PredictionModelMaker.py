from deeppavlov.core.data.simple_vocab import SimpleVocabulary
from deeppavlov.dataset_iterators.basic_classification_iterator import BasicClassificationDatasetIterator
from deeppavlov.dataset_readers.basic_classification_reader import BasicClassificationDatasetReader
from deeppavlov.models.embedders.bow_embedder import BoWEmbedder
from deeppavlov.models.sklearn import SklearnComponent
from deeppavlov.models.preprocessors.str_lower import str_lower
from deeppavlov.models.tokenizers.nltk_moses_tokenizer import NLTKMosesTokenizer
import pandas as pd
from time import sleep

if __name__ == '__main__' and False:
    # data = pd.read_csv('new_makes_n_models.csv')
    # X = data['Keyword']
    # y = data.drop(columns=['Keyword'])
    # model = DecisionTreeClassifier()
    # model.fit(X, y)
    # predictions = model.predict(['a3'])
    # print(predictions)
    # model = build_model(configs.classifiers.insults_kaggle_bert, download=True)
    # predictions = model(['hey, how are you?', 'You are so dumb!'])
    # print(predictions)
    # all_df = pd.concat([
    #     pd.read_csv('new_makes_n_models.csv'),
    #     pd.read_csv('classics_makes_n_models.csv'),
    #     pd.read_csv('./query_mapper/mapper.csv')
    # ])
    # print(all_df)
    # all_df.to_csv('all_makes_n_models.csv')

    dr = BasicClassificationDatasetReader().read(
        data_path='./',
        train='new_makes_n_models.csv',
        x='Keyword',
        y='Make'
    )


    # print([(k, len(dr[k])) for k in dr.keys()])
    # initialize data iterator splitting `train` field to `train` and `valid` in proportion 0.8/0.2
    train_iterator = BasicClassificationDatasetIterator(
        data=dr,
        field_to_split='train',  # field that will be splitted
        split_fields=['train', 'valid'],  # fields to which the fiald above will be splitted
        split_proportions=[0.8, 0.2],  # proportions for splitting
        split_seed=23,  # seed for splitting dataset
        seed=42)  # seed for iteration over dataset
    # x_train, y_train = train_iterator.get_instances(data_type='train')
    # for x, y in list(zip(x_train, y_train))[:5]:
    #     print('x:', x)
    #     print('y:', y)
    #     print('=================')
    # print(str_lower(['Is it freezing in Offerman, California?']))
    tokenizer = NLTKMosesTokenizer()
    # print(tokenizer(['Is it freezing in Offerman, California?']))
    train_x_lower_tokenized = str_lower(tokenizer(train_iterator.get_instances(data_type='train')[0]))
    classes_vocab = SimpleVocabulary(
        save_path='./classes.dict',
        load_path='./classes.dict')
    classes_vocab.fit((train_iterator.get_instances(data_type='train')[1]))
    classes_vocab.save()
    # print(list(classes_vocab.items()))
    token_vocab = SimpleVocabulary(
        save_path='./tokens.dict',
        load_path='./tokens.dict',
        min_freq=2,
        special_tokens=('<PAD>', '<UNK>',),
        unk_token='<UNK>')
    token_vocab.fit(train_x_lower_tokenized)
    token_vocab.save()
    # print(len(token_vocab))
    # print(token_vocab.freqs.most_common()[:10])
    bow = BoWEmbedder(depth=token_vocab.len)
    tfidf = SklearnComponent(
        model_class="sklearn.feature_extraction.text:TfidfVectorizer",
        infer_method="transform",
        save_path='./tfidf_v0.pkl',
        load_path='./tfidf_v0.pkl',
        mode='train')
    tfidf.fit(str_lower(train_iterator.get_instances(data_type='train')[0]))
    tfidf.save()
    # print(len(tfidf.model.vocabulary_))

    x_train, y_train = train_iterator.get_instances(data_type="train")
    x_valid, y_valid = train_iterator.get_instances(data_type="valid")

    cls = SklearnComponent(
        model_class="sklearn.linear_model:LogisticRegression",
        infer_method="predict",
        save_path='./logreg_v0.pkl',
        load_path='./logreg_v0.pkl',
        C=1,
        mode='train')

    # cls.fit(tfidf(x_train), y_train)
    # cls.save()
    # cls.load()

    x_valid = ('328i',)

    y_valid_pred = cls(tfidf(x_valid))
    print(y_valid_pred[0])
    # y_valid_pred = cls('a3')
    # print("Text sample: {}".format(x_valid[0]))
    # print("True label: {}".format(y_valid[0]))
    # print("Predicted label: {}".format(y_valid_pred[0]))




    # Model
if __name__ == '__main__':
    # data = pd.read_csv('new_makes_n_models.csv')
    # X = data['Keyword']
    # y = data.drop(columns=['Keyword'])
    # model = DecisionTreeClassifier()
    # model.fit(X, y)
    # predictions = model.predict(['a3'])
    # print(predictions)
    # model = build_model(configs.classifiers.insults_kaggle_bert, download=True)
    # predictions = model(['hey, how are you?', 'You are so dumb!'])
    # print(predictions)
    # all_df = pd.concat([
    #     pd.read_csv('new_makes_n_models.csv'),
    #     pd.read_csv('classics_makes_n_models.csv'),
    #     pd.read_csv('./query_mapper/mapper.csv')
    # ])
    # print(all_df)
    # all_df.to_csv('all_makes_n_models.csv')

    dr = BasicClassificationDatasetReader().read(
        data_path='./',
        train='new_makes_n_models.csv',
        x='Keyword',
        y='Model'
    )


    # print([(k, len(dr[k])) for k in dr.keys()])
    # initialize data iterator splitting `train` field to `train` and `valid` in proportion 0.8/0.2
    train_iterator = BasicClassificationDatasetIterator(
        data=dr,
        field_to_split='train',  # field that will be splitted
        split_fields=['train', 'valid'],  # fields to which the fiald above will be splitted
        split_proportions=[0.8, 0.2],  # proportions for splitting
        split_seed=23,  # seed for splitting dataset
        seed=42)  # seed for iteration over dataset
    x_train, y_train = train_iterator.get_instances(data_type='train')
    # for x, y in list(zip(x_train, y_train))[:5]:
    #     print('x:', x)
    #     print('y:', y)
    #     print('=================')
    # print(str_lower(['Is it freezing in Offerman, California?']))
    tokenizer = NLTKMosesTokenizer()
    # print(tokenizer(['Is it freezing in Offerman, California?']))
    train_x_lower_tokenized = str_lower(tokenizer(train_iterator.get_instances(data_type='train')[0]))
    classes_vocab = SimpleVocabulary(
        save_path='./classes.dict',
        load_path='./classes.dict')
    classes_vocab.fit((train_iterator.get_instances(data_type='train')[1]))
    classes_vocab.save()
    # print(list(classes_vocab.items()))
    token_vocab = SimpleVocabulary(
        save_path='./tokens.dict',
        load_path='./tokens.dict',
        min_freq=2,
        special_tokens=('<PAD>', '<UNK>',),
        unk_token='<UNK>')
    token_vocab.fit(train_x_lower_tokenized)
    token_vocab.save()
    # print(len(token_vocab))
    # print(token_vocab.freqs.most_common()[:10])
    bow = BoWEmbedder(depth=token_vocab.len)
    tfidf = SklearnComponent(
        model_class="sklearn.feature_extraction.text:TfidfVectorizer",
        infer_method="transform",
        save_path='./tfidf_v1.pkl',
        load_path='./tfidf_v1.pkl',
        mode='train')
    tfidf.fit(str_lower(train_iterator.get_instances(data_type='train')[0]))
    tfidf.save()
    # print(len(tfidf.model.vocabulary_))

    x_train, y_train = train_iterator.get_instances(data_type="train")
    x_valid, y_valid = train_iterator.get_instances(data_type="valid")

    cls = SklearnComponent(
        model_class="sklearn.linear_model:LogisticRegression",
        infer_method="predict",
        save_path='./logreg_v1.pkl',
        load_path='./logreg_v1.pkl',
        C=1,
        mode='train')

    cls.fit(tfidf(x_train), y_train)
    cls.save()
    # cls.load()

    x_valid = ('328i',)

    y_valid_pred = cls(tfidf(x_valid))
    print(y_valid_pred[0])
    # y_valid_pred = cls('a3')
    # print("Text sample: {}".format(x_valid[0]))
    # print("True label: {}".format(y_valid[0]))
    # print("Predicted label: {}".format(y_valid_pred[0]))
