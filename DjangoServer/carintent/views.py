from django.shortcuts import render
from deeppavlov.models.sklearn import SklearnComponent

# Create your views here.


def get_make_n_model(request):
    tfidf = SklearnComponent(
        model_class="sklearn.feature_extraction.text:TfidfVectorizer",
        infer_method="transform",
        save_path='./tfidf_v0.pkl',
        load_path='./tfidf_v0.pkl',
        mode='train')
    cls = SklearnComponent(
        model_class="sklearn.linear_model:LogisticRegression",
        infer_method="predict",
        save_path='./logreg_v0.pkl',
        load_path='./logreg_v0.pkl',
        C=1,
        mode='train')

    tfidf_model = SklearnComponent(
        model_class="sklearn.feature_extraction.text:TfidfVectorizer",
        infer_method="transform",
        save_path='./tfidf_v1.pkl',
        load_path='./tfidf_v1.pkl',
        mode='train')
    cls_model = SklearnComponent(
        model_class="sklearn.linear_model:LogisticRegression",
        infer_method="predict",
        save_path='./logreg_v1.pkl',
        load_path='./logreg_v1.pkl',
        C=1,
        mode='train')

    # print(request.GET.get('q'))
    query = (request.GET.get('q'),)
    make_predict = cls(tfidf(query))
    model_predict = cls_model(tfidf_model(query))

    return render(request, 'result.html', {'query': query[0], 'make': make_predict[0], 'model': model_predict[0]})
