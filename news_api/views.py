from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import joblib
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LR = joblib.load(os.path.join(BASE_DIR, "modelo_LR.pkl"))
DT = joblib.load(os.path.join(BASE_DIR, "modelo_DT.pkl"))
GBC = joblib.load(os.path.join(BASE_DIR, "modelo_GBC.pkl"))
RFC = joblib.load(os.path.join(BASE_DIR, "modelo_RFC.pkl"))
vectorization = joblib.load(os.path.join(BASE_DIR, "vectorizador.pkl"))

def wordopt(text):
    return text.lower()

def output_label(n):
    return "Real" if n == 1 else "Fake"

@csrf_exempt   # 👈 Esto desactiva la verificación CSRF para esta función
def predict_news(request):
    if request.method == "POST":
        import json
        body = json.loads(request.body)
        news_text = body.get("text", "")

        if not news_text:
            return JsonResponse({"error": "No text provided"}, status=400)

        testing_news = {"text": [news_text]}
        new_def_test = pd.DataFrame(testing_news)
        new_def_test["text"] = new_def_test["text"].apply(wordopt)
        new_xv_test = vectorization.transform(new_def_test["text"])

        results = {
            "logistic": output_label(LR.predict(new_xv_test)[0]),
            "decision_tree": output_label(DT.predict(new_xv_test)[0]),
            "gradient_boosting": output_label(GBC.predict(new_xv_test)[0]),
            "random_forest": output_label(RFC.predict(new_xv_test)[0]),
        }

        return JsonResponse({"results": results})

    return JsonResponse({"error": "Only POST method allowed"}, status=405)
