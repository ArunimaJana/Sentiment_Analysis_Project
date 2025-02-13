from django.http import JsonResponse
from django.shortcuts import render
from .utils import predict_sentiment
from . import utils

def sentiment_analysis_view(request):
    if request.method == "POST":
        review = request.POST.get("review")  # Get review from the form
        if not review:
            return JsonResponse({"error": "Review text is required."})
        
        # Predict sentiment
        sentiment = utils.predict_sentiment(review)
        return JsonResponse({"review": review, "sentiment": sentiment})

    return render(request, "sentiment_form.html")  # Render the input form
