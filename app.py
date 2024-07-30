from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import joblib as jb
import uvicorn
import re

def stemming(content, stemmer):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [stemmer.stem(word) for word in stemmed_content if word not in set(stopwords.words('english'))]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# Define file paths for model, scaler and vectorizer
model_path = 'PrepareModel/decisiontree.sav'
scaler_path = 'PrepareModel/minmaxscaler.sav'
vectorizer_path = 'PrepareModel/vectorizer.sav'

# Load the model, scaler, and vectorizer from the file paths
DecisionTreeClassifier = jb.load(model_path)
scaler = jb.load(scaler_path)
vectorizer = jb.load(vectorizer_path)
stemmer = PorterStemmer()


# Create application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name = "static")
templates = Jinja2Templates(directory="templates")


# Route for the home page
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route for the predict page
@app.post("/predict")
async def predict(request: Request, Text: str = Form(...)):
    # Stemming content
    stemmed_Text = stemming(Text, stemmer)
    # Vectorize data
    vectorized_text = vectorizer.transform([stemmed_Text])
    # Make prediction using the loaded model
    prediction = DecisionTreeClassifier.predict(vectorized_text)[0]
    print("prediction", prediction)
    if prediction == 0:
        prediction = "Positive"
    else:
        prediction = "Negative"

    return templates.TemplateResponse("index.html", {"request": request, "prediction": prediction})


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
