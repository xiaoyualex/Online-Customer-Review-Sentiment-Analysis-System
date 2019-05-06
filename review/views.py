from . import app
from .forms import FeatureForm

from flask import render_template

from models.review import ReviewModel


@app.route('/')
def index():
    return render_template('welcome.html')


@app.route('/form/', methods=('GET', 'POST'))
def form():
    myform = FeatureForm()

    if myform.is_submitted():
        line = myform.review_text.data
        review_model = ReviewModel()
        sentiment, hightwords = review_model.predict(line, highlight=True)

        return render_template('result.html',
                               line=line,
                               highlight_words=hightwords,
                               sentiment=sentiment)

    return render_template('form.html', form=myform)


@app.route('/result/')
def submit():
    return render_template('result.html')


@app.route('/about')
def about():
    return 'The about page'


@app.route('/author')
def author():
    return 'Put your name here'
