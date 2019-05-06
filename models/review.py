from .base import BaseModel


class ReviewModel(BaseModel):
    def __init__(self):
        super().__init__()

        self.load_vec('models/tfidf_vec.pkl')
        self.load_model('models/model.pkl')

    def predict(self, line, highlight=True):
        sentiment = super(ReviewModel, self).predict(line)

        # highlight words, hack
        if highlight:
            highlight_words = [w for w in self.preprocessing(line).split()
                               if super(ReviewModel, self).predict(w) == sentiment]
            return sentiment, highlight_words
        else:
            return sentiment
