from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
from nltk.sentiment import SentimentAnalyzer

app = Flask(__name__)
CORS(app)
sa = SentimentAnalyzer()


@app.route('/health')
def home():
    return 'Api is Healthy'


@app.route('/get_sentiment', methods=['POST'])
def run():
    try:
        comments = request.json.get('comments')
        if not comments:
            return jsonify({'message': 'comments not provided'}), 400

        overall_sentiment = 0

        for comment in comments:
            sentiment_score = sa.polarity_scores(comment)['compound']
            overall_sentiment += sentiment_score

        overall_sentiment /= len(comments)

        result = {
            'overall_sentiment': overall_sentiment,
            'comments_count': len(comments)
        }

        return jsonify(result)
    except Exception as e:
        return jsonify(str(e))


if __name__ == '__main__':
    p = int(os.getenv('PORT'))
    app.run(host='0.0.0.0', port=p)
