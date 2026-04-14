from flask import Flask, request, render_template, jsonify
import logging
from search_engine import SearchEngine, AutocompleteService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
engine = SearchEngine()
autocomplete = AutocompleteService()

def create_app():
    app = Flask(__name__)
    @app.route('/')
    def index():
        return render_template('index.html')
    @app.route('/search')
    def search():
        try:
            query = request.args.get('ten_hang', '')
            results = engine.search(query)
            return render_template('results.html', hang_hoa=[r.to_dict() for r in results])
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return render_template('results.html', hang_hoa=[], error=str(e))
    
    @app.route('/get_suggestions')
    def suggestions():
        return jsonify(autocomplete.suggest(request.args.get('keyword', '')))
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
