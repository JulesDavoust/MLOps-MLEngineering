from flask import Flask, request, jsonify
from functions.intern import home

def intern_routes(app):
    @app.route('/', methods=['GET'])
    def get_home():
        return jsonify(home())
    