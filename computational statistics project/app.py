import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from io import StringIO

app = Flask(__name__)

# PCA implementation from scratch
def pca_from_scratch(data, n_components=2):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Compute the covariance matrix
    cov_matrix = np.cov(scaled_data.T)
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    # Select the top n_components
    selected_vectors = eigenvectors[:, :n_components]
    
    # Project the data onto principal components
    pca_result = np.dot(scaled_data, selected_vectors)
    
    return pca_result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        num_components = int(request.form.get('num_components', 2))
        pca_result = pca_from_scratch(df, n_components=num_components)
        pca_result_2d = pca_result[:, :2] if num_components >= 2 else pca_result
        pca_result_3d = pca_result[:, :3] if num_components >= 3 else None
        return jsonify({
            'pca_data_2d': pca_result_2d.tolist(),
            'pca_data_3d': pca_result_3d.tolist() if pca_result_3d is not None else [],
            'num_components': num_components
        })
    return jsonify({'error': 'Invalid file format or missing file'}), 400

if __name__ == '__main__':
    app.run(debug=True)
