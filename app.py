# for testing you can use : curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" --data-binary @file name.json -v
# Set your Google Cloud credentials environment variable 

from flask import Flask, request, jsonify # type: ignore
from google.cloud import aiplatform
from google.protobuf.struct_pb2 import Value # type: ignore
from google.protobuf import json_format # type: ignore
import os


app = Flask(__name__)


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/username/location/filename.json'

def predict_tabular_classification_sample(project, endpoint_id, location, instances):
    client_options = {"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    
    # Prepare instances by converting Python dicts to protobuf Value format
    proto_instances = [json_format.ParseDict(instance, Value()) for instance in instances]

    response = client.predict(endpoint=endpoint, instances=proto_instances)
    return response


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Define the path to the JSON file
        json_file_path = './input_data.json'  # Update this to your JSON file path

        print("HELLO")

        content = {}; # request.json
        project = "420975770009"
        endpoint_id = "3116321617349705728"
        location = "us-central1"
        instances = []; # content.get('instances', [])

        print("RESPONSE 2222222")

        response = predict_tabular_classification_sample(project, endpoint_id, location, instances)

        print("RESPONSE")
        print(response)

        predictions = [dict(prediction) for prediction in response.predictions]

        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)