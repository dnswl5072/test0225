"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import json
import os

import joblib
import numpy
from azureml.core.model import Model
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.schema_decorators import input_schema, output_schema


def init():
    # load the model from file into a global object
    global model

    # we assume that we have just one model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder
    # (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = Model.get_model_path(os.getenv("AZUREML_MODEL_DIR").split("/")[-2])

    model = joblib.load(model_path)


input_sample = numpy.array([[0, 0, 0, 1.1], [0.1, -0.1, -0.5, 0.5],])
output_sample = numpy.array([True, True])


# Inference_schema generates a schema for your web service
# It then creates an OpenAPI (Swagger) specification for the web service
# at http://<scoring_base_url>/swagger.json
@input_schema("data", NumpyParameterType(input_sample))
@output_schema(NumpyParameterType(output_sample))
def run(data, request_headers):
    result = model.predict(data)

    # Demonstrate how we can log custom data into the Application Insights
    # traces collection.
    # The 'X-Ms-Request-id' value is generated internally and can be used to
    # correlate a log entry with the Application Insights requests collection.
    # The HTTP 'traceparent' header may be set by the caller to implement
    # distributed tracing (per the W3C Trace Context proposed specification)
    # and can be used to correlate the request to external systems.
    log = json.dumps(
        {
            "RequestId": request_headers.get("X-Ms-Request-Id", ""),
            "TraceParent": request_headers.get("Traceparent", ""),
            "NumberOfPredictions": len(result),
        }
    )
    print(log)

    return {"result": result.tolist()}


if __name__ == "__main__":
    # Test scoring
    init()
    test_row = '{"data":[[1,2,3,4,5,6,7,8,9,10],[10,9,8,7,6,5,4,3,2,1]]}'
    prediction = run(test_row, {})
    print("Test result: ", prediction)
