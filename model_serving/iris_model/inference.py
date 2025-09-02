from mlserver import MLModel, types
import pickle
import numpy as np

class RefModel(MLModel):

    async def load(self):
        print('model loaded')

        with open('model_serving/iris_model/model/model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        return await super().load()
       
    async def predict(self, payload: types.InferenceRequest) ->  types.InferenceResponse:
        iris = np.array(payload.inputs[0].data.root)
        iris = np.expand_dims(iris, axis=0)

        response = self.model.predict(iris)

        response = response.tolist()

        return types.InferenceResponse(
            id=payload.id,
            model_name=self.name,
            model_version=self.version,
            outputs=[
                types.ResponseOutput(
                    name="iris_response",
                    shape=[len(response)],
                    datatype="INT32",
                    data=[response],
                    parameters=types.Parameters(content_type="np"),
                )
            ],
        )
    
       
    
       
