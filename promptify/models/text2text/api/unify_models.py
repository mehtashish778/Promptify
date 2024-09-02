from promptify.parser.parser import Parser
from promptify.models.text2text.api.base_model import Model
from typing import Optional, Dict
from unify.clients import Unify as UnifyClient




class UnifyModel(Model):
    name = "Unify"
    description = "Unify API"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_tokens= 1000,
        api_wait=60,
        api_retry=6,
        json_depth_limit: int = 20,
        **kwargs,
    ):
        self.endpoint = endpoint
        self.model = model
        self.provider = provider
        self.json_depth_limit = json_depth_limit
        self.max_tokens = max_tokens

        # Initialize the base Model class
        super().__init__(api_key, model, api_wait, api_retry)
        
        self.additional_params = kwargs
        self._verify_model()  # Verify the model configuration
        self._initialize_parser()
        self._get_client()  # Initialize the Unify client

        
    def get_description(self):
        return self.description

    def get_endpoint(self, endpoint: str, model: str , provider: str):
        if not self.model and not self.provider:
            return self.endpoint
        
    def supported_models(self):
        pass
    
    def get_parameters(self) -> Dict[str, str]:
        return {
            "api_key": self.api_key,
            "endpoint": self.endpoint,
            "model": self.model,
            "provider": self.provider,
            **self.additional_params
        }
        
    
    def set_key(self, api_key: str):
        self.api_key = api_key
    
    def set_model(self, model: str, provider: str):
        if not self.endpoint:
            self.model = model
            self.provider = provider
            
                    
    def _verify_model(self):
        if self.endpoint and (self.model or self.provider):
            raise ValueError("If endpoint is provided, model and provider should not be specified.")
        if not self.endpoint and  not (self.model and self.provider):
            raise ValueError("If endpoint is not provided, both model and provider must be specified.")
        
    def _get_client(self):
        if self.endpoint:
            self._client = UnifyClient(api_key = self.api_key, endpoint=self.endpoint)
        else:
            self._client = UnifyClient(api_key=self.api_key, model=self.model, provider=self.provider)
        
    def _initialize_parser(self):
        self.parser = Parser()
    

    
    # def model_output_raw(self, prompt: str):
    #     data = {}
    #     raw_response = self._client.generate(prompt, max_tokens=self.max_tokens).strip(" \n")
    #     data['text'] = str(raw_response)
    #     return data
    
    
    # def model_output(self, prompt, json_depth_limit) -> Dict:
    #     data = self.model_output_raw(prompt)
    #     data["parsed"] = self.parser.fit(data["text"], json_depth_limit)
    #     return data

    # def run(self, prompt: str):
    #     parameters = self.get_parameters()  # Get the parameters as a dictionary
    #     parameters["prompt"] = prompt  # Add the prompt to the parameters
    #     response = self.model_output(prompt, self.json_depth_limit)  # Pass the prompt to the model_output method
    #     return response


    def model_output_raw(self, prompt: str):
        data = {}
        full_response = ""
        for chunk in self._client.generate(prompt):
            full_response += chunk
            yield {"text": full_response}
        data['text'] = full_response.strip(" \n")
        return data

    def model_output(self, prompt, json_depth_limit) -> Dict:
        data = {"text": "", "parsed": None}
        for chunk in self.model_output_raw(prompt):
            data["text"] = chunk["text"]
            data["parsed"] = self.parser.fit(data["text"], json_depth_limit)
            yield data

    def run(self, prompt: str):
        parameters = self.get_parameters()
        parameters["prompt"] = prompt
        for response in self.model_output(prompt, self.json_depth_limit):
            yield response
            
    
    
# # Define the API key for the OpenAI model
# api_key  = "1a2Yi8+xTGIsQ8bwxgSUhOvztnIhLPgJALzg5Ys98lI="


# # Create an instance of the OpenAI model, Currently supporting Openai's all model, In future adding more generative models from Hugginface and other platforms
# model = UnifyModel(api_key = api_key, model="llama-3-8b-chat", provider="fireworks-ai")

# print(model.run("what is the capital of france?"))