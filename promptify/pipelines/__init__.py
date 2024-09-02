from pathlib import Path
import types
from tqdm import tqdm
from typing import Any, Dict, Optional
from promptify.prompter.conversation_logger import *
from promptify.utils.data_utils import *
from promptify.prompter.prompt_cache import PromptCache
from promptify.models.text2text.api.unify_models import UnifyModel



class Pipeline:
    def __init__(self, prompter, model, structured_output=True, **kwargs):
        if not isinstance(prompter, list):
            prompter = [prompter]

        self.prompters = prompter
        self.model = model
        self.json_depth_limit: int = kwargs.get("json_depth_limit", 20)
        self.cache_prompt = kwargs.get("cache_prompt", True)
        self.cache_size = kwargs.get("cache_size", 200)
        self.prompt_cache = PromptCache(self.cache_size)
        self.conversation_path = kwargs.get("output_path", Path.cwd())
        self.structured_output = structured_output

        self.model_args_count = self.model.run.__code__.co_argcount
        self.model_variables = self.model.run.__code__.co_varnames[
            1 : self.model_args_count
        ]

        self.conversation_path = os.getcwd()
        self.model_dict = {
            key: value
            for key, value in model.__dict__.items()
            if is_string_or_digit(value)
        }

        if isinstance(self.model, UnifyModel) and self.model.endpoint:
            self.model_dict['model'] = self.model.endpoint

        self.logger = ConversationLogger(self.conversation_path, self.model_dict)



    def fit(self, text_input: str, **kwargs) -> Any:
        """
        Processes an input text through the pipeline: generates a prompt, gets a response from the model,
        caches the response, logs the conversation, and returns the output.
        """

        outputs_list = []
        for prompter in tqdm(self.prompters):
            try:
                template, variables_dict = prompter.generate(text_input, self.model, **kwargs)
            except ValueError as e:
                print(f"Error in generating prompt: {e}")
                return None

            if kwargs.get("verbose", False):
                print(template)

            output = self._get_output_from_cache_or_model(template)
            if output is None:
                return None

            # Handle generator output for UnifyModel
            if isinstance(self.model, UnifyModel):
                final_output = list(output)[-1] if isinstance(output, types.GeneratorType) else output
            else:
                final_output = output

            if "jinja" in prompter.template:
                prompt_name = prompter.template
            else:
                prompt_name = "Unknown"

            if self.structured_output:
                message = create_message(
                    template,
                    variables_dict,
                    final_output["text"] if isinstance(final_output, dict) else final_output,
                    final_output["parsed"]["data"]["completion"] if isinstance(final_output, dict) and final_output.get("parsed") else None,
                    prompt_name,
                )
            else:
                message = create_message(
                    template, variables_dict, final_output, None, prompt_name
                )

            self.logger.add_message(message)
            outputs_list.append(final_output)

        return outputs_list

    def _get_output_from_cache_or_model(self, template):
        output = None

        if self.cache_prompt:
            output = self.prompt_cache.get(template)

        if output is None:
            try:
                response = self.model.execute_with_retry(prompt=template)
                
                # Handle generator output for UnifyModel
                if isinstance(self.model, UnifyModel):
                    output = list(response)[-1] if isinstance(response, types.GeneratorType) else response
                else:
                    output = response

            except Exception as e:
                print(f"Error in model execution: {e}")
                return None

            if self.cache_prompt:
                self.prompt_cache.add(template, output)

        return output
