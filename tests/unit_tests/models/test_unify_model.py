import pytest
from unittest.mock import MagicMock, patch
from promptify.models.text2text.api.unify_models import UnifyModel




@pytest.fixture
def unify_model():
    # Creating a UnifyModel instance with mock data
    return UnifyModel(api_key="your_tapi_key", model="llama-3-8b-chat", provider="fireworks-ai")

def test_initialization_with_model_and_provider(unify_model):
    assert unify_model.api_key == "your_tapi_key"
    assert unify_model.model == "llama-3-8b-chat"
    assert unify_model.provider == "fireworks-ai"
    assert unify_model.endpoint is None

def test_initialization_with_endpoint():
    model = UnifyModel(api_key="your_tapi_key", endpoint="llama-3-8b-chat@fireworks-ai")
    assert model.api_key == "your_tapi_key"
    assert model.endpoint == "llama-3-8b-chat@fireworks-ai"
    assert model.model is None
    assert model.provider is None

def test_set_api_key(unify_model):
    unify_model.set_key("your_tapi_key")
    assert unify_model.api_key == "your_tapi_key"

def test_set_model_and_provider(unify_model):
    unify_model.set_model("gpt-3", "openai")
    assert unify_model.model == "gpt-3"
    assert unify_model.provider == "openai"

def test_verify_model_with_endpoint():
    model = UnifyModel(api_key="your_tapi_key", endpoint="llama-3-8b-chat@fireworks-ai")
    # No exceptions should be raised here
    model._verify_model()

def test_verify_model_with_model_and_provider():
    model = UnifyModel(api_key="your_tapi_key", model="llama-3-8b-chat", provider="fireworks-ai")
    # No exceptions should be raised here
    model._verify_model()

def test_invalid_model_configuration():
    with pytest.raises(ValueError):
        # Both endpoint and model/provider are provided, so it should raise ValueError
        UnifyModel(api_key="your_tapi_key", endpoint="llama-3-8b-chat@fireworks-ai", model="llama-3-8b-chat", provider="fireworks-ai")

def test_invalid_model_configuration_missing_info():
    with pytest.raises(ValueError):
        # Neither endpoint nor both model and provider are provided, so it should raise ValueError
        UnifyModel(api_key="your_tapi_key")

def test_get_client_with_endpoint():
    with patch('promptify.models.text2text.api.unify_models.UnifyClient') as MockUnifyClient:
        unify_model = UnifyModel(api_key="your_tapi_key", endpoint="llama-3-8b-chat@fireworks-ai")
        unify_model._get_client()
        
        # Ensure that the Unify client was called with the correct arguments
        MockUnifyClient.assert_called_with(api_key="your_tapi_key", endpoint="llama-3-8b-chat@fireworks-ai")


def test_run():
    # Initialize UnifyModel with mock data
    unify_model = UnifyModel(api_key="your_tapi_key", model="llama-3-8b-chat", provider="fireworks-ai")

    # Create a mock client and mock the generate method
    mock_client = MagicMock()
    mock_client.generate.return_value = ["Hello, world!"]

    # Set the mock client in the unify_model instance
    unify_model._client = mock_client

    # Mock the parser's fit method to return a parsed dictionary structure
    unify_model.parser.fit = MagicMock(return_value={"parsed": "mocked parsed output"})

    # Run the method with a test prompt and collect the results
    result = list(unify_model.run("Test prompt"))  # Collect all yielded results into a list

    # Assertions
    mock_client.generate.assert_called_once_with("Test prompt")  # Ensure the client generate method was called with the correct prompt

    # Check if the actual call to fit included the correct arguments
    unify_model.parser.fit.assert_called_once()
    call_args = unify_model.parser.fit.call_args[0]  # Get the actual arguments passed to fit
    assert call_args[0] == "Hello, world!"  # Ensure the first argument is as expected
    assert call_args[1] == 20  # Ensure the second argument is the json_depth_limit

    # The expected result should match the output structure that includes both "parsed" and "text"
    expected_result = {
        "parsed": {"parsed": "mocked parsed output"},
        "text": "Hello, world!"
    }

    # Since the `run` method yields, result will be a list of dictionaries; we check the first one
    assert result[-1] == expected_result  # Check that the last yielded result matches the expected parsed output
