from numbers import Number
from typing import Optional, Dict, Any, Mapping
from .utils import TWINKLE_REQUEST_ID, get_base_url, get_api_key
import requests


def http_get(
    url: Optional[str]=None,
    params: Optional[Dict[str, Any]] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: int = 300,
) -> requests.Response:
    """
    Send HTTP GET request with required headers.

    Args:
        url: The target URL
        params: Query parameters
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    url = url or get_base_url()
    
    headers = {
        "X-Ray-Serve-Request-Id": TWINKLE_REQUEST_ID,
        "Authorization": 'Bearer ' + get_api_key(),
    }
    
    if additional_headers:
        headers.update(additional_headers)
    _params = {}
    for key, value in params.items():
        if hasattr(value, 'processor_id'):
            _params[key] = value.processor_id
        elif hasattr(value, '__dict__'):
            from twinkle.server.twinkle.serialize import serialize_object
            _params[key] = serialize_object(value)
        else:
            _params[key] = value
    response = requests.get(
        url,
        headers=headers,
        params=_params,
        timeout=timeout,
    )
    
    return response


def http_post(
    url: Optional[str]=None,
    json_data: Optional[Dict[str, Any]] = {},
    data: Optional[Any] = {},
    additional_headers: Optional[Dict[str, str]] = {},
    timeout: int = 300,
) -> requests.Response:
    """
    Send HTTP POST request with required headers.

    Args:
        url: The target URL
        json_data: JSON data to send in request body
        data: Form data or raw data to send in request body
        additional_headers: Additional headers to include
        timeout: Request timeout in seconds

    Returns:
        requests.Response object
    """
    url = url or get_base_url()
    
    headers = {
        "X-Ray-Serve-Request-Id": TWINKLE_REQUEST_ID,
        "Authorization": 'Bearer ' + get_api_key(),
    }
    
    if additional_headers:
        headers.update(additional_headers)
    _params = {}
    for key, value in json_data.items():
        if hasattr(value, 'processor_id'):
            _params[key] = value.processor_id
        elif hasattr(value, '__dict__'):
            from twinkle.server.twinkle.common.serialize import serialize_object
            _params[key] = serialize_object(value)
        else:
            _params[key] = value
    response = requests.post(
        url,
        headers=headers,
        json=_params,
        data=data,
        timeout=timeout,
    )
    
    return response
