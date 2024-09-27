import copy
from typing import Dict

from fastapi import APIRouter

from app.models import example as example_models
from src.tools.startup import params


route_settings = params['api']['example']

endpoint_settings = route_settings.pop('endpoints')
router = APIRouter(**route_settings)

@router.get(
    path="/get",
    response_model=example_models.OutputBody,
    responses={
        200: {"Description": "Example get method"}})
def example_get() -> example_models.OutputBody:
    """
    """
    # Call a pipeline or perform some action

    return {'message': 'GET called successfully'}

@router.post(
    path="/post",
    response_model=example_models.OutputBody,
    responses={
        200: {"Description": "Example post method"}})
def example_documents(body: example_models.InputBody) -> example_models.OutputBody:
    """

    """
    print(body.file)
    # Call a pipeline or perform some action

    return {'message': 'POST called successfully'}



