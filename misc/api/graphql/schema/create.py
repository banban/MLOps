from ariadne import make_executable_schema
from graphql.type.schema import GraphQLSchema

from schema.author import AUTHOR_TYPEDEF
from schema.category import CATEGORY_TYPEDEF
from schema.model import MODEL_TYPEDEF, model_query
from schema.types import MAIN_TYPEDEF, mutation, query


def create_schema() -> GraphQLSchema:
    return make_executable_schema(
        [MAIN_TYPEDEF, MODEL_TYPEDEF, AUTHOR_TYPEDEF, CATEGORY_TYPEDEF], [query, model_query, mutation]
    )
